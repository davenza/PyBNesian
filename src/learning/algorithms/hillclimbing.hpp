#ifndef PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP

#include <indicators/cursor_control.hpp>
#include <dataset/dataset.hpp>
#include <learning/scores/scores.hpp>
#include <learning/operators/operators.hpp>
#include <util/progress.hpp>
#include <util/vector.hpp>

namespace py = pybind11; 

using dataset::DataFrame;
using learning::scores::Score;
using learning::operators::Operator, learning::operators::OperatorType, learning::operators::ArcOperator, 
      learning::operators::ChangeNodeType, learning::operators::OperatorTabuSet, learning::operators::OperatorPool;

using util::ArcStringVector;

namespace learning::algorithms {

    // TODO: Include start graph.
    py::object hc(const DataFrame& df, std::string bn_str, std::string score_str, std::vector<std::string> operators_str,
            ArcStringVector& arc_blacklist, ArcStringVector& arc_whitelist, FactorStringTypeVector& type_whitelist,
                  int max_indegree, int max_iters, double epsilon, int patience, int verbose = 0);

    class GreedyHillClimbing {
    public:
        template<typename Model>
        Model estimate(const DataFrame& df, 
                       OperatorPool& op_pool,
                       const Model& start,
                       ArcStringVector& arc_blacklist,
                       ArcStringVector& arc_whitelist,
                       int max_indegree,
                       int max_iters, 
                       double epsilon,
                       int verbose = 0);

        template<typename Model>
        Model estimate_validation(const DataFrame& df, 
                                 OperatorPool& op_pool, 
                                 Score& validation_score,
                                 const Model& start,
                                 ArcStringVector& arc_blacklist,
                                 ArcStringVector& arc_whitelist,
                                 FactorStringTypeVector& type_whitelist,
                                 int max_indegree,
                                 int max_iters,
                                 double epsilon, 
                                 int patience,
                                 int verbose = 0);
    };

    template<typename Model>
    Model GreedyHillClimbing::estimate(const DataFrame& df,
                                       OperatorPool& op,
                                       const Model& start,
                                       ArcStringVector& arc_blacklist,
                                       ArcStringVector& arc_whitelist,
                                       int max_indegree,
                                       int max_iters,
                                       double epsilon,
                                       int verbose) {
        indicators::show_console_cursor(false);
        auto spinner = util::indeterminate_spinner(verbose);
        spinner->update_status("Checking dataset...");

        Model::requires(df);

        auto current_model = start;
        current_model.check_blacklist(arc_blacklist);
        current_model.force_whitelist(arc_whitelist);

        op.set_arc_blacklist(arc_blacklist);
        op.set_arc_whitelist(arc_whitelist);
        op.set_max_indegree(max_indegree);

        spinner->update_status("Caching scores...");

        op.cache_scores(current_model);

        auto iter = 0;
        while(iter < max_iters) {
            auto best_op = op.find_max(current_model);

            if (!best_op || best_op->delta() <= epsilon) {
                break;
            }

            best_op->apply(current_model);

            op.update_scores(current_model, *best_op);
            ++iter;
            
            spinner->update_status(best_op->ToString());
        }

        spinner->mark_as_completed("Finished Hill-climbing!");
        indicators::show_console_cursor(true);
        return current_model;
    }

    template<typename Model>
    double validation_delta_score(const Model& model, const Score& val_score, 
                                  const Operator* op, VectorXd& current_local_scores) {
        switch(op->type()) {
            case OperatorType::ADD_ARC: {
                auto dwn_op = dynamic_cast<const ArcOperator*>(op);
                auto target_index = model.index(dwn_op->target());
                auto parents = model.parent_indices(target_index);
                auto source_index = model.index(dwn_op->source());
                parents.push_back(source_index);

                double prev = current_local_scores(target_index);
                current_local_scores(target_index) = val_score.local_score(model, target_index, parents.begin(), parents.end());

                return current_local_scores(target_index) - prev;
            }
            case OperatorType::REMOVE_ARC: {
                auto dwn_op = dynamic_cast<const ArcOperator*>(op);
                auto target_index = model.index(dwn_op->target());
                auto parents = model.parent_indices(target_index);
                auto source_index = model.index(dwn_op->source());

                util::swap_remove_v(parents, source_index);

                double prev = current_local_scores(target_index);
                current_local_scores(target_index) = val_score.local_score(model, target_index, parents.begin(), parents.end() - 1);

                return current_local_scores(target_index) - prev;
            }
            case OperatorType::FLIP_ARC: {
                auto dwn_op = dynamic_cast<const ArcOperator*>(op);
                auto target_index = model.index(dwn_op->target());
                auto target_parents = model.parent_indices(target_index);
                auto source_index = model.index(dwn_op->source());
                auto source_parents = model.parent_indices(source_index);

                util::swap_remove_v(target_parents, source_index);
                source_parents.push_back(target_index);

                double prev_source = current_local_scores(source_index);
                double prev_target = current_local_scores(target_index);
                current_local_scores(source_index) = val_score.local_score(model, source_index, source_parents.begin(), source_parents.end());
                current_local_scores(target_index) = val_score.local_score(model, target_index, target_parents.begin(), target_parents.end() - 1);

                return current_local_scores(source_index) +
                        current_local_scores(target_index) -
                        prev_source -
                        prev_target;
            }
            case OperatorType::CHANGE_NODE_TYPE: {
                auto dwn_op = dynamic_cast<const ChangeNodeType*>(op);
                auto node_index = model.index(dwn_op->node());
                auto new_node_type = dwn_op->node_type();
                auto parents = model.parent_indices(node_index);
                
                double prev = current_local_scores(node_index);
                current_local_scores(node_index) = val_score.local_score(new_node_type, node_index, parents.begin(), parents.end());
                return current_local_scores(node_index) - prev;
            }
            default:
                throw std::invalid_argument("Unreachable code. Wrong operator in HoldoutLikelihood::delta_score().");
        }
    }
    
    template<typename Model>
    Model GreedyHillClimbing::estimate_validation(const DataFrame& df,
                             OperatorPool& op_pool, 
                             Score& validation_score,
                             const Model& start,
                             ArcStringVector& arc_blacklist,
                             ArcStringVector& arc_whitelist,
                             FactorStringTypeVector& type_whitelist,
                             int max_indegree,
                             int max_iters,
                             double epsilon, 
                             int patience,
                             int verbose) {
        indicators::show_console_cursor(false);
        auto spinner = util::indeterminate_spinner(verbose);
        spinner->update_status("Checking dataset...");

        Model::requires(df);

        auto current_model = start;
        current_model.check_blacklist(arc_blacklist);
        current_model.force_whitelist(arc_whitelist);

        if constexpr(std::is_same_v<Model, SemiparametricBN>)
            current_model.force_type_whitelist(type_whitelist);

        op_pool.set_arc_blacklist(arc_blacklist);
        op_pool.set_arc_whitelist(arc_whitelist);
        op_pool.set_type_whitelist(type_whitelist);
        op_pool.set_max_indegree(max_indegree);

        
        auto best_model = start;

        spinner->update_status("Caching scores...");
        VectorXd local_validation(current_model.num_nodes());
        for (auto n = 0; n < current_model.num_nodes(); ++n) {
            local_validation(n) = validation_score.local_score(current_model, n);
        }

        op_pool.cache_scores(current_model);
        int p = 0;
        double validation_offset = 0;

        OperatorTabuSet tabu_set;
        
        auto iter = 0;
        while(iter < max_iters) {
            auto best_op = op_pool.find_max(current_model, tabu_set);
            if (!best_op || best_op->delta() <= epsilon) {
                break;
            }

            double validation_delta = validation_delta_score<Model>(current_model, validation_score, best_op.get(), local_validation);
            
            best_op->apply(current_model);
            if ((validation_delta + validation_offset) > 0) {
                p = 0;
                validation_offset = 0;
                best_model = current_model;
                tabu_set.clear();
            } else {
                if (++p >= patience)
                    break;
                validation_offset += validation_delta;
                tabu_set.insert(best_op->opposite());
            }


            op_pool.update_scores(current_model, *best_op);

            spinner->update_status(best_op->ToString() + " | Validation delta: " + std::to_string(validation_delta));
            ++iter;
        }

        spinner->mark_as_completed("Finished Hill-climbing!");
        indicators::show_console_cursor(true);
        return best_model;
    }
}

#endif //PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP