#ifndef PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP

#include <indicators/cursor_control.hpp>
#include <dataset/dataset.hpp>
#include <models/ConditionalBayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/operators/operators.hpp>
#include <util/progress.hpp>
#include <util/vector.hpp>

namespace py = pybind11; 

using dataset::DataFrame;
using models::ConditionalBayesianNetworkBase;
using learning::scores::Score;
using learning::operators::Operator, learning::operators::OperatorType, learning::operators::ArcOperator, 
      learning::operators::ChangeNodeType, learning::operators::OperatorTabuSet, learning::operators::OperatorSet,
      learning::operators::LocalScoreCache;

using util::ArcStringVector;

namespace learning::algorithms {

    // TODO: Include start graph.
    std::unique_ptr<BayesianNetworkBase> hc(const DataFrame& df, 
                                            const BayesianNetworkBase* start,
                                            const std::string& bn_str, 
                                            const std::optional<std::string>& score_str,
                                            const std::optional<std::vector<std::string>>& operators_str,
                                            const ArcStringVector& arc_blacklist,
                                            const ArcStringVector& arc_whitelist,
                                            const FactorStringTypeVector& type_whitelist,
                                            int max_indegree,
                                            int max_iters,
                                            double epsilon,
                                            int patience,
                                            std::optional<unsigned int> seed,
                                            int num_folds,
                                            double test_holdout_ratio,
                                            int verbose = 0);

    template<typename T>
    std::unique_ptr<T> estimate_hc(OperatorSet& op_set,
                                   Score& score,
                                   const T& start,
                                   const ArcSet& arc_blacklist,
                                   const ArcSet& arc_whitelist,
                                   int max_indegree,
                                   int max_iters,
                                   double epsilon,
                                   int verbose) {
        indicators::show_console_cursor(false);
        auto spinner = util::indeterminate_spinner(verbose);
        spinner->update_status("Checking dataset...");

        auto current_model = start.clone();
        current_model->check_blacklist(arc_blacklist);
        current_model->force_whitelist(arc_whitelist);

        op_set.set_arc_blacklist(arc_blacklist);
        op_set.set_arc_whitelist(arc_whitelist);
        op_set.set_max_indegree(max_indegree);

        spinner->update_status("Caching scores...");

        op_set.cache_scores(*current_model, score);

        auto iter = 0;
        while(iter < max_iters) {
            auto best_op = op_set.find_max(*current_model);
            if (!best_op || best_op->delta() <= epsilon) {
                break;
            }

            best_op->apply(*current_model);

            op_set.update_scores(*current_model, score, *best_op);
            ++iter;
            
            spinner->update_status(best_op->ToString());
        }

        spinner->mark_as_completed("Finished Hill-climbing!");
        indicators::show_console_cursor(true);
        return current_model;
    }

    template<typename T>
    double validation_delta_score(const T& model,
                                  const Score& val_score, 
                                  const Operator* op,
                                  LocalScoreCache& current_local_scores) {
        
        switch(op->type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto dwn_op = dynamic_cast<const ArcOperator*>(op);
                double prev = current_local_scores.local_score(model, dwn_op->target());
                current_local_scores.update_local_score(model, val_score, *op);
                return current_local_scores.local_score(model, dwn_op->target()) - prev;
            }
            case OperatorType::FLIP_ARC: {
                auto dwn_op = dynamic_cast<const ArcOperator*>(op);
                double prev = current_local_scores.local_score(model, dwn_op->source()) +
                              current_local_scores.local_score(model, dwn_op->target());
                current_local_scores.update_local_score(model, val_score, *op);
                
                return current_local_scores.local_score(model, dwn_op->source()) +
                       current_local_scores.local_score(model, dwn_op->target()) - prev;
            }
            case OperatorType::CHANGE_NODE_TYPE: {
                auto dwn_op = dynamic_cast<const ChangeNodeType*>(op);
                double prev = current_local_scores.local_score(model, dwn_op->node());
                current_local_scores.update_local_score(model, val_score, *op);
                return current_local_scores.local_score(model, dwn_op->node()) - prev;
            }
            default:
                throw std::invalid_argument("Unreachable code. Wrong operator in validation_delta_score().");
        }
    }
    
    template<typename T>
    std::unique_ptr<T> estimate_validation_hc(OperatorSet& op_set,
                                              Score& score,
                                              Score& validation_score,
                                              const T& start,
                                              const ArcSet& arc_blacklist,
                                              const ArcSet& arc_whitelist,
                                              const FactorStringTypeVector& type_whitelist,
                                              int max_indegree,
                                              int max_iters,
                                              double epsilon, 
                                              int patience,
                                              int verbose) {
        if (!util::compatible_score(start, validation_score.type())) {
            throw std::invalid_argument("Invalid score " + validation_score.ToString() + 
                                        " for model type " + start.type().ToString() + ".");
        }

        indicators::show_console_cursor(false);
        auto spinner = util::indeterminate_spinner(verbose);
        spinner->update_status("Checking dataset...");

        auto current_model = start.clone();
        current_model->check_blacklist(arc_blacklist);
        current_model->force_whitelist(arc_whitelist);

        if (current_model->type() == BayesianNetworkType::SPBN) {
            auto current_spbn = dynamic_cast<SemiparametricBN&>(*current_model);
            current_spbn.force_type_whitelist(type_whitelist);
        }

        op_set.set_arc_blacklist(arc_blacklist);
        op_set.set_arc_whitelist(arc_whitelist);
        op_set.set_type_whitelist(type_whitelist);
        op_set.set_max_indegree(max_indegree);

        auto best_model = start.clone();

        spinner->update_status("Caching scores...");

        LocalScoreCache local_validation(*current_model);
        local_validation.cache_local_scores(*current_model, validation_score);

        op_set.cache_scores(*current_model, score);
        int p = 0;
        double validation_offset = 0;

        OperatorTabuSet tabu_set;
        
        auto iter = 0;
        while(iter < max_iters) {
            auto best_op = op_set.find_max(*current_model, tabu_set);
            if (!best_op || best_op->delta() <= epsilon) {
                break;
            }

            best_op->apply(*current_model);
            double validation_delta = validation_delta_score(*current_model,
                                                             validation_score,
                                                             best_op.get(),
                                                             local_validation);
            
            if ((validation_delta + validation_offset) > 0) {
                p = 0;
                validation_offset = 0;
                best_model = current_model->clone();
                tabu_set.clear();
            } else {
                if (++p >= patience)
                    break;
                validation_offset += validation_delta;
                tabu_set.insert(best_op->opposite());
            }


            op_set.update_scores(*current_model, score, *best_op);

            spinner->update_status(best_op->ToString() + " | Validation delta: " + std::to_string(validation_delta));
            ++iter;
        }

        spinner->mark_as_completed("Finished Hill-climbing!");
        indicators::show_console_cursor(true);
        return best_model;
    }

    class GreedyHillClimbing {
    public:
        std::unique_ptr<BayesianNetworkBase> estimate(OperatorSet& op_set,
                                                      Score& score,
                                                      const BayesianNetworkBase& start,
                                                      const ArcStringVector& arc_blacklist,
                                                      const ArcStringVector& arc_whitelist,
                                                      int max_indegree,
                                                      int max_iters, 
                                                      double epsilon,
                                                      int verbose = 0);

        std::unique_ptr<ConditionalBayesianNetworkBase> estimate(
                                                    OperatorSet& op_set,
                                                    Score& score,
                                                    const ConditionalBayesianNetworkBase& start,
                                                    const ArcStringVector& arc_blacklist,
                                                    const ArcStringVector& arc_whitelist,
                                                    int max_indegree,
                                                    int max_iters, 
                                                    double epsilon,
                                                    int verbose = 0);

        std::unique_ptr<BayesianNetworkBase> estimate_validation(OperatorSet& op_set,
                                                                 Score& score,
                                                                 Score& validation_score,
                                                                 const BayesianNetworkBase& start,
                                                                 const ArcStringVector& arc_blacklist,
                                                                 const ArcStringVector& arc_whitelist,
                                                                 const FactorStringTypeVector& type_whitelist,
                                                                 int max_indegree,
                                                                 int max_iters,
                                                                 double epsilon, 
                                                                 int patience,
                                                                 int verbose = 0);


        std::unique_ptr<BayesianNetworkBase> estimate_validation(OperatorSet& op_set,
                                                                Score& score,
                                                                Score& validation_score,
                                                                const ConditionalBayesianNetworkBase& start,
                                                                const ArcStringVector& arc_blacklist,
                                                                const ArcStringVector& arc_whitelist,
                                                                const FactorStringTypeVector& type_whitelist,
                                                                int max_indegree,
                                                                int max_iters,
                                                                double epsilon, 
                                                                int patience,
                                                                int verbose = 0);
    };
}

#endif //PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP