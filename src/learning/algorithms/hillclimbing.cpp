#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_whitelists.hpp>
#include <util/validate_options.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>
#include <learning/operators/operators.hpp>

using namespace dataset;

using Eigen::VectorXd, Eigen::MatrixXd;;
using models::GaussianNetwork, models::BayesianNetworkType;
using learning::scores::ScoreType, learning::scores::BIC, learning::scores::CVLikelihood, learning::scores::HoldoutLikelihood;
using learning::operators::OperatorSet, learning::operators::OperatorSetType, 
      learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet;

using util::ArcStringVector;


namespace learning::algorithms {
 
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
                                            int verbose) {
        
        auto iseed = [seed]() {
            if (seed) return *seed;
            else return std::random_device{}();
        }();

        auto bn_type = [start, &bn_str]() {
            if (start) return start->type();
            else return util::check_valid_bn_string(bn_str);
        }();

        auto score_type = util::check_valid_score_string(score_str, bn_type);
        auto operators_type = util::check_valid_operators_string(operators_str, bn_type);

        auto operators = util::check_valid_operators(bn_type, operators_type, 
                                arc_blacklist, arc_whitelist, max_indegree, type_whitelist);
        
        if (max_iters == 0) max_iters = std::numeric_limits<int>::max();

        std::unique_ptr<BayesianNetworkBase> created_start_model = [start, bn_type, &df]() -> std::unique_ptr<BayesianNetworkBase> {
            if (!start) {
                switch (bn_type) {
                    case BayesianNetworkType::GBN:
                        return std::make_unique<GaussianNetwork>(df.column_names());
                    case BayesianNetworkType::SPBN:
                        return std::make_unique<SemiparametricBN>(df.column_names());
                    default:
                        throw std::invalid_argument("Wrong BayesianNetwork type. Unreachable code!");
                }
            } else {
                return nullptr;
            }
        }();

        const auto start_model = [start, &created_start_model]() -> const BayesianNetworkBase* {
            if (start) return start;
            else return created_start_model.get();
        }();

        GreedyHillClimbing hc;

        if (score_type == ScoreType::PREDICTIVE_LIKELIHOOD) {
            HoldoutLikelihood validation_score(df, test_holdout_ratio, iseed);
            auto score = util::check_valid_score(validation_score.training_data(), 
                            bn_type, score_type, iseed, num_folds, test_holdout_ratio);
    
            return hc.estimate_validation(*operators, *score, validation_score, *start_model, arc_blacklist, arc_whitelist, 
                                        type_whitelist, max_indegree, max_iters, epsilon, patience, verbose);

        } else {
            auto score = util::check_valid_score(df, bn_type, score_type, iseed, num_folds, test_holdout_ratio);

            return hc.estimate(*operators, *score, *start_model, arc_blacklist, arc_whitelist,
                                max_indegree, max_iters, epsilon, verbose);
        }
    }

    std::unique_ptr<BayesianNetworkBase> estimate_hc(OperatorSet& op_set,
                                                     Score& score,
                                                     const BayesianNetworkBase& start,
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

    std::unique_ptr<BayesianNetworkBase> GreedyHillClimbing::estimate(OperatorSet& op_set,
                                                                      Score& score,
                                                                      const BayesianNetworkBase& start,
                                                                      const ArcStringVector& arc_blacklist,
                                                                      const ArcStringVector& arc_whitelist,
                                                                      int max_indegree,
                                                                      int max_iters,
                                                                      double epsilon,
                                                                      int verbose) {
        auto restrictions = util::validate_restrictions(start, arc_blacklist, arc_whitelist);

        return estimate_hc(op_set,
                           score,
                           start,
                           restrictions.arc_blacklist,
                           restrictions.arc_whitelist,
                           max_indegree,
                           max_iters,
                           epsilon,
                           verbose);

    }

    double validation_delta_score(const BayesianNetworkBase& model, const Score& val_score, 
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
                const auto& score_spbn = dynamic_cast<const ScoreSPBN&>(val_score);
                current_local_scores(node_index) = score_spbn.local_score(new_node_type, node_index, parents.begin(), parents.end());
                return current_local_scores(node_index) - prev;
            }
            default:
                throw std::invalid_argument("Unreachable code. Wrong operator in HoldoutLikelihood::delta_score().");
        }
    }

    std::unique_ptr<BayesianNetworkBase> estimate_validation_hc(OperatorSet& op_set,
                                                                Score& score,
                                                                Score& validation_score,
                                                                const BayesianNetworkBase& start,
                                                                const ArcSet& arc_blacklist,
                                                                const ArcSet& arc_whitelist,
                                                                const FactorStringTypeVector& type_whitelist,
                                                                int max_indegree,
                                                                int max_iters,
                                                                double epsilon, 
                                                                int patience,
                                                                int verbose) {
        util::check_node_type_list(start, type_whitelist);
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
        VectorXd local_validation(current_model->num_nodes());
        for (auto n = 0; n < current_model->num_nodes(); ++n) {
            local_validation(n) = validation_score.local_score(*current_model, n);
        }

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

            double validation_delta = validation_delta_score(*current_model, validation_score, best_op.get(), local_validation);
            
            best_op->apply(*current_model);
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

    std::unique_ptr<BayesianNetworkBase> GreedyHillClimbing::estimate_validation(OperatorSet& op_set,
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
                                                                                 int verbose) {
        auto restrictions = util::validate_restrictions(start, arc_blacklist, arc_whitelist);

        return estimate_validation_hc(op_set,
                                      score,
                                      validation_score,
                                      start,
                                      restrictions.arc_blacklist,
                                      restrictions.arc_whitelist,
                                      type_whitelist,
                                      max_indegree,
                                      max_iters,
                                      epsilon,
                                      patience,
                                      verbose);

    }

    std::unique_ptr<ConditionalBayesianNetworkBase> GreedyHillClimbing::estimate_conditional(
                                                    OperatorSet& op_set,
                                                    Score& score,
                                                    const ConditionalBayesianNetworkBase& start,
                                                    const ArcStringVector& arc_blacklist,
                                                    const ArcStringVector& arc_whitelist,
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
}