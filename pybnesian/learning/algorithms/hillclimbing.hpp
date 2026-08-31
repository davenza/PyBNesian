#ifndef PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP

#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/operators/operators.hpp>
#include <learning/algorithms/callbacks/callback.hpp>
#include <util/validate_whitelists.hpp>
#include <util/math_constants.hpp>
#include <util/progress.hpp>
#include <util/vector.hpp>

namespace py = pybind11;

using dataset::DataFrame;
using learning::algorithms::callbacks::Callback;
using learning::operators::Operator, learning::operators::ArcOperator, learning::operators::ChangeNodeType,
    learning::operators::OperatorTabuSet, learning::operators::OperatorSet, learning::operators::LocalScoreCache;
using learning::scores::Score;
using models::BayesianNetworkType, models::ConditionalBayesianNetworkBase;

using util::ArcStringVector;

namespace learning::algorithms {

std::shared_ptr<BayesianNetworkBase> hc(const DataFrame& df,
                                        const std::shared_ptr<BayesianNetworkType> bn_type,
                                        const std::shared_ptr<BayesianNetworkBase> start,
                                        const std::optional<std::string>& score_str,
                                        const std::optional<std::vector<std::string>>& operators_str,
                                        const ArcStringVector& arc_blacklist,
                                        const ArcStringVector& arc_whitelist,
                                        const FactorTypeVector& type_blacklist,
                                        const FactorTypeVector& type_whitelist,
                                        const std::shared_ptr<Callback> callback,
                                        int max_indegree,
                                        int max_iters,
                                        double epsilon,
                                        int patience,
                                        std::optional<unsigned int> seed,
                                        int num_folds,
                                        double test_holdout_ratio,
                                        int verbose = 0);
/**
 * @brief Calculates the validation delta score for each of the variables.
 *
 * @tparam T Type of the Bayesian network.
 * @param model Bayesian network.
 * @param val_score Validated score.
 * @param variables List of variables.
 * @param current_local_scores Local score cache.
 * @return double The validation delta score.
 */
template <typename T>
double validation_delta_score(const T& model,
                              const ValidatedScore& val_score,
                              const std::vector<std::string>& variables,
                              LocalScoreCache& current_local_scores) {
    double prev = 0;
    double nnew = 0;
    for (const auto& n : variables) {
        prev += current_local_scores.local_score(model, n);
        current_local_scores.update_vlocal_score(model, val_score, n);
        nnew += current_local_scores.local_score(model, n);
    }

    return nnew - prev;
}
/**
 * @brief Executes a greedy hill-climbing algorithm for Bayesian network structure learning.
 *
 * @tparam zero_patience True if patience == 0, False otherwise.
 * @tparam S Type of the score.
 * @tparam T Type of the Bayesian network.
 * @param op_set Set of operators in the search process.
 * @param score Score that drives the search.
 * @param start Initial structure. A BayesianNetworkBase or ConditionalBayesianNetworkBase.
 * @param arc_blacklist List of arcs blacklist (forbidden arcs).
 * @param arc_whitelist List of arcs whitelist (forced arcs).
 * @param type_blacklist List of type blacklist (forbidden pbn.FactorType).
 * @param type_whitelist List of type whitelist (forced pbn.FactorType).
 * @param callback Callback object that is called after each iteration.
 * @param max_indegree Maximum indegree allowed in the graph.
 * @param max_iters Maximum number of search iterations.
 * @param epsilon Minimum delta score allowed for each operator. If (best_op->delta() - epsilon) < util::machine_tol,
 * then the search process is stopped.
 * @param patience The patience parameter (only used with ValidatedScore).
 * @param verbose If True the progress will be displayed, otherwise nothing will be displayed.
 * @return std::shared_ptr<T> The estimated Bayesian network structure of the same type as start.
 */
template <bool zero_patience, typename S, typename T>
std::shared_ptr<T> estimate_hc(OperatorSet& op_set,
                               S& score,
                               const T& start,
                               const ArcStringVector& arc_blacklist,
                               const ArcStringVector& arc_whitelist,
                               const FactorTypeVector& type_blacklist,
                               const FactorTypeVector& type_whitelist,
                               const std::shared_ptr<Callback> callback,
                               int max_indegree,
                               int max_iters,
                               double epsilon,
                               int patience,
                               int verbose) {
    std::string log_str = "HILL-CLIMBING::estimate_hc:\t";
    try {
        util::formatted_log_t(verbose, log_str + "Begins");
        // We copy the arc_blacklist
        // auto arc_blacklist_copy = arc_blacklist;

        // Spinner for the progress bar
        auto spinner = util::indeterminate_spinner(verbose);
        spinner->update_status("Checking dataset...");

        // Model initialization
        auto current_model = start.clone();
        // Model type validation
        current_model->force_type_whitelist(type_whitelist);
        if (current_model->has_unknown_node_types()) {
            auto score_data = score.data();

            if (score_data->num_columns() == 0) {
                throw std::invalid_argument(
                    "The score does not have data to detect the node types. Set the node types for"
                    " all the nodes in the Bayesian network or use an score that uses data (it implements "
                    "Score::data).");
            }

            score_data.raise_has_columns(current_model->nodes());
            current_model->set_unknown_node_types(score_data, type_blacklist);
        }
        // Model arc validation
        current_model->check_blacklist(
            arc_blacklist);  // Checks whether the arc_blacklist is valid for the current_model
        current_model->force_whitelist(arc_whitelist);  // Include the given whitelisted arcs. It checks the validity of
                                                        // the graph after including the arc whitelist.

        // OperatorSet initialization
        op_set.set_arc_blacklist(arc_blacklist);
        op_set.set_arc_whitelist(arc_whitelist);
        op_set.set_type_blacklist(type_blacklist);
        op_set.set_type_whitelist(type_whitelist);
        op_set.set_max_indegree(max_indegree);

        // Search model initialization
        auto prev_current_model = current_model->clone();
        auto best_model = current_model;

        spinner->update_status("Caching scores...");

        // NOTE: Here the score of each node is calculated (log-likelihood fit)
        // Partiendo de que se empieza con los nodos sin padres, se calcula el score independiente, y da que no es 0
        // Opciones:
        // 1. Se calcula el score independiente, y tras fallar el fit de log-likelihood se hace regularización?
        // 2. Se elimina la variable?
        // 3. Hacer try-catch para que cuando de error, se añada regularización al score

        // TODO: Peta si hay variables sueltas con varianza 0 al hacer cross-validation -> Arreglar?
        // Initializes the local validation scores for the current model
        util::formatted_log_t(verbose, log_str + "Local Validation TBC");
        LocalScoreCache local_validation = [&]() {                 // Local validation scores (lambda expression)
            if constexpr (std::is_base_of_v<ValidatedScore, S>) {  // If the score is a ValidatedScore
                LocalScoreCache lc(*current_model);                // Local score cache
                lc.cache_vlocal_scores(*current_model, score);     // Cache the local scores
                return lc;
            } else if constexpr (std::is_base_of_v<Score, S>) {  // If the score is a generic Score
                return LocalScoreCache{};
            } else {
                static_assert(util::always_false<S>, "Wrong Score class for hill-climbing.");
            }
        }();

        util::formatted_log_t(verbose, log_str + "Local Validation Calculated");
        // Cache scores
        util::formatted_log_t(verbose, log_str + "op_set.cache_scores TBC");
        // Caches the delta score values of each operator in the set.
        op_set.cache_scores(*current_model, score);
        int p = 0;
        double accumulated_offset = 0;

        util::formatted_log_t(verbose, log_str + "Scores cached");
        OperatorTabuSet tabu_set;

        if (callback) callback->call(*current_model, nullptr, score, 0);
        util::formatted_log_t(verbose, log_str + "Hill climbing iterations begin");
        // Hill climbing iterations begin
        auto iter = 0;
        while (iter < max_iters) {
            ++iter;
            // Finds the best operator
            // HC Algorithm lines 8 -> 16 [Atienza et al. (2022)]
            // NOTE: Here the best operators are evaluated (log-likelihood fit)
            util::formatted_log_t(verbose, log_str + "Best operator TBC");
            auto best_op = [&]() {
                if constexpr (zero_patience)
                    return op_set.find_max(*current_model);
                else
                    return op_set.find_max(*current_model, tabu_set);
            }();

            if (!best_op || (best_op->delta() - epsilon) < util::machine_tol) {
                util::formatted_log_t(verbose, log_str + "No improvement in best_op");
                break;
            }
            util::formatted_log_t(verbose, log_str + "Best operator Calculated" + best_op->ToString());
            // If the best operator is nullptr or the delta is less than epsilon, then the search process fails and
            // stops

            // S_validation puede pasar try { Algorithm lines 17 -> 24 [Atienza et al. (2022)] Applies the best operator
            // to the current model
            best_op->apply(*current_model);
            // Returns the nodes changed by the best operator
            auto nodes_changed = best_op->nodes_changed(*current_model);

            // Calculates the validation delta
            util::formatted_log_t(verbose, log_str + "Validation Delta TBC");

            double validation_delta = [&]() {
                if constexpr (std::is_base_of_v<ValidatedScore, S>) {
                    return validation_delta_score(*current_model, score, nodes_changed, local_validation);
                } else {
                    return best_op->delta();
                }
            }();
            util::formatted_log_t(verbose, log_str + "Validation Delta Calculated");
            // Updates the best model if the validation delta is greater than 0
            if ((validation_delta + accumulated_offset) >
                util::machine_tol) {  // If the validation delta is greater than 0, then the current model is the best
                                      // model
                util::formatted_log_t(verbose, log_str + "Validation Delta is greater than 0");
                if constexpr (!zero_patience) {
                    if (p > 0) {
                        best_model = current_model;
                        p = 0;
                        accumulated_offset = 0;
                    }

                    tabu_set.clear();
                }
            } else {  // If the validation delta is less than 0, then the current model is not the best model
                util::formatted_log_t(verbose, log_str + "Validation Delta is less than 0");
                if constexpr (zero_patience) {
                    best_model = prev_current_model;
                    break;
                } else {
                    if (p == 0) best_model = prev_current_model->clone();
                    if (++p > patience) break;
                    accumulated_offset += validation_delta;
                    tabu_set.insert(best_op->opposite(*current_model));  // Add the opposite operator to the tabu set
                }
            }

            // Updates the previous current model
            best_op->apply(*prev_current_model);

            if (callback) callback->call(*current_model, best_op.get(), score, iter);

            util::formatted_log_t(verbose, log_str + "Updating scores");
            // NOTE: Here the scores node are reevaluated (log-likelihood fit)
            op_set.update_scores(*current_model, score, nodes_changed);
            util::formatted_log_t(verbose, log_str + "Scores updated");
            if constexpr (std::is_base_of_v<ValidatedScore, S>) {
                spinner->update_status(best_op->ToString() +
                                       " | Validation delta: " + std::to_string(validation_delta));
            } else if constexpr (std::is_base_of_v<Score, S>) {
                spinner->update_status(best_op->ToString());
            } else {
                static_assert(util::always_false<S>, "Wrong Score class for hill-climbing.");
            }

        }  // End of Hill climbing iterations

        op_set.finished();

        if (callback) callback->call(*best_model, nullptr, score, iter);

        spinner->mark_as_completed("Finished Hill-climbing!");
        return best_model;
    } catch (util::singular_covariance_data& e) {
        util::formatted_log_t(verbose, log_str + "catch");
        throw e;
        //     auto arc_best_op = dynamic_cast<ArcOperator*>(best_op.get());
        //     auto source_arc = arc_best_op->source();
        //     auto target_arc = arc_best_op->target();

        //     std::cout << e.what() << std::endl;
        //     std::cout << "Source arc:\t" << source_arc << std::endl;
        //     std::cout << "Target arc:\t" << target_arc << std::endl;

        //     arc_blacklist_copy.push_back(std::make_pair(source_arc, target_arc));
        //     std::cout << "New arc_blacklist:\t" << arc_blacklist << std::endl;
        //     op_set.set_arc_blacklist(arc_blacklist_copy);
    }
}
/**
 * @brief Depending on the validated_score and the patience of the hill climbing algorithm it estimates the
 * structure of the Bayesian network.
 *
 * @tparam T
 * @param op_set
 * @param score
 * @param start
 * @param arc_blacklist
 * @param arc_whitelist
 * @param type_blacklist
 * @param type_whitelist
 * @param callback
 * @param max_indegree
 * @param max_iters
 * @param epsilon
 * @param patience
 * @param verbose
 * @return std::shared_ptr<T>
 */
template <typename T>
std::shared_ptr<T> estimate_downcast_score(OperatorSet& op_set,
                                           Score& score,
                                           const T& start,
                                           const ArcStringVector& arc_blacklist,
                                           const ArcStringVector& arc_whitelist,
                                           const FactorTypeVector& type_blacklist,
                                           const FactorTypeVector& type_whitelist,
                                           const std::shared_ptr<Callback> callback,
                                           int max_indegree,
                                           int max_iters,
                                           double epsilon,
                                           int patience,
                                           int verbose) {
    if (auto validated_score = dynamic_cast<ValidatedScore*>(&score)) {
        if (patience == 0) {
            return estimate_hc<true>(op_set,
                                     *validated_score,
                                     start,
                                     arc_blacklist,
                                     arc_whitelist,
                                     type_blacklist,
                                     type_whitelist,
                                     callback,
                                     max_indegree,
                                     max_iters,
                                     epsilon,
                                     patience,
                                     verbose);
        } else {
            return estimate_hc<false>(op_set,
                                      *validated_score,
                                      start,
                                      arc_blacklist,
                                      arc_whitelist,
                                      type_blacklist,
                                      type_whitelist,
                                      callback,
                                      max_indegree,
                                      max_iters,
                                      epsilon,
                                      patience,
                                      verbose);
        }
    } else {
        if (patience == 0) {
            return estimate_hc<true>(op_set,
                                     score,
                                     start,
                                     arc_blacklist,
                                     arc_whitelist,
                                     type_blacklist,
                                     type_whitelist,
                                     callback,
                                     max_indegree,
                                     max_iters,
                                     epsilon,
                                     patience,
                                     verbose);
        } else {
            return estimate_hc<false>(op_set,
                                      score,
                                      start,
                                      arc_blacklist,
                                      arc_whitelist,
                                      type_blacklist,
                                      type_whitelist,
                                      callback,
                                      max_indegree,
                                      max_iters,
                                      epsilon,
                                      patience,
                                      verbose);
        }
    }
}
/**
 * @brief Checks the parameters of the hill climbing algorithm and estimates the structure of a Bayesian network.
 *
 * @tparam T
 * @param op_set
 * @param score
 * @param start
 * @param arc_blacklist
 * @param arc_whitelist
 * @param type_blacklist
 * @param type_whitelist
 * @param callback
 * @param max_indegree
 * @param max_iters
 * @param epsilon
 * @param patience
 * @param verbose
 * @return std::shared_ptr<T>
 */
template <typename T>
std::shared_ptr<T> estimate_checks(OperatorSet& op_set,
                                   Score& score,
                                   const T& start,
                                   const ArcStringVector& arc_blacklist,
                                   const ArcStringVector& arc_whitelist,
                                   const FactorTypeVector& type_blacklist,
                                   const FactorTypeVector& type_whitelist,
                                   const std::shared_ptr<Callback> callback,
                                   int max_indegree,
                                   int max_iters,
                                   double epsilon,
                                   int patience,
                                   int verbose) {
    if (!score.compatible_bn(start)) {
        throw std::invalid_argument("BayesianNetwork is not compatible with the score.");
    }

    util::validate_restrictions(start, arc_blacklist, arc_whitelist);
    util::validate_type_restrictions(start, type_blacklist, type_whitelist);

    return estimate_downcast_score(op_set,
                                   score,
                                   start,
                                   arc_blacklist,
                                   arc_whitelist,
                                   type_blacklist,
                                   type_whitelist,
                                   callback,
                                   max_indegree,
                                   max_iters,
                                   epsilon,
                                   patience,
                                   verbose);
}

class GreedyHillClimbing {
public:
    /**
     * @brief Estimates the structure of a Bayesian network. The estimated Bayesian network is of the same type as
     * start. The set of operators allowed in the search is operators. The delta score of each operator is evaluated
     * using the score. The initial structure of the algorithm is the model start.
     *
     * @tparam T Type of the Bayesian network.
     * @param op_set Set of operators in the search process.
     * @param score pbn.core that drives the search.
     * @param start Initial structure. A BayesianNetworkBase or ConditionalBayesianNetworkBase.
     * @param arc_blacklist List of arcs blacklist (forbidden arcs).
     * @param arc_whitelist List of arcs whitelist (forced arcs).
     * @param type_blacklist List of type blacklist (forbidden pbn.FactorType).
     * @param type_whitelist List of type whitelist (forced pbn.FactorType).
     * @param callback Callback object that is called after each iteration.
     * @param max_indegree Maximum indegree allowed in the graph.
     * @param max_iters Maximum number of search iterations.
     * @param epsilon Minimum delta score allowed for each operator. If the new operator is less than epsilon, the
     * search process is stopped.
     * @param patience he patience parameter (only used with pbn.ValidatedScore).
     * @param verbose If True the progress will be displayed, otherwise nothing will be displayed.
     * @return std::shared_ptr<T> The estimated Bayesian network structure of the same type as start.
     */
    template <typename T>
    std::shared_ptr<T> estimate(OperatorSet& op_set,
                                Score& score,
                                const T& start,
                                const ArcStringVector& arc_blacklist,
                                const ArcStringVector& arc_whitelist,
                                const FactorTypeVector& type_blacklist,
                                const FactorTypeVector& type_whitelist,
                                const std::shared_ptr<Callback> callback,
                                int max_indegree,
                                int max_iters,
                                double epsilon,
                                int patience,
                                int verbose = 0) {
        return estimate_checks(op_set,
                               score,
                               start,
                               arc_blacklist,
                               arc_whitelist,
                               type_blacklist,
                               type_whitelist,
                               callback,
                               max_indegree,
                               max_iters,
                               epsilon,
                               patience,
                               verbose);
    }
};

}  // namespace learning::algorithms

#endif  // PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP
