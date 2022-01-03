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
    auto spinner = util::indeterminate_spinner(verbose);
    spinner->update_status("Checking dataset...");

    auto current_model = start.clone();
    current_model->force_type_whitelist(type_whitelist);

    if (current_model->has_unknown_node_types()) {
        auto score_data = score.data();

        if (score_data->num_columns() == 0) {
            throw std::invalid_argument(
                "The score does not have data to detect the node types. Set the node types for"
                " all the nodes in the Bayesian network or use an score that uses data (it implements Score::data).");
        }

        score_data.raise_has_columns(current_model->nodes());
        current_model->set_unknown_node_types(score_data, type_blacklist);
    }

    current_model->check_blacklist(arc_blacklist);
    current_model->force_whitelist(arc_whitelist);

    op_set.set_arc_blacklist(arc_blacklist);
    op_set.set_arc_whitelist(arc_whitelist);
    op_set.set_type_blacklist(type_blacklist);
    op_set.set_type_whitelist(type_whitelist);
    op_set.set_max_indegree(max_indegree);

    auto prev_current_model = current_model->clone();
    auto best_model = current_model;

    spinner->update_status("Caching scores...");

    LocalScoreCache local_validation = [&]() {
        if constexpr (std::is_base_of_v<ValidatedScore, S>) {
            LocalScoreCache lc(*current_model);
            lc.cache_vlocal_scores(*current_model, score);
            return lc;
        } else if constexpr (std::is_base_of_v<Score, S>) {
            return LocalScoreCache{};
        } else {
            static_assert(util::always_false<S>, "Wrong Score class for hill-climbing.");
        }
    }();

    op_set.cache_scores(*current_model, score);
    int p = 0;
    double accumulated_offset = 0;

    OperatorTabuSet tabu_set;

    if (callback) callback->call(*current_model, nullptr, score, 0);

    auto iter = 0;
    while (iter < max_iters) {
        ++iter;

        auto best_op = [&]() {
            if constexpr (zero_patience)
                return op_set.find_max(*current_model);
            else
                return op_set.find_max(*current_model, tabu_set);
        }();

        if (!best_op || (best_op->delta() - epsilon) < util::machine_tol) {
            break;
        }

        best_op->apply(*current_model);

        auto nodes_changed = best_op->nodes_changed(*current_model);

        double validation_delta = [&]() {
            if constexpr (std::is_base_of_v<ValidatedScore, S>) {
                return validation_delta_score(*current_model, score, nodes_changed, local_validation);
            } else {
                return best_op->delta();
            }
        }();

        if ((validation_delta + accumulated_offset) > util::machine_tol) {
            if constexpr (!zero_patience) {
                if (p > 0) {
                    best_model = current_model;
                    p = 0;
                    accumulated_offset = 0;
                }

                tabu_set.clear();
            }
        } else {
            if constexpr (zero_patience) {
                best_model = prev_current_model;
                break;
            } else {
                if (p == 0) best_model = prev_current_model->clone();
                if (++p > patience) break;
                accumulated_offset += validation_delta;
                tabu_set.insert(best_op->opposite(*current_model));
            }
        }

        best_op->apply(*prev_current_model);

        if (callback) callback->call(*current_model, best_op.get(), score, iter);

        op_set.update_scores(*current_model, score, nodes_changed);

        if constexpr (std::is_base_of_v<ValidatedScore, S>) {
            spinner->update_status(best_op->ToString() + " | Validation delta: " + std::to_string(validation_delta));
        } else if constexpr (std::is_base_of_v<Score, S>) {
            spinner->update_status(best_op->ToString());
        } else {
            static_assert(util::always_false<S>, "Wrong Score class for hill-climbing.");
        }
    }

    op_set.finished();

    if (callback) callback->call(*best_model, nullptr, score, iter);

    spinner->mark_as_completed("Finished Hill-climbing!");
    return best_model;
}

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
