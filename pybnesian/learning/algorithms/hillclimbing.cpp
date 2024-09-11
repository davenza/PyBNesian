#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_options.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <models/KDENetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>
#include <learning/operators/operators.hpp>

using namespace dataset;

using Eigen::VectorXd, Eigen::MatrixXd;
;
using learning::operators::OperatorSet, learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet;
using learning::scores::BIC, learning::scores::CVLikelihood, learning::scores::HoldoutLikelihood;
using models::BayesianNetworkType, models::GaussianNetwork, models::SemiparametricBN, models::KDENetwork;

using util::ArcStringVector;

namespace learning::algorithms {

/**
 * @brief Executes a greedy hill-climbing algorithm for Bayesian network structure learning. This calls
 GreedyHillClimbing.estimate().
 *
 * @param df DataFrame used to learn a Bayesian network model.
 * @param bn_type BayesianNetworkType of the returned model. If start is given, bn_type is ignored. Defaults to
 * pbn.SemiparametricBNType().
 * @param start Initial structure of the GreedyHillClimbing. If None, a new Bayesian network model is created. Defaults
 * to None.
 * @param score_str A string representing the score used to drive the search.
            The possible options are: “bic” for BIC, “bge” for BGe, “cv-lik” for CVLikelihood, “holdout-lik” for
 HoldoutLikelihood, “validated-lik for ValidatedLikelihood. Defaults to "validated-lik".
 * @param operators_str Set of operators in the search process. Defaults to ["arcs", "node_type"].
 * @param arc_blacklist List of arcs blacklist (forbidden arcs). Defaults to [].
 * @param arc_whitelist List of arcs whitelist (forced  arcs). Defaults to [].
 * @param type_blacklist List of type blacklist (forbidden types). Defaults to [].
 * @param type_whitelist List of type whitelist (forced types). Defaults to [].
 * @param callback Callback object that is called after each iteration. Defaults to None.
 * @param max_indegree Maximum indegree allowed in the graph. Defaults to 0.
 * @param max_iters Maximum number of search iterations. Defaults to 2147483647.
 * @param epsilon Minimum delta score allowed for each operator. If the new operator is less than epsilon, the search
 process is stopped. Defaults to 0.
 * @param patience The patience parameter (only used with pbn.ValidatedScore).  Defaults to 0.
 * @param seed Seed parameter of the score (if needed). Defaults to None.
 * @param num_folds Number of folds for the CVLikelihood and ValidatedLikelihood scores. Defaults to 10.
 * @param test_holdout_ratio Parameter for the HoldoutLikelihood and ValidatedLikelihood scores. Defaults to 0.2.
 * @param verbose If True the progress will be displayed, otherwise nothing will be displayed. Defaults to 0.
 * @return std::shared_ptr<BayesianNetworkBase> The estimated Bayesian network structure.
 */
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
                                        int verbose) {
    if (!bn_type && !start) {
        throw std::invalid_argument("\"bn_type\" or \"start\" parameter must be specified.");
    }
    // If seed is not given, it is set to a random value.
    auto iseed = [seed]() {
        if (seed)
            return *seed;
        else
            return std::random_device{}();
    }();

    // If bn_type is not given, it is set to the type of the given start model.
    const auto& bn_type_ = [&start, &bn_type]() -> const BayesianNetworkType& {
        if (start)
            return start->type_ref();
        else
            return *bn_type;
    }();

    // Checks if the given operators are valid for the given Bayesian network type ["arcs", "node_type"].
    auto operators = util::check_valid_operators(
        bn_type_, operators_str, arc_blacklist, arc_whitelist, max_indegree, type_whitelist);

    // If max_iters is 0, it is set to the maximum integer value.
    if (max_iters == 0) max_iters = std::numeric_limits<int>::max();

    // If start is given, it is used as the initial model. Otherwise, a new model is created.
    const auto start_model = [&start, &bn_type_, &df]() -> const std::shared_ptr<BayesianNetworkBase> {
        if (start)
            return start;
        else
            return bn_type_.new_bn(df.column_names());
    }();

    GreedyHillClimbing hc;

    // If score is not given, it is set to the default score for the given Bayesian network type.
    auto score = util::check_valid_score(df, bn_type_, score_str, iseed, num_folds, test_holdout_ratio);
    return hc.estimate(*operators,
                       *score,
                       *start_model,
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

}  // namespace learning::algorithms
