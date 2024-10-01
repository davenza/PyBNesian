#include <util/validate_options.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/KDENetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <models/DiscreteBN.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/bge.hpp>
#include <learning/scores/validated_likelihood.hpp>

using learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet, learning::operators::OperatorPool;
using learning::scores::BIC, learning::scores::BGe, learning::scores::ValidatedLikelihood;
using models::GaussianNetworkType, models::KDENetworkType, models::SemiparametricBNType, models::DiscreteBNType;

namespace util {

/**
 * @brief Checks if the given score is valid for the given Bayesian network type e.g., "bic","bge, "cv-lik",
 * "holdout-lik", "validated-lik".
 *
 * @param df
 * @param bn_type
 * @param score
 * @param seed
 * @param num_folds
 * @param test_holdout_ratio
 * @return std::unique_ptr<Score>
 */
std::unique_ptr<Score> check_valid_score(const DataFrame& df,
                                         const BayesianNetworkType& bn_type,
                                         const std::optional<std::string>& score,
                                         int seed,
                                         int num_folds,
                                         double test_holdout_ratio) {
    if (score) {  // If score is specified
        if (*score == "bic") return std::make_unique<BIC>(df);
        if (*score == "bge") return std::make_unique<BGe>(df);
        if (*score == "cv-lik") return std::make_unique<CVLikelihood>(df, num_folds, seed);
        if (*score == "holdout-lik") return std::make_unique<HoldoutLikelihood>(df, test_holdout_ratio, seed);
        if (*score == "validated-lik")
            return std::make_unique<ValidatedLikelihood>(df, test_holdout_ratio, num_folds, seed);
        else
            throw std::invalid_argument(
                "Wrong Bayesian Network score \"" + *score +
                "\" specified. The possible alternatives are "
                "\"bic\" (Bayesian Information Criterion), \"bge\" (Bayesian Gaussian equivalent), "
                "\"cv-lik\" (Cross-Validated likelihood), \"holdout-l\" (Hold-out likelihood) "
                " or \"validated-lik\" (Validated likelihood with cross-validation).");
    } else {  // If score is not specified
        if (bn_type == GaussianNetworkType::get_ref()) {
            return std::make_unique<BIC>(df);  // Default score for GaussianNetworkType
        } else if (bn_type == SemiparametricBNType::get_ref() || bn_type == KDENetworkType::get_ref()) {
            return std::make_unique<ValidatedLikelihood>(
                df, test_holdout_ratio, num_folds, seed);  // Default score for SemiparametricBNType and KDENetworkType
        } else {
            throw std::invalid_argument("Default score not defined for " + bn_type.ToString() + ".");
        }
    }
}

/**
 * @brief Checks if the given operators are valid for the given Bayesian network type ["arcs", "node_type"].
 * Otherwise, it returns the default operators for the given Bayesian network type
 *
 * @param bn_type
 * @param operators
 * @param arc_blacklist
 * @param arc_whitelist
 * @param max_indegree
 * @param type_whitelist
 * @return std::shared_ptr<OperatorSet>
 */
std::shared_ptr<OperatorSet> check_valid_operators(const BayesianNetworkType& bn_type,
                                                   const std::optional<std::vector<std::string>>& operators,
                                                   const ArcStringVector& arc_blacklist,
                                                   const ArcStringVector& arc_whitelist,
                                                   int max_indegree,
                                                   const FactorTypeVector& type_whitelist) {
    std::vector<std::shared_ptr<OperatorSet>> res;

    if (operators && !operators->empty()) {  // If operators are specified
        for (auto& op : *operators) {
            if (op == "arcs") {
                res.push_back(std::make_shared<ArcOperatorSet>(arc_blacklist, arc_whitelist, max_indegree));
            } else if (op == "node_type") {
                if (bn_type != SemiparametricBNType::get_ref()) {
                    throw std::invalid_argument(
                        "Operator \"node_type\" is not compabible with "
                        "Bayesian network type \"" +
                        bn_type.ToString() + "\"");
                }

                res.push_back(std::make_shared<ChangeNodeTypeSet>(type_whitelist));
            } else
                throw std::invalid_argument("Wrong operator set \"" + op +
                                            "\". Valid choices are:"
                                            "\"arcs\" (Changes in arcs; addition, removal and flip) or "
                                            "\"node_type\" (Change of node type)");
        }
    } else {  // If operators are not specified
        if (bn_type == GaussianNetworkType::get_ref())
            res.push_back(std::make_shared<ArcOperatorSet>(arc_blacklist, arc_whitelist, max_indegree));
        else if (bn_type == SemiparametricBNType::get_ref()) {
            res.push_back(std::make_shared<ArcOperatorSet>(arc_blacklist, arc_whitelist, max_indegree));
            res.push_back(std::make_shared<ChangeNodeTypeSet>(type_whitelist));
        } else if (bn_type == KDENetworkType::get_ref())
            res.push_back(std::make_shared<ArcOperatorSet>(arc_blacklist, arc_whitelist, max_indegree));
        else
            throw std::invalid_argument("Default operators not defined for " + bn_type.ToString() + ".");
    }

    if (res.size() == 1) {
        return std::move(res[0]);
    } else {
        return std::make_unique<OperatorPool>(std::move(res));
    }
}

}  // namespace util