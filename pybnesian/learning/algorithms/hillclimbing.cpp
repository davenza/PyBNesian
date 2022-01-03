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

    auto iseed = [seed]() {
        if (seed)
            return *seed;
        else
            return std::random_device{}();
    }();

    const auto& bn_type_ = [&start, &bn_type]() -> const BayesianNetworkType& {
        if (start)
            return start->type_ref();
        else
            return *bn_type;
    }();

    auto operators = util::check_valid_operators(
        bn_type_, operators_str, arc_blacklist, arc_whitelist, max_indegree, type_whitelist);

    if (max_iters == 0) max_iters = std::numeric_limits<int>::max();

    const auto start_model = [&start, &bn_type_, &df]() -> const std::shared_ptr<BayesianNetworkBase> {
        if (start)
            return start;
        else
            return bn_type_.new_bn(df.column_names());
    }();

    GreedyHillClimbing hc;
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
