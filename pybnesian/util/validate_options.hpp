#ifndef PYBNESIAN_UTIL_VALIDATE_OPTIONS_HPP
#define PYBNESIAN_UTIL_VALIDATE_OPTIONS_HPP

#include <learning/operators/operators.hpp>
#include <models/BayesianNetwork.hpp>

using learning::operators::OperatorSet;
using models::BayesianNetworkType;

namespace util {

std::unique_ptr<Score> check_valid_score(const DataFrame& df,
                                         const BayesianNetworkType& bn_type,
                                         const std::optional<std::string>& score,
                                         int seed,
                                         int num_folds,
                                         double test_holdout_ratio);

std::shared_ptr<OperatorSet> check_valid_operators(const BayesianNetworkType& bn_type,
                                                   const std::optional<std::vector<std::string>>& operators,
                                                   const ArcStringVector& arc_blacklist,
                                                   const ArcStringVector& arc_whitelist,
                                                   int max_indegree,
                                                   const FactorTypeVector& type_whitelist);

}  // namespace util

#endif  // PYBNESIAN_UTIL_VALIDATE_OPTIONS_HPP