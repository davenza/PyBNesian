#ifndef PYBNESIAN_UTIL_VALIDATE_OPTIONS_HPP
#define PYBNESIAN_UTIL_VALIDATE_OPTIONS_HPP

#include <learning/operators/operators.hpp>

using learning::operators::OperatorSet, learning::operators::OperatorSetType;

namespace util {
    using OperatorSetTypeS = std::unordered_set<OperatorSetType, typename OperatorSetType::HashType>;

    BayesianNetworkType check_valid_bn_string(const std::string& bn_type);
    ScoreType check_valid_score_string(const std::optional<std::string>& score, BayesianNetworkType bn_type);
    OperatorSetTypeS check_valid_operators_string(const std::optional<std::vector<std::string>>& operators, 
                                                  BayesianNetworkType bn_type);

    std::unique_ptr<Score> check_valid_score(const DataFrame& df, 
                                             BayesianNetworkType bn_type,
                                             ScoreType score,
                                             int seed,
                                             int num_folds,
                                             double test_holdout_ratio);
    
    std::shared_ptr<OperatorSet> check_valid_operators(BayesianNetworkType bn_type, 
                                                       const OperatorSetTypeS& operators,
                                                       const ArcStringVector& arc_blacklist,
                                                       const ArcStringVector& arc_whitelist,
                                                       int max_indegree,
                                                       const FactorStringTypeVector& type_whitelist);
}

#endif //PYBNESIAN_UTIL_VALIDATE_OPTIONS_HPP