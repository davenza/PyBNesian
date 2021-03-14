#ifndef PYBNESIAN_LEARNING_ALGORITHMS_MMHC_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_MMHC_HPP

#include <models/BayesianNetwork.hpp>
#include <learning/operators/operators.hpp>
#include <learning/independences/independence.hpp>

using learning::independences::IndependenceTest;
using learning::operators::OperatorSet;
using models::BayesianNetworkBase, models::ConditionalBayesianNetworkBase;

namespace learning::algorithms {

class MMHC {
public:
    std::shared_ptr<BayesianNetworkBase> estimate(const IndependenceTest& test,
                                                  OperatorSet& op_set,
                                                  Score& score,
                                                  const std::vector<std::string>& nodes,
                                                  const std::string& bn_str,
                                                  const ArcStringVector& varc_blacklist,
                                                  const ArcStringVector& varc_whitelist,
                                                  const EdgeStringVector& vedge_blacklist,
                                                  const EdgeStringVector& vedge_whitelist,
                                                  const FactorTypeVector& type_whitelist,
                                                  int max_indegree,
                                                  int max_iters,
                                                  double epsilon,
                                                  int patience,
                                                  double alpha,
                                                  int verbose = 0);

    std::shared_ptr<ConditionalBayesianNetworkBase> estimate_conditional(
        const IndependenceTest& test,
        OperatorSet& op_set,
        Score& score,
        const std::vector<std::string>& nodes,
        const std::vector<std::string>& interface_nodes,
        const std::string& bn_str,
        const ArcStringVector& varc_blacklist,
        const ArcStringVector& varc_whitelist,
        const EdgeStringVector& vedge_blacklist,
        const EdgeStringVector& vedge_whitelist,
        const FactorTypeVector& type_whitelist,
        int max_indegree,
        int max_iters,
        double epsilon,
        int patience,
        double alpha,
        int verbose = 0);
};

}  // namespace learning::algorithms

#endif  // PYBNESIAN_LEARNING_ALGORITHMS_MMHC_HPP