#ifndef PYBNESIAN_LEARNING_ALGORITHMS_MMHC_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_MMHC_HPP

#include <models/BayesianNetwork.hpp>
#include <learning/operators/operators.hpp>
#include <learning/independences/independence.hpp>

using models::BayesianNetworkBase;
using learning::operators::OperatorPool;
using learning::independences::IndependenceTest;

namespace learning::algorithms {

    class MMHC {
    public:
        std::unique_ptr<BayesianNetworkBase> estimate(const IndependenceTest& test,
                                                      OperatorPool& op_pool,
                                                      Score* validation_score,
                                                      const std::string& bn_str,
                                                      const ArcStringVector& varc_blacklist,
                                                      const ArcStringVector& varc_whitelist,
                                                      const EdgeStringVector& vedge_blacklist,
                                                      const EdgeStringVector& vedge_whitelist,
                                                      const FactorStringTypeVector& type_whitelist,
                                                      int max_indegree,
                                                      int max_iters, 
                                                      double epsilon,
                                                      int patience,
                                                      double alpha,
                                                      int verbose = 0);
    };
}


#endif //PYBNESIAN_LEARNING_ALGORITHMS_MMHC_HPP