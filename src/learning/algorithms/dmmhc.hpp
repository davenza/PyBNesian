#ifndef PYBNESIAN_LEARNING_ALGORITHMS_DMMHC_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_DMMHC_HPP

#include <models/DynamicBayesianNetwork.hpp>
#include <learning/operators/operators.hpp>
#include <learning/independences/independence.hpp>

using models::DynamicBayesianNetworkBase;
using learning::operators::OperatorSet;
using learning::independences::IndependenceTest;

namespace learning::algorithms {


    class DMMHC {
    public:
        std::unique_ptr<DynamicBayesianNetworkBase> estimate(const IndependenceTest& test,
                                                             OperatorSet& op_set,
                                                             Score& score,
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
                                                             int markovian_order,
                                                             int verbose = 0);
    };
}

#endif //PYBNESIAN_LEARNING_ALGORITHMS_DMMHC_HPP