#ifndef PYBNESIAN_LEARNING_ALGORITHMS_DMMHC_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_DMMHC_HPP

#include <models/DynamicBayesianNetwork.hpp>
#include <learning/operators/operators.hpp>
#include <learning/independences/independence.hpp>

using learning::independences::DynamicIndependenceTest;
using learning::operators::OperatorSet;
using learning::scores::DynamicScore;
using models::DynamicBayesianNetworkBase;

namespace learning::algorithms {

class DMMHC {
public:
    std::shared_ptr<DynamicBayesianNetworkBase> estimate(const DynamicIndependenceTest& test,
                                                         OperatorSet& op_set,
                                                         DynamicScore& score,
                                                         const std::vector<std::string>& variables,
                                                         const BayesianNetworkType& bn_type,
                                                         int markovian_order,
                                                         const std::shared_ptr<Callback> static_callback,
                                                         const std::shared_ptr<Callback> transition_callback,
                                                         int max_indegree,
                                                         int max_iters,
                                                         double epsilon,
                                                         int patience,
                                                         double alpha,
                                                         int verbose = 0);
};

}  // namespace learning::algorithms

#endif  // PYBNESIAN_LEARNING_ALGORITHMS_DMMHC_HPP