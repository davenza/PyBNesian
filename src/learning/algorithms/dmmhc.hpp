#ifndef PYBNESIAN_LEARNING_ALGORITHMS_DMMHC_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_DMMHC_HPP

#include <models/DynamicBayesianNetwork.hpp>
#include <learning/operators/operators.hpp>
#include <learning/independences/independence.hpp>

using models::DynamicBayesianNetworkBase;
using learning::operators::OperatorSet;
using learning::independences::DynamicIndependenceTest;
using learning::scores::DynamicScore;

namespace learning::algorithms {


    class DMMHC {
    public:
        std::unique_ptr<DynamicBayesianNetworkBase> estimate(const DynamicIndependenceTest& test,
                                                             OperatorSet& op_set,
                                                             DynamicScore& score,
                                                             const std::vector<std::string>& variables,
                                                             const std::string& bn_str,
                                                             int markovian_order,
                                                             int max_indegree,
                                                             int max_iters, 
                                                             double epsilon,
                                                             int patience,
                                                             double alpha,
                                                             int verbose = 0);
    };
}

#endif //PYBNESIAN_LEARNING_ALGORITHMS_DMMHC_HPP