#ifndef PYBNESIAN_LEARNING_ALGORITHMS_CALLBACKS_CALLBACK_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_CALLBACKS_CALLBACK_HPP

#include <models/BayesianNetwork.hpp>
#include <learning/operators/operators.hpp>
#include <learning/scores/scores.hpp>

using models::BayesianNetworkBase;
using learning::operators::Operator;
using learning::scores::Score;

namespace learning::algorithms::callbacks {

    class Callback {
    public:
        virtual ~Callback() = default;
        virtual void call(BayesianNetworkBase& model, Operator* new_operator, Score& score, int num_iter) const = 0;
    };
}

#endif //PYBNESIAN_LEARNING_ALGORITHMS_CALLBACKS_CALLBACK_HPP