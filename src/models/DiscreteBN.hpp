#ifndef PYBNESIAN_MODELS_DISCRETEBN_HPP
#define PYBNESIAN_MODELS_DISCRETEBN_HPP

#include <models/BayesianNetwork.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <util/virtual_clone.hpp>

using factors::discrete::DiscreteFactor;
using util::clone_inherit;

namespace models {

    template<template<BayesianNetworkType> typename _BNClass>
    struct BN_traits<_BNClass<Discrete>> {
        using CPD = DiscreteFactor;
        using DagClass = std::conditional_t<
                            util::GenericInstantation<BayesianNetworkType>::is_template_instantation_v<
                                                                                BayesianNetwork,
                                                                                _BNClass<Gaussian>>,
                            Dag,
                            ConditionalDag>;
        template<BayesianNetworkType Type>
        using BNClass = _BNClass<Type>;
    };

    template<>
    class BayesianNetwork<Discrete> : public clone_inherit<BayesianNetwork<Discrete>, 
                                                           BayesianNetworkImpl<BayesianNetwork<Discrete>, 
                                                                               BayesianNetworkBase>> {
    public:
        inline static constexpr auto TYPE = Discrete;
        using clone_inherit<BayesianNetwork<Discrete>, 
                            BayesianNetworkImpl<BayesianNetwork<Discrete>, 
                                                BayesianNetworkBase>>::clone_inherit;
        std::string ToString() const override {
            return "DiscreteNetwork";
        }        
    };

    template<>
    class ConditionalBayesianNetwork<Discrete> : public clone_inherit<ConditionalBayesianNetwork<Discrete>, 
                                                        ConditionalBayesianNetworkImpl<ConditionalBayesianNetwork<Discrete>>> {
    public:
        inline static constexpr auto TYPE = Discrete;
        using clone_inherit<ConditionalBayesianNetwork<Discrete>, 
                            ConditionalBayesianNetworkImpl<ConditionalBayesianNetwork<Discrete>>>::clone_inherit;

        std::string ToString() const override {
            return "ConditionalDiscreteNetwork";
        }
    };

}

#endif //PYBNESIAN_MODELS_DISCRETEBN_HPP
