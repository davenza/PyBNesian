#ifndef PYBNESIAN_MODELS_DISCRETEBN_HPP
#define PYBNESIAN_MODELS_DISCRETEBN_HPP

#include <models/BayesianNetwork.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <util/virtual_clone.hpp>

using factors::discrete::DiscreteFactor;
using util::clone_inherit;

namespace models {

    template<template<BayesianNetworkType::Value> typename _BNClass>
    struct BN_traits<_BNClass<BayesianNetworkType::Discrete>> {
        using CPD = DiscreteFactor;
        using BaseClass = std::conditional_t<
                    util::GenericInstantation<BayesianNetworkType::Value>::is_template_instantation_v<
                                                                        BayesianNetwork,
                                                                        _BNClass<BayesianNetworkType::Discrete>>,
                    BayesianNetworkBase,
                    ConditionalBayesianNetworkBase>;
        using DagClass = std::conditional_t<
                    util::GenericInstantation<BayesianNetworkType::Value>::is_template_instantation_v<
                                                                        BayesianNetwork,
                                                                        _BNClass<BayesianNetworkType::Discrete>>,
                    Dag,
                    ConditionalDag>;
        template<BayesianNetworkType::Value Type>
        using BNClass = _BNClass<Type>;
        inline static constexpr auto TYPE = BayesianNetworkType::Discrete;
    };

    template<>
    class BayesianNetwork<BayesianNetworkType::Discrete>
        : public clone_inherit<DiscreteBN, BayesianNetworkImpl<DiscreteBN>> {
    public:
        inline static constexpr auto TYPE = BN_traits<DiscreteBN>::TYPE;
        using clone_inherit::clone_inherit;
        std::string ToString() const override {
            return "DiscreteNetwork";
        }        
    };

    template<>
    class ConditionalBayesianNetwork<BayesianNetworkType::Discrete>
        : public clone_inherit<ConditionalDiscreteBN, ConditionalBayesianNetworkImpl<ConditionalDiscreteBN>> {
    public:
        inline static constexpr auto TYPE = BN_traits<ConditionalDiscreteBN>::TYPE;
        using clone_inherit::clone_inherit;

        std::string ToString() const override {
            return "ConditionalDiscreteNetwork";
        }
    };

    template<>
    class DynamicBayesianNetwork<BayesianNetworkType::Discrete> 
        : public clone_inherit<DynamicDiscreteBN, DynamicBayesianNetworkImpl<DynamicDiscreteBN>> {
    public:
        inline static constexpr auto TYPE = BN_traits<DynamicDiscreteBN>::TYPE;
        using clone_inherit::clone_inherit;

        std::string ToString() const override {
            return "DynamicGaussianNetwork";
        }
    };
}

#endif //PYBNESIAN_MODELS_DISCRETEBN_HPP
