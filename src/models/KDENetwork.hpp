#ifndef PYBNESIAN_MODELS_KDENETWORK_HPP
#define PYBNESIAN_MODELS_KDENETWORK_HPP

#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <factors/continuous/CKDE.hpp>

namespace models {

    template<template<BayesianNetworkType::Value> typename _BNClass>
    struct BN_traits<_BNClass<BayesianNetworkType::KDENetwork>,
                     std::enable_if_t<(is_unconditional_bn_v<_BNClass<BayesianNetworkType::KDENetwork>> ||
                                        is_conditional_bn_v<_BNClass<BayesianNetworkType::KDENetwork>>),
                                        void>
    > {
        using CPD = CKDE;
        using BaseClass = std::conditional_t<
                                is_unconditional_bn_v<_BNClass<BayesianNetworkType::KDENetwork>>,
                                BayesianNetworkBase,
                                ConditionalBayesianNetworkBase
                        >;
        using DagClass = std::conditional_t<is_unconditional_bn_v<_BNClass<BayesianNetworkType::KDENetwork>>,
                                Dag,
                                ConditionalDag
                        >;
        template<BayesianNetworkType::Value Type>
        using BNClass = _BNClass<Type>;
        inline static constexpr auto TYPE = BayesianNetworkType::KDENetwork;
    };

    template<>
    class BayesianNetwork<BayesianNetworkType::KDENetwork>
        : public clone_inherit<KDENetwork, BayesianNetworkImpl<KDENetwork>> {
    public:
        inline static constexpr auto TYPE = BN_traits<KDENetwork>::TYPE;
        using clone_inherit::clone_inherit;

        std::string ToString() const override {
            return "KDENetwork";
        }
    };

    template<>
    class ConditionalBayesianNetwork<BayesianNetworkType::KDENetwork>
        : public clone_inherit<ConditionalKDENetwork, ConditionalBayesianNetworkImpl<ConditionalKDENetwork>> {
    public:
        inline static constexpr auto TYPE = BN_traits<ConditionalKDENetwork>::TYPE;
        using clone_inherit::clone_inherit;

        std::string ToString() const override {
            return "ConditionalKDENetwork";
        }
    };

    template<>
    class DynamicBayesianNetwork<BayesianNetworkType::KDENetwork> 
        : public clone_inherit<DynamicKDENetwork, DynamicBayesianNetworkImpl<DynamicKDENetwork>> {
    public:
        inline static constexpr auto TYPE = BN_traits<DynamicKDENetwork>::TYPE;
        using clone_inherit::clone_inherit;

        std::string ToString() const override {
            return "DynamicKDENetwork";
        }
    };
}

#endif //PYBNESIAN_MODELS_KDENETWORK_HPP