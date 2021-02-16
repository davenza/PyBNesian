#ifndef PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP
#define PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP

#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>

using factors::continuous::LinearGaussianCPD;

namespace models {
    
    template<template<BayesianNetworkType::Value> typename _BNClass>
    struct BN_traits<_BNClass<BayesianNetworkType::Gaussian>> {
        using CPD = LinearGaussianCPD;
        using BaseClass = std::conditional_t<
                    util::GenericInstantation<BayesianNetworkType::Value>::is_template_instantation_v<
                                                                        BayesianNetwork,
                                                                        _BNClass<BayesianNetworkType::Gaussian>>,
                    BayesianNetworkBase,
                    ConditionalBayesianNetworkBase>;
        using DagClass = std::conditional_t<
                    util::GenericInstantation<BayesianNetworkType::Value>::is_template_instantation_v<
                                                                        BayesianNetwork,
                                                                        _BNClass<BayesianNetworkType::Gaussian>>,
                    Dag,
                    ConditionalDag>;
        template<BayesianNetworkType::Value Type>
        using BNClass = _BNClass<Type>;
        inline static constexpr auto TYPE = BayesianNetworkType::Gaussian;
    };

    template<>
    class BayesianNetwork<BayesianNetworkType::Gaussian>
        : public clone_inherit<GaussianNetwork, BayesianNetworkImpl<GaussianNetwork>> {
    public:
        inline static constexpr auto TYPE = BN_traits<GaussianNetwork>::TYPE;
        using clone_inherit::clone_inherit;

        std::string ToString() const override {
            return "GaussianNetwork";
        }
    };

    template<>
    class ConditionalBayesianNetwork<BayesianNetworkType::Gaussian>
        : public clone_inherit<ConditionalGaussianNetwork, ConditionalBayesianNetworkImpl<ConditionalGaussianNetwork>> {
    public:
        inline static constexpr auto TYPE = BN_traits<ConditionalGaussianNetwork>::TYPE;
        using clone_inherit::clone_inherit;

        std::string ToString() const override {
            return "ConditionalGaussianNetwork";
        }
    };

    template<>
    class DynamicBayesianNetwork<BayesianNetworkType::Gaussian> 
        : public clone_inherit<DynamicGaussianNetwork, DynamicBayesianNetworkImpl<DynamicGaussianNetwork>> {
    public:
        inline static constexpr auto TYPE = BN_traits<DynamicGaussianNetwork>::TYPE;
        using clone_inherit::clone_inherit;

        std::string ToString() const override {
            return "DynamicGaussianNetwork";
        }
    };
}

#endif //PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP
