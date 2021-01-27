#ifndef PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP
#define PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP

#include <models/BayesianNetwork.hpp>
// #include <models/ConditionalBayesianNetwork.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>

using factors::continuous::LinearGaussianCPD;

namespace models {
    
    template<template<BayesianNetworkType> typename _BNClass>
    struct BN_traits<_BNClass<Gaussian>> {
        using CPD = LinearGaussianCPD;
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
    class BayesianNetwork<Gaussian> : public clone_inherit<BayesianNetwork<Gaussian>, 
                                                           BayesianNetworkImpl<BayesianNetwork<Gaussian>, 
                                                                               BayesianNetworkBase>> {
    public:
        inline static constexpr auto TYPE = Gaussian;
        using clone_inherit<BayesianNetwork<Gaussian>, 
                            BayesianNetworkImpl<BayesianNetwork<Gaussian>, BayesianNetworkBase>>::clone_inherit;

        std::string ToString() const override {
            return "GaussianNetwork";
        }
    };

    template<>
    class ConditionalBayesianNetwork<Gaussian> : public clone_inherit<ConditionalBayesianNetwork<Gaussian>, 
                                                        ConditionalBayesianNetworkImpl<ConditionalBayesianNetwork<Gaussian>>> {
    public:
        inline static constexpr auto TYPE = Gaussian;
        using clone_inherit<ConditionalBayesianNetwork<Gaussian>, 
                            ConditionalBayesianNetworkImpl<ConditionalBayesianNetwork<Gaussian>>>::clone_inherit;

        std::string ToString() const override {
            return "ConditionalGaussianNetwork";
        }
    };



    // class GaussianNetwork : public clone_inherit<GaussianNetwork, GaussianNetworkImpl<GaussianNetwork, BayesianNetwork>> {
    // public:
    //     // using GaussianNetworkImpl<GaussianNetwork, BayesianNetwork>::GaussianNetworkImpl;
    //     using clone_inherit<GaussianNetwork, GaussianNetworkImpl<GaussianNetwork, BayesianNetwork>>::clone_inherit;

    //     std::string ToString() const override {
    //         return "GaussianNetwork";
    //     }        
    // };

    // class ConditionalGaussianNetwork : public clone_inherit<ConditionalGaussianNetwork,
    //                                                         GaussianNetworkImpl<ConditionalGaussianNetwork, ConditionalBayesianNetwork>> {
    // public:
    //     // using GaussianNetworkImpl<GaussianNetwork, ConditionalBayesianNetwork>::GaussianNetworkImpl;
    //     using clone_inherit<ConditionalGaussianNetwork,
    //                         GaussianNetworkImpl<ConditionalGaussianNetwork, ConditionalBayesianNetwork>>::clone_inherit;

    //     std::string ToString() const override {
    //         return "ConditionalGaussianNetwork";
    //     }        
    // };
}

#endif //PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP