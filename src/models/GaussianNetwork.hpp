#ifndef PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP
#define PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP

#include <models/BayesianNetwork.hpp>
#include <models/ConditionalBayesianNetwork.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>

using factors::continuous::LinearGaussianCPD;

namespace models {
    
    class GaussianNetwork;
    class ConditionalGaussianNetwork;

    template<>
    struct BN_traits<GaussianNetwork> {
        using CPD = LinearGaussianCPD;
    };

    template<>
    struct BN_traits<ConditionalGaussianNetwork> {
        using CPD = LinearGaussianCPD;
    };


    template<typename Derived, template<typename> typename BNType>
    class GaussianNetworkImpl : public BNType<Derived> {
    public:
        using BNType<Derived>::BNType;

        static void requires(const DataFrame& df) {
            requires_continuous_data(df);
        }

        BayesianNetworkType type() const override {
            return BayesianNetworkType::GBN;
        }

        // std::unique_ptr<BayesianNetworkBase> clone() const override {
        //     return std::make_unique<Derived>(static_cast<const Derived&>(*this));
        // }

        py::tuple __getstate__() const {
            return BNType<Derived>::__getstate__();
        }

        static Derived __setstate__(py::tuple& t) {
            return BNType<Derived>::__setstate__(t);
        }

        static Derived __setstate__(py::tuple&& t) {
            return BNType<Derived>::__setstate__(t);
        }
    };

    class GaussianNetwork : public GaussianNetworkImpl<GaussianNetwork, BayesianNetwork> {
    public:
        using GaussianNetworkImpl<GaussianNetwork, BayesianNetwork>::GaussianNetworkImpl;
        
        std::string ToString() const override {
            return "GaussianNetwork";
        }        
    };

    class ConditionalGaussianNetwork : public GaussianNetworkImpl<GaussianNetwork, ConditionalBayesianNetwork> {
    public:
        using GaussianNetworkImpl<GaussianNetwork, ConditionalBayesianNetwork>::GaussianNetworkImpl;
        
        std::string ToString() const override {
            return "ConditionalGaussianNetwork";
        }        
    };
}

#endif //PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP