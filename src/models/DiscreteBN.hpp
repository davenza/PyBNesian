#ifndef PYBNESIAN_MODELS_DISCRETEBN_HPP
#define PYBNESIAN_MODELS_DISCRETEBN_HPP

#include <models/BayesianNetwork.hpp>
#include <factors/discrete/DiscreteFactor.hpp>

using factors::discrete::DiscreteFactor;

namespace models {

    class DiscreteBN;
    class ConditionalDiscreteBN;

    template<>
    struct BN_traits<DiscreteBN> {
        using CPD = DiscreteFactor;
    };

    template<>
    struct BN_traits<ConditionalDiscreteBN> {
        using CPD = DiscreteFactor;
    };

    template<typename Derived, template<typename> typename BNType>
    class DiscreteNetworkImpl : public BNType<Derived> {
    public:
        using BNType<Derived>::BNType;

        static void requires(const DataFrame& df) {
            requires_discrete_data(df);
        }

        BayesianNetworkType type() const override {
            return BayesianNetworkType::DISCRETEBN;
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

    class DiscreteBN : public DiscreteNetworkImpl<DiscreteBN, BayesianNetwork> {
    public:
        using DiscreteNetworkImpl<DiscreteBN, BayesianNetwork>::DiscreteNetworkImpl;
        
        std::string ToString() const override {
            return "DiscreteBN";
        }        
    };

    class ConditionalDiscreteBN : public DiscreteNetworkImpl<ConditionalDiscreteBN, ConditionalBayesianNetwork> {
    public:
        using DiscreteNetworkImpl<ConditionalDiscreteBN, ConditionalBayesianNetwork>::DiscreteNetworkImpl;
        
        std::string ToString() const override {
            return "ConditionalGaussianNetwork";
        }        
    };
}

#endif //PYBNESIAN_MODELS_DISCRETEBN_HPP
