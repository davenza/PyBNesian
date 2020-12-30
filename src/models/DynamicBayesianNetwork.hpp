#ifndef PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP
#define PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP

#include <models/BayesianNetwork.hpp>

using models::BayesianNetworkBase;

namespace models {


    class DynamicBayesianNetworkBase {
    public:
        virtual ~DynamicBayesianNetworkBase() = default;
        virtual BayesianNetworkBase& static_bn() = 0;
        virtual BayesianNetworkBase& transition_bn() = 0;

        virtual std::vector<std::string> variables() const = 0;
        virtual int markovian_order() const = 0;




    };


    template<typename Derived>
    class DynamicBayesianNetwork : public DynamicBayesianNetworkBase {
    public:
        DynamicBayesianNetwork(Derived static_bn, Derived transition_bn) : m_static(static_bn),
                                                                           m_transition(transition_bn) {}

        BayesianNetworkBase& static_bn() override { return m_static; }
        BayesianNetworkBase& transition_bn() override { return m_transition; }

        std::vector<std::string> variables() const override;
        int markovian_order() const override;
        


    private:    
        Derived m_static;
        Derived m_transition;
    };

}

#endif //PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP