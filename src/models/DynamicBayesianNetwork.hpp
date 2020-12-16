#ifndef PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP
#define PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP

#include <models/BayesianNetwork.hpp>

using models::BayesianNetworkBase;

namespace models {


    class DynamicBayesianNetworkBase {

    };


    template<typename Derived>
    class DynamicBayesianNetwork : public DynamicBayesianNetworkBase {
    public:

        DynamicBayesianNetwork(Derived g0, Derived transition) : m_g0(g0), m_transition(transition) {}


        Derived& g0() { return m_g0; }
        Derived& transition() { return m_transition; }



    private:    
        Derived m_g0;
        Derived m_transition;
    };

}

#endif //PYBNESIAN_MODELS_DYNAMICBAYESIANNETWORK_HPP