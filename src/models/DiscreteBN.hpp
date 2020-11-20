#ifndef PYBNESIAN_MODELS_DISCRETEBN_HPP
#define PYBNESIAN_MODELS_DISCRETEBN_HPP

#include <models/BayesianNetwork.hpp>
#include <factors/discrete/DiscreteFactor.hpp>

using factors::discrete::DiscreteFactor;

namespace models {

    class DiscreteBN;

    template<>
    struct BN_traits<DiscreteBN> {
        using CPD = DiscreteFactor;
    };

    class DiscreteBN : public BayesianNetwork<DiscreteBN> {
    public:
        // using DagType = D;
        using CPD = DiscreteFactor;
        DiscreteBN(const std::vector<std::string>& nodes) : 
                                            BayesianNetwork<DiscreteBN>(nodes) {}
        DiscreteBN(const ArcStringVector& arcs) : BayesianNetwork<DiscreteBN>(arcs) {}
        DiscreteBN(const std::vector<std::string>& nodes, const ArcStringVector& arcs) : 
                                            BayesianNetwork<DiscreteBN>(nodes, arcs) {}
        DiscreteBN(const Dag& graph) : BayesianNetwork<DiscreteBN>(graph) {}
        DiscreteBN(Dag&& graph) : BayesianNetwork<DiscreteBN>(std::move(graph)) {}

        
        static void requires(const DataFrame& df) {
            requires_discrete_data(df);
        }

        std::string ToString() const override {
            return "DiscreteBN";
        }

        BayesianNetworkType type() const override {
            return BayesianNetworkType::DISCRETEBN;
        }

        py::tuple __getstate__() const {
            return BayesianNetwork<DiscreteBN>::__getstate__();
        }

        static DiscreteBN __setstate__(py::tuple& t) {
            return BayesianNetwork<DiscreteBN>::__setstate__(t);
        }

        static DiscreteBN __setstate__(py::tuple&& t) {
            return BayesianNetwork<DiscreteBN>::__setstate__(t);
        }
    };


}

#endif //PYBNESIAN_MODELS_DISCRETEBN_HPP
