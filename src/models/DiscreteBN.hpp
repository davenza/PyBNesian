#ifndef PGM_DATASET_DISCRETEBN_HPP
#define PGM_DATASET_DISCRETEBN_HPP

#include <models/BayesianNetwork.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <graph/dag.hpp>

using graph::AdjMatrixDag;
using factors::discrete::DiscreteFactor;
// using models::BayesianNetworkType, models::BayesianNetwork;

namespace models {

    template<typename D = AdjMatrixDag>
    class DiscreteBN : public BayesianNetwork<DiscreteBN<D>> {
    public:
        using DagType = D;
        using CPD = LinearGaussianCPD;
        DiscreteBN(const std::vector<std::string>& nodes) : 
                                            BayesianNetwork<DiscreteBN<D>>(nodes) {}
        DiscreteBN(const ArcVector& arcs) : BayesianNetwork<DiscreteBN<D>>(arcs) {}
        DiscreteBN(const std::vector<std::string>& nodes, const ArcVector& arcs) : 
                                            BayesianNetwork<DiscreteBN<D>>(nodes, arcs) {}

        
        static void requires(const DataFrame& df) {
            requires_discrete_data(df);
        }

        std::string ToString() const {
            return "DiscreteBN";
        }

        constexpr BayesianNetworkType type() const override {
            return BayesianNetworkType::DISCRETEBN;
        }
    };

    template<typename D>
    struct BN_traits<DiscreteBN<D>> {
        using DagType = D;
        using CPD = DiscreteFactor;
    };
}

#endif //PGM_DATASET_DISCRETEBN_HPP
