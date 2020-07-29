#ifndef PGM_DATASET_DISCRETEBN_HPP
#define PGM_DATASET_DISCRETEBN_HPP

#include <models/BayesianNetwork.hpp>
#include <factors/discrete/DiscreteFactor.hpp>

using factors::discrete::DiscreteFactor;
// using models::BayesianNetworkType, models::BayesianNetwork;

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
        DiscreteBN(const ArcVector& arcs) : BayesianNetwork<DiscreteBN>(arcs) {}
        DiscreteBN(const std::vector<std::string>& nodes, const ArcVector& arcs) : 
                                            BayesianNetwork<DiscreteBN>(nodes, arcs) {}

        
        static void requires(const DataFrame& df) {
            requires_discrete_data(df);
        }

        std::string ToString() const override {
            return "DiscreteBN";
        }

        BayesianNetworkType type() const override {
            return BayesianNetworkType::DISCRETEBN;
        }
    };


}

#endif //PGM_DATASET_DISCRETEBN_HPP
