#ifndef PGM_DATASET_GAUSSIANNETWORK_HPP
#define PGM_DATASET_GAUSSIANNETWORK_HPP

#include <models/BayesianNetwork.hpp>

namespace models {
    
    class GaussianNetwork;
    
    template<>
    struct BN_traits<GaussianNetwork> {
        using CPD = LinearGaussianCPD;
    };

    // template<typename D = AdjMatrixDag>
    class GaussianNetwork : public BayesianNetwork<GaussianNetwork> {
    public:
        // using DagType = D;
        using CPD = LinearGaussianCPD;
        GaussianNetwork(const std::vector<std::string>& nodes) : 
                                            BayesianNetwork<GaussianNetwork>(nodes) {}
        GaussianNetwork(const ArcVector& arcs) : BayesianNetwork<GaussianNetwork>(arcs) {}
        GaussianNetwork(const std::vector<std::string>& nodes, const ArcVector& arcs) : 
                                            BayesianNetwork<GaussianNetwork>(nodes, arcs) {}

        GaussianNetwork(const Dag& graph) : BayesianNetwork<GaussianNetwork>(graph) {}
        GaussianNetwork(Dag&& graph) : BayesianNetwork<GaussianNetwork>(std::move(graph)) {}
        
        static void requires(const DataFrame& df) {
            requires_continuous_data(df);
        }

        std::string ToString() const override {
            return "GaussianNetwork";
        }

        BayesianNetworkType type() const override {
            return BayesianNetworkType::GBN;
        }
    };
}

#endif //PGM_DATASET_GAUSSIANNETWORK_HPP