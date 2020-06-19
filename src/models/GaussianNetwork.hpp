#ifndef PGM_DATASET_GAUSSIANNETWORK_HPP
#define PGM_DATASET_GAUSSIANNETWORK_HPP

#include <graph/dag.hpp>

using graph::AdjMatrixDag;

namespace models {
    template<typename D = AdjMatrixDag>
    class GaussianNetwork : public BayesianNetwork<GaussianNetwork<D>> {
    public:
        using DagType = D;
        using CPD = LinearGaussianCPD;
        GaussianNetwork(const std::vector<std::string>& nodes) : 
                                            BayesianNetwork<GaussianNetwork<D>>(nodes) {}
        GaussianNetwork(const std::vector<std::string>& nodes, const ArcVector& arcs) : 
                                            BayesianNetwork<GaussianNetwork<D>>(nodes, arcs) {}

        
        static void requires(const DataFrame& df) {
            requires_continuous_data(df);
        }
    };

    template<typename D>
    struct BN_traits<GaussianNetwork<D>> {
        using DagType = D;
        using CPD = LinearGaussianCPD;
    };

}
#endif //PGM_DATASET_GAUSSIANNETWORK_HPP