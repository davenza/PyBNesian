#ifndef PGM_DATASET_BAYESIANNETWORK_HPP
#define PGM_DATASET_BAYESIANNETWORK_HPP

#include <dataset/dataset.hpp>
#include <graph/dag.hpp>

using dataset::DataFrame;
using graph::AdjMatrixDag;

namespace models {


    enum BayesianNetworkType {
        GAUSSIAN_NETWORK
    };

    template<BayesianNetworkType T, typename DagType = AdjMatrixDag>
    class BayesianNetwork {
    
    public:
        using node_descriptor = typename DagType::node_descriptor;
        using edge_descriptor = typename DagType::edge_descriptor;

        BayesianNetwork(const std::vector<std::string>& nodes) : g(nodes), nodes(nodes) {};
        BayesianNetwork(const std::vector<std::string>& nodes, std::vector<std::pair<std::string, std::string>>& arcs) : g(nodes, arcs), nodes(nodes) {};

        static void requires(const DataFrame& df);



        void print() {
            g.print();
        }

    private:
        // TODO: Allow change the type of Dag.
        DagType g;
        std::vector<std::string> nodes;
    };

    using GaussianNetwork = BayesianNetwork<BayesianNetworkType::GAUSSIAN_NETWORK>;

}




#endif //PGM_DATASET_BAYESIANNETWORK_HPP