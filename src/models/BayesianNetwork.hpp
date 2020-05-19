#ifndef PGM_DATASET_BAYESIANNETWORK_HPP
#define PGM_DATASET_BAYESIANNETWORK_HPP

#include <iterator>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>

using dataset::DataFrame;
using graph::AdjMatrixDag, graph::dag_node_iterator;

namespace models {


    enum BayesianNetworkType {
        GAUSSIAN_NETWORK
    };


    // template<typename it> node_iterator(it b, it e) -> node_iterator<typename std::iterator_traits<Iterator>::value_type>;

    template<BayesianNetworkType T, typename DagType = AdjMatrixDag>
    class BayesianNetwork {
    
    public:
        using node_descriptor = typename DagType::node_descriptor;
        using edge_descriptor = typename DagType::edge_descriptor;

        using node_iterator_t = typename DagType::node_iterator_t;

        using nodes_size_type = typename DagType::nodes_size_type;
        using edges_size_type = typename DagType::edges_size_type;

        BayesianNetwork(const std::vector<std::string>& nodes) : g(nodes), m_nodes(nodes) {};
        BayesianNetwork(const std::vector<std::string>& nodes, std::vector<std::pair<std::string, std::string>>& arcs) : g(nodes, arcs), m_nodes(nodes) {};

        static void requires(const DataFrame& df);

        dag_node_iterator<node_iterator_t> nodes() const { 
            return g.nodes();
        }


        nodes_size_type num_nodes() const {
            return g.num_nodes();
        }

        edges_size_type num_edges() const {
            return g.num_edges();
        }

        const std::vector<std::string>& names() const {
            return m_nodes;
        }

        

        // edge_iterator edges() const;

        // nodestr_iterator nodes_str() const;
        // edgestr_iterator edges_str() const;

        // void print() {
        //     g.print();
        // }

    private:
        // TODO: Allow change the type of Dag.
        DagType g;
        std::vector<std::string> m_nodes;
    };

    using GaussianNetwork = BayesianNetwork<BayesianNetworkType::GAUSSIAN_NETWORK>;

}




#endif //PGM_DATASET_BAYESIANNETWORK_HPP