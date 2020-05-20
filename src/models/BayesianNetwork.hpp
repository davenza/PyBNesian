#ifndef PGM_DATASET_BAYESIANNETWORK_HPP
#define PGM_DATASET_BAYESIANNETWORK_HPP

#include <iterator>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>

using dataset::DataFrame;
using graph::AdjMatrixDag, graph::AdjListDag, graph::dag_node_iterator;
using boost::source;

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

        BayesianNetwork(const std::vector<std::string>& nodes);
        BayesianNetwork(const std::vector<std::string>& nodes, std::vector<std::pair<std::string, std::string>>& arcs);

        static void requires(const DataFrame& df);

        // dag_node_iterator<node_iterator_t> nodes() const { 
        //     return g.nodes();
        // }

        nodes_size_type num_nodes() const {
            return g.num_nodes();
        }

        edges_size_type num_edges() const {
            return g.num_edges();
        }

        const std::vector<std::string>& nodes() const {
            return m_nodes;
        }

        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
        }

        const std::string& node(int node_index) const {
            return m_nodes[node_index];
        }

        std::vector<std::reference_wrapper<const std::string>> get_parents(node_descriptor node) const {
            std::vector<std::reference_wrapper<const std::string>> parents;
            auto it_parents = g.get_parents(node);

            for (auto it = it_parents.first; it != it_parents.second; ++it) {
                auto parent = g.source(*it);
                auto parent_index = g.index(parent);
                parents.push_back(m_nodes[parent_index]);
            }

            return parents;
        }

        std::vector<std::reference_wrapper<const std::string>> get_parents(int node_index) const {
            return get_parents(g.node(node_index));
        }

        std::vector<std::reference_wrapper<const std::string>> get_parents(const std::string& node) const {
            return get_parents(m_indices.at(node));
        }

        std::vector<int> get_parent_indices(node_descriptor node) {
            std::vector<int> parent_indices;
            auto it_parents = g.get_parents(node);

            for (auto it = it_parents.first; it != it_parents.second; ++it) {
                parent_indices.push_back(g.index(*it));
            }

            return parent_indices;
        }

        std::vector<int> get_parent_indices(int node_index) const {
            return get_parent_indices(g.node(node_index));
        }

        std::vector<int> get_parent_indices(const std::string& node) const {
            return get_parent_indices(m_indices.at(node));
        }


        
        void print() {
            g.print();
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
        // Change to FNV hash function?
        std::unordered_map<std::string, int> m_indices;
    };

    template<BayesianNetworkType T, typename DagType>
    BayesianNetwork<T, DagType>::BayesianNetwork(const std::vector<std::string>& nodes) : g(nodes.size()), m_nodes(nodes), m_indices(nodes.size()) {
        int i = 0;
        for (const std::string& str : nodes) {
            m_indices.insert(std::make_pair(str, i));
            ++i;
        }
    };

    template<BayesianNetworkType T, typename DagType>
    BayesianNetwork<T, DagType>::BayesianNetwork(const std::vector<std::string>& nodes, 
                                                 std::vector<std::pair<std::string, std::string>>& arcs) 
                                                 : g(nodes.size(), arcs), m_nodes(nodes), m_indices(nodes.size())
    {
        int i = 0;
        for (const std::string& str : nodes) {
            m_indices.insert(std::make_pair(str, i));
            ++i;
        }
    };

    using GaussianNetwork = BayesianNetwork<BayesianNetworkType::GAUSSIAN_NETWORK>;
    using GaussianNetworkList = BayesianNetwork<BayesianNetworkType::GAUSSIAN_NETWORK, AdjListDag>;

}




#endif //PGM_DATASET_BAYESIANNETWORK_HPP