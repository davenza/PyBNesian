#ifndef PGM_DATASET_BAYESIANNETWORK_HPP
#define PGM_DATASET_BAYESIANNETWORK_HPP

#include <iterator>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>

using dataset::DataFrame;
using graph::AdjMatrixDag, graph::AdjListDag, graph::dag_node_iterator;
using boost::source;

using graph::arc_vector; 
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
        BayesianNetwork(const std::vector<std::string>& nodes, const arc_vector& arcs);

        static void requires(const DataFrame& df);

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

        node_descriptor node(int node_index) const {
            return g.node(node_index);
        }

        node_descriptor node(const std::string& node) const {
            return g.node(m_indices.at(node));
        }

        std::vector<std::reference_wrapper<const std::string>> get_parents(node_descriptor node) const {
            std::vector<std::reference_wrapper<const std::string>> parents;
            auto it_parents = g.get_parent_edges(node);

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

        std::vector<int> get_parent_indices(node_descriptor node) const {
            std::vector<int> parent_indices;
            auto it_parents = g.get_parent_edges(node);

            for (auto it = it_parents.first; it != it_parents.second; ++it) {
                parent_indices.push_back(g.index(g.source(*it)));
            }

            return parent_indices;
        }

        std::vector<int> get_parent_indices(int node_index) const {
            return get_parent_indices(g.node(node_index));
        }

        std::vector<int> get_parent_indices(const std::string& node) const {
            return get_parent_indices(m_indices.at(node));
        }

        bool has_edge(node_descriptor source, node_descriptor dest) const {
            return g.has_edge(source, dest);
        }

        bool has_edge(int source, int dest) const {
            return g.has_edge(g.node(source), g.node(dest));
        }

        bool has_edge(const std::string& source, const std::string& dest) const {
            return g.has_edge(m_indices.at(source), m_indices.at(dest));
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
                                                 const arc_vector& edges) 
                                                 : g(nodes.size()), m_nodes(nodes), m_indices(nodes.size())
    {
        int i = 0;
        for (const std::string& str : nodes) {
            m_indices.insert(std::make_pair(str, i));
            ++i;
        }

        for(auto edge : edges) {
            g.add_edge(node(edge.first), node(edge.second));
        }

    };

    using GaussianNetwork = BayesianNetwork<BayesianNetworkType::GAUSSIAN_NETWORK>;
    using GaussianNetworkList = BayesianNetwork<BayesianNetworkType::GAUSSIAN_NETWORK, AdjListDag>;

}




#endif //PGM_DATASET_BAYESIANNETWORK_HPP