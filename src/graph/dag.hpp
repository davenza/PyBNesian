#ifndef PGM_DATASET_DAG_HPP
#define PGM_DATASET_DAG_HPP

#include <iostream>
#include <pybind11/pybind11.h>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>


namespace py = pybind11;
using boost::adjacency_matrix, boost::adjacency_list, boost::directedS, boost::setS, boost::vecS, boost::property, boost::vertex_index_t,
    boost::property_map, boost::vertex_index;

using adj_m = adjacency_matrix<directedS, property<vertex_index_t, int>>;
using adj_l = adjacency_list<setS, vecS, directedS, property<vertex_index_t, int>>;

namespace graph {

    using arc_vector = std::vector<std::pair<std::string, std::string>>;


    template<typename it>
    class dag_node_iterator {
    public:
        using iterator_category = typename std::iterator_traits<it>::iterator_category;
        using value_type = typename std::iterator_traits<it>::value_type;
        using difference_type = typename std::iterator_traits<it>::difference_type;
        using pointer = typename std::iterator_traits<it>::pointer;
        using reference = typename std::iterator_traits<it>::reference;

        dag_node_iterator(it b, it e) : m_begin(b), m_end(e) {}

        it begin() { return m_begin; }

        it end() { return m_end; }
    private:
        it m_begin;
        it m_end;
    };

    template<typename Graph>
    class Dag {
    public:
        using node_descriptor = typename Graph::vertex_descriptor;
        using edge_descriptor = typename Graph::edge_descriptor;

        using node_iterator_t = typename boost::graph_traits<Graph>::vertex_iterator;
        using edge_iterator_t = typename boost::graph_traits<Graph>::edge_iterator;
        using in_edge_iterator_t = typename boost::graph_traits<Graph>::in_edge_iterator;

        using nodes_size_type = typename boost::graph_traits<Graph>::vertices_size_type;
        using edges_size_type = typename boost::graph_traits<Graph>::edges_size_type;

        template<typename = std::enable_if_t<std::is_default_constructible_v<Graph>>>
        Dag() = delete;

        Dag(int nnodes) : g(nnodes) { };

        // TODO Implement adding arcs.
        Dag(int nnodes, const arc_vector& arcs) : g(nnodes) {};

        nodes_size_type num_nodes() const {
            return num_vertices(g);
        }

        edges_size_type num_edges() const {
            return num_edges(g);
        }

        std::pair<in_edge_iterator_t, in_edge_iterator_t> get_parent_edges(node_descriptor node) const {
            return in_edges(node, g);
        }

        int index(node_descriptor n) const {
            return get(boost::vertex_index, g)[n];
        }

        node_descriptor node(nodes_size_type index) const {
            return vertex(index, g);
        }

        node_descriptor source(edge_descriptor edge) const {
            return boost::source(edge, g);
        }

        node_descriptor target(edge_descriptor edge) const {
            return boost::target(edge, g);
        }

        bool has_edge(node_descriptor source, node_descriptor dest) const {
            return boost::edge(source, dest, g).second;
        }

        void add_edge(node_descriptor source, node_descriptor dest) {
            boost::add_edge(source, dest, g);
        }
        
        dag_node_iterator<node_iterator_t> nodes() const;

        void print();

    private:
        Graph g;
    };

    using AdjMatrixDag = Dag<adj_m>;
    using AdjListDag = Dag<adj_l>;

    template<typename Graph>
    void
    Dag<Graph>::print() {
        node_iterator_t nit, nend;
        edge_iterator_t eit, eend;

        std::cout << "Using dag type: " << typeid(Graph).name() << std::endl;
        for(std::tie(nit, nend) = vertices(g); nit != nend; ++nit)
            std::cout << "Descriptor: " << *nit << ", Index: " << index(*nit) << std::endl;

        for(std::tie(eit, eend) = edges(g); eit != eend; ++eit)
            std::cout << boost::source(*eit, g) << " -> " << boost::target(*eit, g) << std::endl;
    }

    template<typename Graph>
    dag_node_iterator<typename Dag<Graph>::node_iterator_t> Dag<Graph>::nodes() const {
        auto it = vertices(g);
        return dag_node_iterator(it.first, it.second);
    }
}



#endif //PGM_DATASET_DAG_HPP