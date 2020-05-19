#ifndef PGM_DATASET_DAG_HPP
#define PGM_DATASET_DAG_HPP

#include <iostream>
#include <pybind11/pybind11.h>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>


namespace py = pybind11;
using boost::adjacency_matrix, boost::adjacency_list, boost::directedS, boost::setS, boost::vecS, boost::property, boost::vertex_name_t,
    boost::property_map, boost::vertex_name, boost::tie;

using adj_m = adjacency_matrix<directedS, property<vertex_name_t, std::string>>;
using adj_l = adjacency_list<setS, vecS, directedS, property<vertex_name_t, std::string>>;

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

        using nodes_size_type = typename boost::graph_traits<Graph>::vertices_size_type;
        using edges_size_type = typename boost::graph_traits<Graph>::edges_size_type;

        using NameMap = typename property_map<Graph, vertex_name_t>::type;

        template<typename = std::enable_if_t<std::is_default_constructible_v<Graph>>>
        Dag() = delete;

        Dag(int n_vertex) : g(n_vertex) {
            NameMap node_name = get(vertex_name, g);
            node_iterator_t n, e;

            int i = 1;
            for (tie(n, e) = vertices(g); n != e; ++n, ++i) {
                put(node_name, *n, "n" + std::to_string(i));
            }
        };

        Dag(const std::vector<std::string>& nodes) : g(nodes.size()) {
            NameMap node_name = get(vertex_name, g);
            node_iterator_t n, e;

            int i = 0;
            for (tie(n, e) = vertices(g); n != e; ++n, ++i) {
                put(node_name, *n, nodes[i]);
            }
        };

        Dag(const std::vector<std::string>& nodes, const arc_vector& arcs) : g(nodes.size()) {
            NameMap node_name = get(vertex_name, g);
            node_iterator_t n, e;

            int i = 0;
            for (tie(n, e) = vertices(g); n != e; ++n, ++i) {
                put(node_name, *n, nodes[i]);
            }
        };

        void
        add_edge(node_descriptor u, node_descriptor v);

        void
        add_node(std::string u);

        nodes_size_type num_nodes() const {
            return num_vertices(g);
        }

        edges_size_type num_edges() const {
            return num_edges(g);
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
    Dag<Graph>::add_edge(typename Dag<Graph>::node_descriptor u, typename Dag<Graph>::node_descriptor v) {
        edge_descriptor e;
        bool added;
        tie(e, added) = boost::add_edge(u, v, g);
        std::cout << "Added edge: " << e << ", " << added << std::endl;
    }

    template<typename Graph>
    void
    Dag<Graph>::add_node(std::string u) {
        node_descriptor n;
        n = boost::add_vertex(u, g);
        std::cout << "Added node: " << n << std::endl;
    }

    template<typename Graph>
    void
    Dag<Graph>::print() {
        node_iterator_t nit, nend;
        edge_iterator_t eit, eend;

        for(boost::tie(nit, nend) = vertices(g); nit != nend; ++nit)
            std::cout << "Node: " << get(boost::vertex_name, g)[*nit] << std::endl;

        for(boost::tie(eit, eend) = edges(g); eit != eend; ++eit)
            std::cout << get(boost::vertex_name, g)[source(*eit, g)] << " -> "
              << get(boost::vertex_name, g)[target(*eit, g)] << std::endl;
    }

    template<typename Graph>
    dag_node_iterator<typename Dag<Graph>::node_iterator_t> Dag<Graph>::nodes() const {
        auto it = vertices(g);
        return dag_node_iterator(it.first, it.second);
    }


}



#endif //PGM_DATASET_DAG_HPP