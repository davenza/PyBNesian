#ifndef PGM_DATASET_UNDIRECTED_HPP
#define PGM_DATASET_UNDIRECTED_HPP

#include <pybind11/pybind11.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <graph/graph_types.hpp>
#include <util/util_types.hpp>


using util::EdgeVector;

using graph::UNode, graph::EdgeHash, graph::EdgeEqualTo;

namespace graph {

    class UndirectedGraph {
    public:
        UndirectedGraph() : m_nodes(), m_edges(), m_indices(), free_indices() {}
        UndirectedGraph(const std::vector<std::string>& nodes) : m_nodes(),
                                                                    m_edges(),
                                                                    m_indices(), 
                                                                    free_indices() {
            m_nodes.reserve(nodes.size());
            
            for (size_t i = 0; i < nodes.size(); ++i) {
                UNode n(i, nodes[i]);
                m_nodes.push_back(n);
                m_indices.insert(std::make_pair(nodes[i], i));
            }
        }

        UndirectedGraph(const EdgeVector& edges) : m_nodes(),
                                                    m_edges(),
                                                    m_indices(), 
                                                    free_indices() {

            for (auto& edge : edges) {
                if (m_indices.count(edge.first) == 0) {
                    add_node(edge.first);
                }

                if (m_indices.count(edge.second) == 0) {
                    add_node(edge.second);
                }

                add_edge(edge.first, edge.second);
            }
        }

        UndirectedGraph(const std::vector<std::string>& nodes, 
                        const EdgeVector& edges) : m_nodes(), 
                                                    m_edges(),
                                                    m_indices(),
                                                    free_indices() {
            m_nodes.reserve(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                UNode n(i, nodes[i]);
                m_nodes.push_back(n);
                m_indices.insert(std::make_pair(nodes[i], i));
            }

            for (auto& edge : edges) {
                if (m_indices.count(edge.first) == 0) throw pybind11::index_error(
                    "Node \"" + edge.first + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");
            
                if (m_indices.count(edge.second) == 0) throw pybind11::index_error(
                    "Node \"" + edge.second + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");

                add_edge(edge.first, edge.second);
            }

        }

        static UndirectedGraph Complete(const std::vector<std::string>& nodes);
        
        int num_nodes() const {
            return m_nodes.size() - free_indices.size();
        }

        int num_edges() const {
            return m_edges.size();
        }

        int num_neighbors(int idx) const {
            check_valid_indices(idx);
            return num_neighbors_unsafe(idx);
        }

        int num_neighbors(const std::string& node) const {
            auto f = check_names(node);
            return num_neighbors_unsafe(f->second);
        }

        int num_neighbors_unsafe(int idx) const {
            return m_nodes[idx].neighbors().size();
        }

        std::vector<std::string> nodes() const;

        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
        }

        int index(const std::string& node) const {
            auto f = check_names(node);
            return f->second;
        }

        const std::string& name(int idx) const {
            check_valid_indices(idx);
            return m_nodes[idx].name();
        }

        bool contains_node(const std::string& name) const {
            return m_indices.count(name) > 0;
        }

        EdgeVector edges() const;
        const auto& edge_indices() const { return m_edges; }

        std::vector<std::string> neighbors(int idx) const {
            check_valid_indices(idx);
            return neighbors(m_nodes[idx]);
        }

        std::vector<std::string> neighbors(const std::string& node) const {
            auto f = check_names(node);
            return neighbors(f->second);
        }

        const std::unordered_set<int>& neighbor_indices(int idx) const {
            check_valid_indices(idx);
            return m_nodes[idx].neighbors();
        }

        const std::unordered_set<int>& neighbor_indices(const std::string& node) const {
            auto f = check_names(node);
            return neighbor_indices(f->second);
        }

        void add_node(const std::string& node);

        void remove_node(int idx) {
            check_valid_indices(idx);
            remove_node_unsafe(idx);
        }

        void remove_node(const std::string& node) {
            auto f = check_names(node);
            remove_node_unsafe(f->second);
        }

        void remove_node_unsafe(int index);

        void add_edge(int source, int target) {
            check_valid_indices(source, target);
            add_edge_unsafe(source, target);
        }

        void add_edge(const std::string& source, const std::string& target) {
            auto [f, f2] = check_names(source, target);
            add_edge_unsafe(f->second, f2->second);
        }

        void add_edge_unsafe(int source, int target);

        bool has_edge(int source, int target) const {
            check_valid_indices(source, target);
            return has_edge_unsafe(source, target);
        }

        bool has_edge(const std::string& source, const std::string& target) const {
            auto [f, f2] = check_names(source, target);
            return has_edge_unsafe(f->second, f2->second);
        }

        bool has_edge_unsafe(int source, int target) const {
            const auto& p = m_nodes[target].neighbors();
            return p.find(source) != p.end();
        }

        void remove_edge(int source, int target) {
            check_valid_indices(source, target);
            remove_edge_unsafe(source, target);

        }

        void remove_edge(const std::string& source, const std::string& target) {
            auto [f, f2] = check_names(source, target);
            remove_edge_unsafe(f->second, f2->second);
        }

        void remove_edge_unsafe(int source, int target);

        bool has_path(int source, int target) const {
            check_valid_indices(source, target);
            return has_path_unsafe(source, target);
        }

        bool has_path(const std::string& source, const std::string& target) const {
            auto [f, f2] = check_names(source, target);
            return has_path_unsafe(f->second, f2->second);
        }

        bool has_path_unsafe(int source, int target) const;

        bool is_valid(int idx) const {
            return idx >= 0 && static_cast<size_t>(idx) < m_nodes.size() && m_nodes[idx].is_valid();
        }

        void check_valid_indices(int idx) const {
            if (!is_valid(idx)) {
                throw std::invalid_argument("Node index " + std::to_string(idx) + " invalid.");
            }
        }

        void check_valid_indices(int idx1, int idx2) const {
            if (!is_valid(idx1)) {
                throw std::invalid_argument("Node index " + std::to_string(idx1) + " invalid.");
            }

            if (!is_valid(idx2)) {
                throw std::invalid_argument("Node index " + std::to_string(idx2) + " invalid.");
            }
        }

        typename std::unordered_map<std::string, int>::const_iterator check_names(const std::string& name) const {
            auto f = m_indices.find(name);
            if (f == m_indices.end()) {
                throw std::invalid_argument("Node " + name + " not present in the graph.");
            }

            return f;
        }

        std::pair<typename std::unordered_map<std::string, int>::const_iterator,
                  typename std::unordered_map<std::string, int>::const_iterator> check_names(const std::string& v1,
                                                                                            const std::string& v2) const {
            auto f = m_indices.find(v1);
            if (f == m_indices.end()) {
                throw std::invalid_argument("Node " + v1 + " not present in the graph.");
            }

            auto f2 = m_indices.find(v2);
            if (f2 == m_indices.end()) {
                throw std::invalid_argument("Node " + v2 + " not present in the graph.");
            }

            return std::make_pair(f, f2);
        }

    private:
        std::vector<std::string> neighbors(const UNode& n) const;

        std::vector<UNode> m_nodes;
        std::unordered_set<Edge, EdgeHash, EdgeEqualTo> m_edges;
        std::unordered_map<std::string, int> m_indices;
        std::vector<int> free_indices;
    };
}

#endif //PGM_DATASET_UNDIRECTED_HPP
