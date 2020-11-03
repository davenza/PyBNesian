#ifndef PGM_DATASET_PDAG_HPP
#define PGM_DATASET_PDAG_HPP

#include <pybind11/pybind11.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <graph/graph_types.hpp>
#include <util/util_types.hpp>

using graph::PDNode, graph::Edge, graph::EdgeHash, graph::EdgeEqualTo, graph::Arc, graph::ArcHash;
using util::ArcVector, util::EdgeVector;

namespace graph {
    
    class UndirectedGraph;
    class DirectedGraph;

    class PartiallyDirectedGraph {
    public:
        PartiallyDirectedGraph() : m_nodes(), m_edges(), m_arcs(), m_indices(), free_indices() {}

        PartiallyDirectedGraph(const std::vector<std::string>& nodes) : m_nodes(),
                                                                        m_edges(),
                                                                        m_arcs(),
                                                                        m_indices(),
                                                                        free_indices() {
            m_nodes.reserve(nodes.size());
            
            for (size_t i = 0; i < nodes.size(); ++i) {
                PDNode n(i, nodes[i]);
                m_nodes.push_back(n);
                m_indices.insert(std::make_pair(nodes[i], i));
            }
        }

        PartiallyDirectedGraph(const EdgeVector& edges, const ArcVector& arcs) : m_nodes(),
                                                                                    m_edges(),
                                                                                    m_arcs(),
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

            for (auto& arc : arcs) {
                if (m_indices.count(arc.first) == 0) {
                    add_node(arc.first);
                }

                if (m_indices.count(arc.second) == 0) {
                    add_node(arc.second);
                }

                add_arc(arc.first, arc.second);
            }
        }

        PartiallyDirectedGraph(const std::vector<std::string>& nodes, 
                                const EdgeVector& edges, 
                                const ArcVector& arcs) : m_nodes(),
                                                         m_edges(),
                                                         m_arcs(),
                                                         m_indices(),
                                                         free_indices() {
            m_nodes.reserve(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                PDNode n(i, nodes[i]);
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

            for (auto& arc : arcs) {
                if (m_indices.count(arc.first) == 0) throw pybind11::index_error(
                    "Node \"" + arc.first + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");
            
                if (m_indices.count(arc.second) == 0) throw pybind11::index_error(
                    "Node \"" + arc.second + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");

                add_arc(arc.first, arc.second);
            }
        }

        PartiallyDirectedGraph(UndirectedGraph&& g);

        PartiallyDirectedGraph(DirectedGraph&& g);

        int num_nodes() const {
            return m_nodes.size() - free_indices.size();
        }

        int num_edges() const {
            return m_edges.size();
        }

        int num_arcs() const {
            return m_arcs.size();
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

        int num_parents(int idx) const {
            check_valid_indices(idx);
            return num_parents_unsafe(idx);
        }

        int num_parents(const std::string& node) const {
            auto f = check_names(node);
            return num_parents_unsafe(f->second);
        }

        int num_parents_unsafe(int idx) const {
            return m_nodes[idx].parents().size();
        }

        int num_children(int idx) const {
            check_valid_indices(idx);
            return num_children_unsafe(idx);
        }

        int num_children(const std::string& node) const {
            auto f = check_names(node);
            return num_children_unsafe(f->second);
        }

        int num_children_unsafe(int idx) const {
            return m_nodes[idx].children().size();
        }

        std::vector<std::string> nodes() const;
        const std::vector<PDNode>& node_indices() const { return m_nodes; }
        
        const PDNode& node(int idx) const {
            check_valid_indices(idx);
            return m_nodes[idx]; 
        }
        const PDNode& node(const std::string& name) const {
            auto f = check_names(name);
            return m_nodes[f->second]; 
        }

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

        ArcVector arcs() const;
        const auto& arc_indices() const { return m_arcs; }

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
            return m_nodes[f->second].neighbors();
        }

        std::vector<std::string> parents(int idx) const {
            check_valid_indices(idx);
            return parents(m_nodes[idx]);
        }

        std::vector<std::string> parents(const std::string& node) const {
            auto f = check_names(node);
            return parents(m_nodes[f->second]);
        }

        const std::unordered_set<int>& parent_indices(int idx) const {
            check_valid_indices(idx);
            return m_nodes[idx].parents();
        }

        const std::unordered_set<int>& parent_indices(const std::string& node) const {
            auto f = check_names(node);
            return m_nodes[f->second].parents();
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

        void add_arc(int source, int target) {
            check_valid_indices(source, target);
            add_arc_unsafe(source, target);
        }

        void add_arc(const std::string& source, const std::string& target) {
            auto [f, f2] = check_names(source, target);
            add_arc_unsafe(f->second, f2->second);
        }

        void add_arc_unsafe(int source, int target);

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

        bool has_arc(int source, int target) const {
            check_valid_indices(source, target);
            return has_arc_unsafe(source, target);
        }

        bool has_arc(const std::string& source, const std::string& target) const {
            auto [f, f2] = check_names(source, target);
            return has_arc_unsafe(f->second, f2->second);
        }

        bool has_arc_unsafe(int source, int target) const {
            const auto& p = m_nodes[target].parents();
            return p.find(source) != p.end();
        }
        
        bool has_connection(int source, int target) const {
            check_valid_indices(source, target);
            return has_connection_unsafe(source, target);
        }

        bool has_connection(const std::string& source, const std::string& target) const {
            auto [f, f2] = check_names(source, target);
            return has_connection_unsafe(f->second, f2->second);
        }

        bool has_connection_unsafe(int source, int target) const {
            return has_edge_unsafe(source, target) || has_arc_unsafe(source, target) || has_arc_unsafe(target, source);
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

        void remove_arc(int source, int target) {
            check_valid_indices(source, target);
            remove_arc_unsafe(source, target);

        }

        void remove_arc(const std::string& source, const std::string& target) {
            auto [f, f2] = check_names(source, target);
            remove_arc_unsafe(f->second, f2->second);
        }

        void remove_arc_unsafe(int source, int target);

        void flip_arc(int source, int target) {
            check_valid_indices(source, target);
            flip_arc_unsafe(source, target);
        }

        void flip_arc(const std::string& source, const std::string& target) {
            auto [f, f2] = check_names(source, target);
            flip_arc_unsafe(f->second, f2->second);
        }

        void flip_arc_unsafe(int source, int target);

        void direct(int source, int target) {
            check_valid_indices(source, target);
            direct_unsafe(source, target);
        }

        void direct(const std::string& source, const std::string& target) {
            auto [f, f2] = check_names(source, target);
            direct_unsafe(f->second, f2->second);
        }

        void direct_unsafe(int source, int target);

        void undirect(int source, int target) {
            check_valid_indices(source, target);
            undirect_unsafe(source, target);
        }

        void undirect(const std::string& source, const std::string& target) {
            auto [f, f2] = check_names(source, target);
            undirect_unsafe(f->second, f2->second);
        }

        void undirect_unsafe(int source, int target);

        bool is_valid(int idx) const {
            return idx >= 0 && static_cast<size_t>(idx) < m_nodes.size() && m_nodes[idx].is_valid();
        }
        
        DirectedGraph random_direct(long unsigned int seed) const;
    private:

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

        std::vector<std::string> neighbors(const PDNode& n) const;
        std::vector<std::string> parents(const PDNode& n) const;

        std::vector<PDNode> m_nodes;
        std::unordered_set<Edge, EdgeHash, EdgeEqualTo> m_edges;
        std::unordered_set<Arc, ArcHash> m_arcs;
        std::unordered_map<std::string, int> m_indices;
        std::vector<int> free_indices;
    };

}

#endif //PGM_DATASET_PDAG_HPP
