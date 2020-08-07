#ifndef PGM_DATASET_DAG_HPP
#define PGM_DATASET_DAG_HPP

#include <iostream>
#include <pybind11/pybind11.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <graph/graph_types.hpp>
#include <util/util_types.hpp>

using util::ArcVector;

using graph::DNode, graph::Arc;

namespace graph {


    class DirectedGraph {
    public:
        DirectedGraph() : m_nodes(), m_arcs(), m_indices(), m_roots(), m_leaves(), free_indices() {}

        DirectedGraph(const std::vector<std::string>& nodes) : m_nodes(), 
                                                                m_arcs(),
                                                                m_indices(),
                                                                m_roots(), 
                                                                m_leaves(), 
                                                                free_indices() {
            m_nodes.reserve(nodes.size());
            m_roots.reserve(nodes.size());
            m_leaves.reserve(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                DNode n(i, nodes[i]);
                m_nodes.push_back(n);
                m_indices.insert(std::make_pair(nodes[i], i));
                m_roots.insert(i);
                m_leaves.insert(i);
            }
        };

        DirectedGraph(const ArcVector& arcs) : m_nodes(), 
                                                m_arcs(),
                                                m_indices(), 
                                                m_roots(), 
                                                m_leaves(), 
                                                free_indices() {

            for (auto& arc : arcs) {
                if (m_indices.count(arc.first) == 0) {
                    add_node(arc.first);
                }

                if (m_indices.count(arc.second) == 0) {
                    add_node(arc.second);
                }

                add_arc(arc.first, arc.second);
            }

            topological_sort();
        }

        DirectedGraph(const std::vector<std::string>& nodes, 
                      const ArcVector& arcs) : m_nodes(), 
                                                m_arcs(),
                                                m_indices(),
                                                m_roots(), 
                                                m_leaves(), 
                                                free_indices() {
            m_nodes.reserve(nodes.size());
            m_roots.reserve(nodes.size());
            m_leaves.reserve(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                DNode n(i, nodes[i]);
                m_nodes.push_back(n);
                m_indices.insert(std::make_pair(nodes[i], i));
                m_roots.insert(i);
                m_leaves.insert(i);
            }

            for (auto& arc : arcs) {
                if (m_indices.count(arc.first) == 0) throw pybind11::index_error(
                    "Node \"" + arc.first + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");
            
                if (m_indices.count(arc.second) == 0) throw pybind11::index_error(
                    "Node \"" + arc.second + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");

                add_arc(arc.first, arc.second);
            }

            topological_sort();
        }

        const std::unordered_set<int>& roots() const {
            return m_roots;
        }

        const std::unordered_set<int>& leaves() const {
            return m_leaves;
        }

        int num_nodes() const {
            return m_nodes.size() - free_indices.size();
        }

        int num_arcs() const {
            return m_arcs.size();
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

        ArcVector arcs() const;

        std::vector<std::string> parents(int idx) const {
            check_valid_indices(idx);
            return parents(m_nodes[idx]);
        }

        std::vector<std::string> parents(const std::string& node) const {
            auto f = check_names(node);
            return parents(m_nodes[f->second]);
        }

        std::vector<int> parent_indices(int idx) const {
            check_valid_indices(idx);
            const auto& p = m_nodes[idx].parents();
            return { p.begin(), p.end() };
        }

        std::vector<int> parent_indices(const std::string& node) const {
            auto f = check_names(node);
            const auto& p = m_nodes[f->second].parents();
            return { p.begin(), p.end() };
        }

        std::string parents_to_string(int idx) const {
            check_valid_indices(idx);
            return parents_to_string(m_nodes[idx]);
        }

        std::string parents_to_string(const std::string& node) const {
            auto f = check_names(node);
            return parents_to_string(m_nodes[f->second]);
        }

        void add_node(const std::string& node);

        void remove_node(const std::string& node) {
            auto f = check_names(node);
            remove_node_unsafe(f->second);
        }

        void remove_node(int node) {
            check_valid_indices(node);
            remove_node_unsafe(node);
        }

        void remove_node_unsafe(int index);

        void add_arc(int source, int target) {
            check_valid_indices(source, target);
            add_arc_unsafe(source, target);
        }

        void add_arc(const std::string& source, const std::string& target) {
            auto [f, f2] = check_names(source, target);
            add_arc_unsafe(f->second, f2->second);
        }

        void add_arc_unsafe(int source, int target);

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

        bool can_add_arc(int source, int target) const {
            check_valid_indices(source, target);
            return can_add_arc_unsafe(source, target);
        }

        bool can_add_arc(const std::string& source, const std::string& target) const {
            auto [f, f2] = check_names(source, target);
            return can_add_arc_unsafe(f->second, f2->second);
        }

        bool can_add_arc_unsafe(int source, int target) const;

        bool can_flip_arc(int source, int target) {
            check_valid_indices(source, target);
            return can_flip_arc_unsafe(source, target);
        }

        bool can_flip_arc(const std::string& source, const std::string& target) {
            auto [f, f2] = check_names(source, target);
            return can_flip_arc_unsafe(f->second, f2->second);
        }

        bool can_flip_arc_unsafe(int source, int target);

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

        std::vector<std::string> topological_sort() const;

        bool is_dag() const {
            try {
                topological_sort();
                return true;
            } catch(std::invalid_argument&) {
                return false;
            }
        }

    private:
        std::vector<std::string> parents(const DNode& n) const;
        std::string parents_to_string(const DNode& n) const;

        std::vector<DNode> m_nodes;
        std::unordered_set<Arc, ArcHash> m_arcs;
        // Change to FNV hash function?
        std::unordered_map<std::string, int> m_indices;
        std::unordered_set<int> m_roots;
        std::unordered_set<int> m_leaves;
        std::vector<int> free_indices;
    };
}

#endif //PGM_DATASET_DAG_HPP
