#ifndef PGM_DATASET_NEWDAG_HPP
#define PGM_DATASET_NEWDAG_HPP

#include <iostream>
#include <pybind11/pybind11.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <util/util_types.hpp>

using util::ArcVector;

namespace graph {

    class Node {
    public:
        Node(int idx,
             std::string name,
             std::unordered_set<int> parents = {}, 
             std::unordered_set<int> children = {}) : m_idx(idx), 
                                                m_name(name), 
                                                m_parents(parents), 
                                                m_children(children) {}
        
        void debug_status(std::ostream& os) const {
            os << "[" << m_idx << ", " << m_name << "] parents: [";

            for (auto parent : m_parents) {
                os << parent << ", ";
            }
            
            os << "]; children: [";

            for (auto children : m_children) {
                os << children << ", ";
            }
            os << "]" << std::endl;
        }

        const std::string& name() const {
            return m_name;
        }

        const std::unordered_set<int>& parents() const {
            return m_parents;
        }

        const std::unordered_set<int>& children() const {
            return m_children;
        }

        void add_parent(int p) {
            m_parents.insert(p);
        }

        void add_children(int ch) {
            m_children.insert(ch);
        }

        void remove_parent(int p) {
            m_parents.erase(p);
        }

        void remove_children(int ch) {
            m_children.erase(ch);
        }

        bool is_root() const {
            return m_parents.empty();
        }

        bool is_leaf() const {
            return m_children.empty();
        }

        void invalidate() {
            m_idx = -1;
            m_name.clear();
            m_parents.clear();
            m_children.clear();
        }

        bool is_valid() const {
            return m_idx != -1;
        }
    private:
        int m_idx;
        std::string m_name;
        std::unordered_set<int> m_parents;
        std::unordered_set<int> m_children;
    };

    // class Arc {
    // public:
    //     Arc(int source, int target) : m_source(source), m_target(target) {}

    //     int source() const {
    //         return m_source;
    //     }

    //     int target() const {
    //         return m_target;
    //     }
    // private:
    //     int m_source;
    //     int m_target;
    // };

    class DirectedGraph {
    public:
        DirectedGraph() : m_nodes(), m_num_arcs(0), m_indices(), m_roots(), m_leaves(), free_indices() {}

        DirectedGraph(const std::vector<std::string>& nodes) : m_nodes(), 
                                                                m_num_arcs(0),
                                                                m_indices(),
                                                                m_roots(), 
                                                                m_leaves(), 
                                                                free_indices() {
            m_nodes.reserve(nodes.size());
            m_roots.reserve(nodes.size());
            m_leaves.reserve(nodes.size());
            for (int i = 0; i < nodes.size(); ++i) {
                Node n(i, nodes[i]);
                m_nodes.push_back(n);
                m_indices.insert(std::make_pair(nodes[i], i));
                m_roots.insert(i);
                m_leaves.insert(i);
            }
        };

        DirectedGraph(const ArcVector& arcs) : m_nodes(), 
                                                m_num_arcs(0),
                                                m_indices(), 
                                                m_roots(), 
                                                m_leaves(), 
                                                free_indices() {
            if (!arcs.empty()) {
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

            is_dag();
        }

        DirectedGraph(const std::vector<std::string>& nodes, 
                      const ArcVector& arcs) : m_nodes(), 
                                                m_num_arcs(0),
                                                m_indices(),
                                                m_roots(), 
                                                m_leaves(), 
                                                free_indices() {
            m_nodes.reserve(nodes.size());
            m_roots.reserve(nodes.size());
            m_leaves.reserve(nodes.size());
            for (int i = 0; i < nodes.size(); ++i) {
                Node n(i, nodes[i]);
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

            is_dag();
        }

        void debug_status(std::ostream& os) const {
            os << "Directed Graph" << std::endl;
            os << "Nodes:" << std::endl;

            for (const auto& n : m_nodes) {
                n.debug_status(os);
            }

            os << "Num arcs: " << m_num_arcs << std::endl;

            os << "Indices: ";
            for (const auto& in : m_indices) {
                os << "(" << in.first << ", " << in.second << "), ";
            }
            os << std::endl;

            os << "Roots: ";
            for (const auto r : m_roots) {
                os << m_nodes[r].name() << ", ";
            }
            os << std::endl;


            os << "Leaves: ";
            for (const auto l : m_leaves) {
                os << m_nodes[l].name() << ", ";
            }
            os << std::endl;

            os << "Free indices: ";
            for (const auto f : free_indices) {
                os << f << ", ";
            }
            os << std::endl;
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
            return m_num_arcs;
        }

        int num_parents(int idx) const {
            check_valid_indices(idx);
            return m_nodes[idx].parents().size();
        }

        int num_parents(const std::string& node) const {
            return num_parents(m_indices.at(node));
        }

        int num_children(int idx) const {
            check_valid_indices(idx);
            return m_nodes[idx].children().size();
        }

        int num_children(const std::string& node) const {
            return num_children(m_indices.at(node));
        }

        // const std::vector<Node>& nodes() const {
        //     return m_nodes;
        // }
        std::vector<std::string> nodes() const {
            int visited_nodes = 0;
            std::vector<std::string> res;
            res.reserve(num_nodes());
            for (auto it = m_nodes.begin(); visited_nodes < num_nodes(); ++it) {
                if (it->is_valid()) {
                    res.push_back(it->name());
                    ++visited_nodes;
                }
            }

            return res;
        }
        
        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
        }

        int index(const std::string& node) const {
            return m_indices.at(node);
        }

        bool contains_node(const std::string& name) const {
            return m_indices.count(name) > 0;
        }

        ArcVector arcs() const {
            ArcVector res;
            res.reserve(m_num_arcs);

            std::vector<int> stack {m_roots.begin(), m_roots.end()};

            while (!stack.empty()) {
                auto idx = stack.back();
                stack.pop_back();
                const auto& ch = m_nodes[idx].children();

                for (auto children : ch) {
                    res.push_back(std::make_pair(m_nodes[idx].name(), m_nodes[children].name()));
                    stack.push_back(children);
                }
            }

            return res;
        }

        std::vector<std::string> parents(int idx) const {
            check_valid_indices(idx);
            std::vector<std::string> res;

            const auto& parent_indices = m_nodes[idx].parents();
            res.reserve(parent_indices.size());

            for (auto node : parent_indices) {
                res.push_back(m_nodes[node].name());
            }

            return res;
        }

        std::vector<std::string> parents(const std::string& node) const {
            return parents(m_indices.at(node));
        }

        std::vector<int> parent_indices(int idx) const {
            check_valid_indices(idx);
            const auto& p = m_nodes[idx].parents();
            return { p.begin(), p.end() };
        }

        std::vector<int> parent_indices(const std::string& node) const {
            return parent_indices(m_indices.at(node));
        }

        std::string parents_to_string(int idx) const;

        std::string parents_to_string(const std::string& node) const {
            return parents_to_string(m_indices.at(node));
        }

        const std::string& name(int idx) const {
            check_valid_indices(idx);
            return m_nodes[idx].name();
        }

        void add_node(const std::string& node) {
            int idx = [this]() {
                if (!free_indices.empty()) {
                    int idx = free_indices.back();
                    free_indices.pop_back();
                    return idx;
                }
                else {
                    return num_nodes();
                }
            }();

            Node n(idx, node);
            m_nodes.push_back(n);
            m_indices.insert(std::make_pair(node, idx));
            m_roots.insert(idx);
            m_leaves.insert(idx);
        }

        void remove_node(const std::string& node) {
            remove_node(m_indices.at(node));
        }

        void remove_node(int node) {
            check_valid_indices(node);
            
            for (auto p : m_nodes[node].parents()) {
                m_nodes[p].remove_children(node);
            }

            for (auto ch : m_nodes[node].children()) {
                m_nodes[ch].remove_parent(node);
            }

            if (m_nodes[node].is_root()) {
                m_roots.erase(node);
            }

            if (m_nodes[node].is_leaf()) {
                m_leaves.erase(node);
            }
            
            m_indices.erase(m_nodes[node].name());
            m_num_arcs -= m_nodes[node].parents().size() + m_nodes[node].children().size();
            m_nodes[node].invalidate();
            free_indices.push_back(node);
        }

        void add_arc(int source, int target) {
            check_valid_indices(source, target);
            
            if (!has_arc(source, target)) {
                if (m_nodes[target].is_root()) {
                    m_roots.erase(target);
                }

                if (m_nodes[source].is_leaf()) {
                    m_leaves.erase(source);
                }

                ++m_num_arcs;
                m_nodes[target].add_parent(source);
                m_nodes[source].add_children(target);
            }
        }

        void add_arc(const std::string& source, const std::string& target) {
            add_arc(m_indices.at(source), m_indices.at(target));
        }

        bool has_arc(int source, int target) const {
            check_valid_indices(source, target);
            
            const auto& p = m_nodes[target].parents();
            return p.find(source) != p.end();
        }

        bool has_arc(const std::string& source, const std::string& target) const {
            return has_arc(m_indices.at(source), m_indices.at(target));
        }

        void remove_arc(int source, int target) {
            check_valid_indices(source, target);
         
            if (has_arc(source, target)) {
                --m_num_arcs;
                m_nodes[target].remove_parent(source);
                m_nodes[source].remove_children(target);

                if (m_nodes[target].is_root()) {
                    m_roots.insert(target);
                }

                if (m_nodes[source].is_leaf()) {
                    m_leaves.insert(source);
                }
            }
        }

        void remove_arc(const std::string& source, const std::string& target) {
            remove_arc(m_indices.at(source), m_indices.at(target));
        }

        void flip_arc(int source, int target) {
            check_valid_indices(source, target);

            m_nodes[target].remove_parent(source);
            m_nodes[source].remove_children(target);

            if (m_nodes[target].is_root()) {
                m_roots.insert(target);
            }

            if (m_nodes[source].is_leaf()) {
                m_leaves.insert(source);
            }
            
            if (m_nodes[target].is_leaf()) {
                m_leaves.erase(target);
            }

            if (m_nodes[source].is_root()) {
                m_roots.erase(source);
            }
            
            m_nodes[target].add_children(source);
            m_nodes[source].add_parent(target);

        }

        void flip_arc(const std::string& source, const std::string& target) {
            flip_arc(m_indices.at(source), m_indices.at(target));
        }

        bool can_add_arc(int source, int target) const {
            if (num_parents(source) == 0 || num_children(target) == 0 || !has_path(target, source)) {
                return true;
            }
            return false;
        }

        bool can_add_arc(const std::string& source, const std::string& target) const {
            return can_add_arc(m_indices.at(source), m_indices.at(target));
        }

        bool can_flip_arc(int source, int target) {
            if (num_parents(target) == 0 || num_children(source) == 0) {
                return true;
            } else {
                remove_arc(source, target);
                bool thereis_path = has_path(source, target);
                add_arc(source, target);
                if (thereis_path) {
                    return false;
                } else {
                    return true;
                }
            }
        }

        bool can_flip_arc(const std::string& source, const std::string& target) {
            return can_flip_arc(m_indices.at(source), m_indices.at(target));
        }

        bool has_path(int source, int target) const;

        bool has_path(const std::string& source, const std::string& target) const {
            return has_path(m_indices.at(source), m_indices.at(target));
        }

        bool is_valid(int idx) const {
            return idx >= 0 && idx < m_nodes.size() && m_nodes[idx].is_valid();
        }

        void check_valid_indices(int idx) const {
            if (!is_valid(idx)) {
                throw std::invalid_argument("Node index " + std::to_string(idx) + " invalid.");
            }
        }

        template<typename... Args>
        void check_valid_indices(int idx, Args... indices) const {
            if (!is_valid(idx)) {
                throw std::invalid_argument("Node index " + std::to_string(idx) + " invalid.");
            }

            check_valid_indices(indices...);
        }

        std::vector<std::string> topological_sort() const {
            std::vector<int> incoming_edges;
            incoming_edges.reserve(m_nodes.size());

            for (auto it = m_nodes.begin(); it != m_nodes.end(); ++it) {
                if (it->is_valid()) {
                    incoming_edges.push_back(it->parents().size());
                } else {
                    incoming_edges.push_back(-1);
                }
            }

            std::vector<std::string> top_sort;
            top_sort.reserve(num_nodes());

            std::vector<int> stack{m_roots.begin(), m_roots.end()};

            while (!stack.empty()) {
                auto idx = stack.back();
                stack.pop_back();

                top_sort.push_back(m_nodes[idx].name());
                
                for (const auto& children : m_nodes[idx].children()) {
                    --incoming_edges[children];
                    if (incoming_edges[children] == 0) {
                        stack.push_back(children);
                    }
                }
            }
            
            for (auto it = incoming_edges.begin(); it != incoming_edges.end(); ++it) {
                if (*it > 0) {
                    throw std::invalid_argument("Graph must be a DAG to obtain a topological sort.");
                }
            }

            return top_sort;
        }

        bool is_dag() const {
            topological_sort();
            return true;
        }

    private:
        std::vector<Node> m_nodes;
        int m_num_arcs;
        // Change to FNV hash function?
        std::unordered_map<std::string, int> m_indices;

        std::unordered_set<int> m_roots;
        std::unordered_set<int> m_leaves;
        std::vector<int> free_indices;
    };





}


#endif //PGM_DATASET_NEWDAG_HPP
