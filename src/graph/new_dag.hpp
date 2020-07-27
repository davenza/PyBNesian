#ifndef PGM_DATASET_NEWDAG_HPP
#define PGM_DATASET_NEWDAG_HPP

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
        int index() const {
            return m_idx;
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
            return m_idx == -1;
        }
    private:
        int m_idx;
        std::string m_name;
        std::unordered_set<int> m_parents;
        std::unordered_set<int> m_children;
    };

    class Arc {
    public:
        Arc(int source, int target) : m_source(source), m_target(target) {}

        int source() const {
            return m_source;
        }

        int target() const {
            return m_target
        }
    private:
        int m_source;
        int m_target;
    };

    class DirectedGraph {

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
                m_roots.push_back(i);
                m_leaves.push_back(i);
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
        }

        DirectedGraph(const std::vector<std::string>& nodes, 
                      const ArcVector& arcs) : m_nodes(), 
                                                m_arcs(0),
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
                m_roots.push_back(i);
                m_leaves.push_back(i);
            }

            m_arcs.reserve(arcs.size())
            for (auto& arc : arcs) {
                add_arc(arc.first, arc.second);
            }
        }
        
        const std::vector<int>& roots() const {
            return m_roots;
        }

        const std::vector<int>& leaves() const {
            return m_leaves;
        }

        int num_nodes() const {
            return m_nodes.size() - free_indices.size();
        }

        int num_arcs() const {
            return m_num_arcs;
        }

        int num_parents(int idx) const {
            if (is_valid(idx)) {
                return m_nodes[idx].parents().size();
            } else {
                // TODO Raise error?
            }
        }

        int num_parents(const std::string& node) const {
            num_parents(m_indices.at(node));
        }

        int num_children(int idx) const {
            if (is_valid(idx)) {
                return m_nodes[idx].parents().size();
            } else {
                // TODO Raise error?
            }
        }

        int num_children(const std::string& node) const {
            num_children(m_indices.at(node));
        }

        const std::vector<Node>& nodes() const {
            return m_nodes;
        }
        
        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
        }

        class arcs_iterator {
        public:
            using value_type = std::pair<int, int>;
            using reference = value_type&;
            using pointer = value_type*;
            // FIXME: Check the iterator category operations.
            using iterator_category = std::input_iterator_tag; //or another tag

            arcs_iterator(const DirectedGraph& graph) : g(graph) {}


        private:
            const DirectedGraph g;
            int nnodes;
        };

        // return_type edges();

        std::vector<std::string> parents(int idx) const {
            if (is_valid(idx)) {
                std::vector<std::string> res;

                const auto& parent_indices = m_nodes[idx].parents();
                res.reserve(parent_indices.size());

                for (auto node : parent_indices) {
                    res.push_back(m_nodes[node].name());
                }

                return res;
            } else {
                // TODO Raise error?
            }
        }

        std::vector<std::string> parents(const std::string& node) const {
            parents(m_indices.at(node));
        }

        const std::unordered_set<int>& parent_indices(int idx) const {
            if (is_valid(idx)) {
                return m_nodes[idx].parents();
            } else {
                // TODO Raise error?
            }
        }

        const std::unordered_set<int>& parent_indices(const std::string& node) const {
            return parent_indices(m_indices.at(node));
        }

        std::string parents_to_string(int idx) const;
        std::string parents_to_string(const std::string& node) const {
            return parents_to_string(m_indices.at(node));
        }

        int index(const std::string& node) {
            return m_indices.at(node);
        }

        const std::string& name(int idx) const {
            if (is_valid(idx)) {
                return m_nodes[idx].name();
            }
        }

        int add_node(const std::string& node) {
            if (!free_indices.empty()) {
                auto idx = free_indices.back();
                free_indices.pop_back();
                Node n(idx, node);
                m_nodes[idx] = n;
            }
            else {
                auto idx = num_nodes();
                Node n(idx, node);
                m_nodes.push_back(n);
            }
        }

        void remove_node(const std::string& node) {
            remove_node(m_indices.at(node));
        }

        void remove_node(int node) {
            if (is_valid(node)) {
                for (auto p : m_nodes[node].parents()) {
                    m_nodes[p].remove_children(node);
                }

                for (auto ch : m_nodes[node].children()) {
                    m_nodes[ch].remove_parent(node);
                }

                m_num_arcs -= m_nodes[node].parents().size() + m_nodes[node].children().size();
                m_nodes[node].invalidate();
                free_indices.push_back(node);


            } else {
                // TODO Raise error?
            }
        }

        void add_arc(int source, int target) {
            if (is_valid(source) && is_valid(target)) {
                if (!has_arc(source, target)) {
                    ++m_num_arcs;
                    m_nodes[target].add_parent(source);
                    m_nodes[source].add_children(target);
                }

            } else {
                // TODO Raise error?
            }
            
        }

        void add_arc(const std::string& source, const std::string& target) {
            add_arc(m_indices.at(source), m_indices.at(target));
        }

        bool has_arc(int source, int target) const {
            if (is_valid(source) && is_valid(target)) {
                const auto& p = m_nodes[target].parents();
                return p.find(source) != p.end();
            } else {
                // TODO Raise error?
            }
        }

        bool has_arc(const std::string& source, const std::string& target) const {
            return has_arc(m_indices.at(source), m_indices.at(target));
        }
        void remove_arc(int source, int target) {
            if (is_valid(source) && is_valid(target)) {
                if (has_arc(source, target)) {
                    --m_num_arcs;
                    m_nodes[target].remove_parent(source);
                    m_nodes[source].remove_children(target);
                }
            } else {
                // TODO Raise error?
            }
        }

        void remove_arc(const std::string& source, const std::string& target) {
            remove_arc(m_indices.at(source), m_indices.at(target));
        }

        void flip_arc(int source, int target) {
            if (is_valid(source) && is_valid(target)) {
                m_nodes[target].remove_parent(source);
                m_nodes[target].add_children(source);
                m_nodes[source].remove_children(target);
                m_nodes[source].add_parent(target);
            } else {
                // TODO Raise error?
            }
        }

        void flip_arc(const std::string& source, const std::string& target) {
            flip_arc(m_indices.at(source), m_indices.at(target));
        }

        bool can_add_arc(int source, int target) {
            if (num_parents(source) == 0 || num_children(target) == 0 || !has_path(target, source)) {
                return true;
            }
        }

        bool can_add_arc(const std::string& source, const std::string& target) {
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
        // topological_sort() const;
        
        bool is_dag() const;
    private:
        std::vector<Node> m_nodes;
        int m_num_arcs;
        // Change to FNV hash function?
        std::unordered_map<std::string, int> m_indices;

        std::vector<int> m_roots;
        std::vector<int> m_leaves;
        std::vector<int> free_indices;
    };





}


#endif //PGM_DATASET_NEWDAG_HPP
