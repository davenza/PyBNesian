#include <graph/dag.hpp>


namespace graph {


    std::vector<std::string> DirectedGraph::nodes() const {
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

    ArcVector DirectedGraph::arcs() const {
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

    std::vector<std::string> DirectedGraph::parents(const DNode& n) const {
        std::vector<std::string> res;

        const auto& parent_indices = n.parents();
        res.reserve(parent_indices.size());

        for (auto node : parent_indices) {
            res.push_back(m_nodes[node].name());
        }

        return res;
    }

    std::string DirectedGraph::parents_to_string(const DNode& n) const {
        const auto& pa = n.parents();
        if (!pa.empty()) {
            std::string str = "[" + m_nodes[*pa.begin()].name();
            for (auto it = ++pa.begin(); it != pa.end(); ++it) {
                str += ", " + m_nodes[*it].name();
            }
            str += "]";
            return str;
        } else {
            return "[]";
        }
    }

    void DirectedGraph::add_node(const std::string& node) {
        int idx = [this, &node]() {
            if (!free_indices.empty()) {
                int idx = free_indices.back();
                free_indices.pop_back();
                DNode n(idx, node);
                m_nodes[idx] = n;
                return idx;
            }
            else {
                int idx = m_nodes.size();
                DNode n(idx, node);
                m_nodes.push_back(n);
                return idx;
            }
        }();

        m_indices.insert(std::make_pair(node, idx));
        m_roots.insert(idx);
        m_leaves.insert(idx);
    }

    void DirectedGraph::remove_node_unsafe(int index) {
        for (auto p : m_nodes[index].parents()) {
            m_nodes[p].remove_children(index);
        }

        for (auto ch : m_nodes[index].children()) {
            m_nodes[ch].remove_parent(index);
        }

        if (m_nodes[index].is_root()) {
            m_roots.erase(index);
        }

        if (m_nodes[index].is_leaf()) {
            m_leaves.erase(index);
        }
        
        m_indices.erase(m_nodes[index].name());
        m_num_arcs -= m_nodes[index].parents().size() + m_nodes[index].children().size();
        m_nodes[index].invalidate();
        free_indices.push_back(index);
    }

    void DirectedGraph::add_arc_unsafe(int source, int target) {
        if (!has_arc_unsafe(source, target)) {
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

    void DirectedGraph::remove_arc_unsafe(int source, int target) {
        if (has_arc_unsafe(source, target)) {
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

    void DirectedGraph::flip_arc_unsafe(int source, int target) {
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

    bool DirectedGraph::can_add_arc_unsafe(int source, int target) const {
        if (num_parents_unsafe(source) == 0 || 
            num_children_unsafe(target) == 0 || 
            !has_path_unsafe(target, source)) {
            return true;
        }
        return false;
    }


    bool DirectedGraph::can_flip_arc_unsafe(int source, int target) {
        if (num_parents_unsafe(target) == 0 || num_children_unsafe(source) == 0) {
            return true;
        } else {
            remove_arc_unsafe(source, target);
            bool thereis_path = has_path_unsafe(source, target);
            add_arc_unsafe(source, target);
            if (thereis_path) {
                return false;
            } else {
                return true;
            }
        }
    }

    bool DirectedGraph::has_path_unsafe(int source, int target) const {
        if (has_arc_unsafe(source, target))
            return true;
        else {
            const auto& children = m_nodes[source].children();
            std::vector<int> stack {children.begin(), children.end()};

            while (!stack.empty()) {
                auto v = stack.back();
                stack.pop_back();

                const auto& children = m_nodes[v].children();
        
                if (children.find(target) != children.end())
                    return true;
                
                stack.insert(stack.end(), children.begin(), children.end());
            }

            return false;
        }
    }

    std::vector<std::string> DirectedGraph::topological_sort() const {
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
}