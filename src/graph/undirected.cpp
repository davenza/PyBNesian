#include <graph/undirected.hpp>
#include <boost/dynamic_bitset.hpp>

using boost::dynamic_bitset;

namespace graph {

    UndirectedGraph UndirectedGraph::Complete(const std::vector<std::string>& nodes) {
        UndirectedGraph un;

        std::unordered_set<int> neighbors;
        for(int i = 0; i < nodes.size(); ++i) {
            neighbors.insert(i);
        }

        un.m_nodes.reserve(nodes.size());
        for (int i = 0; i < nodes.size(); ++i) {
            neighbors.erase(i);
            UNode n(i, nodes[i], neighbors);
            un.m_nodes.push_back(n);
            un.m_indices.insert(std::make_pair(nodes[i], i));
            neighbors.insert(i);
        }

        for (int i = 0; i < (nodes.size()-1); ++i) {
            for (int j = i+1; i < nodes.size(); j++) {
                un.m_edges.insert({i, j});
            }
        }

        return un;
    }

    std::vector<std::string> UndirectedGraph::nodes() const {
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

    EdgeVector UndirectedGraph::edges() const {
        EdgeVector res;
        res.reserve(m_edges.size());

        for (auto& edge : m_edges) {
            res.push_back({m_nodes[edge.first].name(), m_nodes[edge.second].name()});
        }

        return res;
    }

    std::vector<std::string> UndirectedGraph::neighbors(const UNode& n) const {
        std::vector<std::string> res;

        const auto& neighbors_indices = n.neighbors();
        res.reserve(neighbors_indices.size());

        for (auto node : neighbors_indices) {
            res.push_back(m_nodes[node].name());
        }

        return res;
    }

    void UndirectedGraph::add_node(const std::string& node) {
        int idx = [this, &node]() {
            if (!free_indices.empty()) {
                int idx = free_indices.back();
                free_indices.pop_back();
                UNode n(idx, node);
                m_nodes[idx] = n;
                return idx;
            }
            else {
                int idx = m_nodes.size();
                UNode n(idx, node);
                m_nodes.push_back(n);
                return idx;
            }
        }();

        m_indices.insert(std::make_pair(node, idx));
    }

    void UndirectedGraph::remove_node_unsafe(int index) {
        for (auto neighbor : m_nodes[index].neighbors()) {
            m_edges.erase(std::make_pair(index, neighbor));
            m_nodes[neighbor].remove_neighbor(index);
        }

        m_indices.erase(m_nodes[index].name());
        m_nodes[index].invalidate();
        free_indices.push_back(index);
    }

    void UndirectedGraph::add_edge_unsafe(int source, int target) {
        if (!has_edge_unsafe(source, target)) {
            m_edges.insert({source, target});
            m_nodes[source].add_neighbor(target);
            m_nodes[target].add_neighbor(source);
        }
    }

    void UndirectedGraph::remove_edge_unsafe(int source, int target) {
        if (has_edge_unsafe(source, target)) {
            m_edges.erase({source, target});
            m_nodes[source].remove_neighbor(target);
            m_nodes[target].remove_neighbor(source);
        }
    }

    bool UndirectedGraph::has_path_unsafe(int source, int target) const {
        if (has_edge_unsafe(source, target)) {
            return true;
        } else {
            dynamic_bitset to_visit(m_nodes.size());
            dynamic_bitset in_stack(m_nodes.size());

            to_visit.set(0, m_nodes.size());
            in_stack.reset(0, m_nodes.size());

            for (auto free : free_indices) {
                to_visit.reset(free);
                in_stack.set(free);
            }

            to_visit.reset(source);
            in_stack.set(source);

            const auto& neighbors = m_nodes[source].neighbors();
            std::vector<int> stack {neighbors.begin(), neighbors.end()};

            for (auto neighbor : neighbors) {
                in_stack.set(neighbor);
            }

            while(!stack.empty()) {
                auto v = stack.back();
                stack.pop_back();

                const auto& neighbors = m_nodes[v].neighbors();

                if (neighbors.find(target) != neighbors.end())
                    return true;

                to_visit.reset(v);

                for (auto neighbor : neighbors) {
                    if (!in_stack[neighbor]) {
                        stack.push_back(neighbor);
                        in_stack.set(neighbor);
                    }
                }
            }

            return false;
        }
    }
}