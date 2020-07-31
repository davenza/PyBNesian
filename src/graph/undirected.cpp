#include <graph/undirected.hpp>
#include <boost/dynamic_bitset.hpp>

using boost::dynamic_bitset;

namespace graph {


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
        if (m_num_edges == 0)
            return res;

        res.reserve(m_num_edges);

        dynamic_bitset to_visit(m_nodes.size());

        to_visit.set(0, m_nodes.size());

        for (auto free : free_indices) {
            to_visit.reset(free);
        }

        std::vector<int> stack;

        while (!stack.empty() || !to_visit.none()) {
            if (stack.empty()) {
                stack.push_back(to_visit.find_first());
            }

            auto idx = stack.back();
            to_visit.reset(idx);
            stack.pop_back();

            const auto& neighbors = m_nodes[idx].neighbors();

            for (auto neighbor : neighbors) {
                if (to_visit[neighbor]) {
                    res.push_back(std::make_pair(m_nodes[idx].name(), m_nodes[neighbor].name()));
                    stack.push_back(neighbor);
                }
            }
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
            m_nodes[neighbor].remove_neighbor(index);
        }

        m_indices.erase(m_nodes[index].name());
        m_num_edges -= m_nodes[index].neighbors().size();
        m_nodes[index].invalidate();
        free_indices.push_back(index);
    }

    void UndirectedGraph::add_edge_unsafe(int source, int target) {
        if (!has_edge_unsafe(source, target)) {
            ++m_num_edges;

            m_nodes[source].add_neighbor(target);
            m_nodes[target].add_neighbor(source);
        }
    }

    void UndirectedGraph::remove_edge_unsafe(int source, int target) {
        if (has_edge_unsafe(source, target)) {
            --m_num_edges;

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