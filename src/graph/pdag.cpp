#include <graph/pdag.hpp>
#include <boost/dynamic_bitset.hpp>

using boost::dynamic_bitset;

namespace graph {

    std::vector<std::string> PartiallyDirectedGraph::nodes() const {
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

    EdgeVector PartiallyDirectedGraph::edges() const {
        EdgeVector res;
        res.reserve(m_edges.size());

        for (auto& edge : m_edges) {
            res.push_back({m_nodes[edge.first].name(), m_nodes[edge.second].name()});
        }

        return res;
    }

    ArcVector PartiallyDirectedGraph::arcs() const {
        ArcVector res;
        res.reserve(m_arcs.size());

        for (auto& arc : m_arcs) {
            res.push_back({m_nodes[arc.first].name(), m_nodes[arc.second].name()});
        }

        return res;
    }

    std::vector<std::string> PartiallyDirectedGraph::neighbors(const PDNode& n) const {
        std::vector<std::string> res;

        const auto& neighbors_indices = n.neighbors();
        res.reserve(neighbors_indices.size());

        for (auto node : neighbors_indices) {
            res.push_back(m_nodes[node].name());
        }

        return res;
    }

    std::vector<std::string> PartiallyDirectedGraph::parents(const PDNode& n) const {
        std::vector<std::string> res;

        const auto& parent_indices = n.parents();
        res.reserve(parent_indices.size());

        for (auto node : parent_indices) {
            res.push_back(m_nodes[node].name());
        }

        return res;
    }

    void PartiallyDirectedGraph::add_node(const std::string& node) {
        int idx = [this, &node]() {
            if (!free_indices.empty()) {
                int idx = free_indices.back();
                free_indices.pop_back();
                PDNode n(idx, node);
                m_nodes[idx] = n;
                return idx;
            }
            else {
                int idx = m_nodes.size();
                PDNode n(idx, node);
                m_nodes.push_back(n);
                return idx;
            }
        }();

        m_indices.insert(std::make_pair(node, idx));
    }

    void PartiallyDirectedGraph::remove_node_unsafe(int index) {
        for (auto neighbor : m_nodes[index].neighbors()) {
            m_edges.erase(std::make_pair(index, neighbor));
            m_nodes[neighbor].remove_neighbor(index);
        }

        for (auto p : m_nodes[index].parents()) {
            m_arcs.erase(std::make_pair(p, index));
            m_nodes[p].remove_children(index);
        }

        for (auto ch : m_nodes[index].children()) {
            m_arcs.erase(std::make_pair(index, ch));
            m_nodes[ch].remove_parent(index);
        }

        m_indices.erase(m_nodes[index].name());
        m_nodes[index].invalidate();
        free_indices.push_back(index);
    }

    void PartiallyDirectedGraph::add_edge_unsafe(int source, int target) {
        if (!has_edge_unsafe(source, target)) {
            m_edges.insert({source, target});
            m_nodes[source].add_neighbor(target);
            m_nodes[target].add_neighbor(source);
        }
    }

    void PartiallyDirectedGraph::add_arc_unsafe(int source, int target) {
        if (!has_arc_unsafe(source, target)) {
            m_arcs.insert({source, target});
            m_nodes[target].add_parent(source);
            m_nodes[source].add_children(target);
        }
    }

    void PartiallyDirectedGraph::remove_edge_unsafe(int source, int target) {
        if (has_edge_unsafe(source, target)) {
            m_edges.erase({source, target});
            m_nodes[source].remove_neighbor(target);
            m_nodes[target].remove_neighbor(source);
        }
    }

    void PartiallyDirectedGraph::remove_arc_unsafe(int source, int target) {
        if (has_arc_unsafe(source, target)) {
            m_arcs.erase({source, target});
            m_nodes[target].remove_parent(source);
            m_nodes[source].remove_children(target);
        }
    }

    void PartiallyDirectedGraph::flip_arc_unsafe(int source, int target) {
        m_arcs.erase({source, target});
        m_arcs.insert({target, source});

        m_nodes[target].remove_parent(source);
        m_nodes[source].remove_children(target);

        m_nodes[target].add_children(source);
        m_nodes[source].add_parent(target);
    }

    void PartiallyDirectedGraph::direct_unsafe(int source, int target) {
        if (has_edge_unsafe(source, target)) {
            m_edges.erase({source, target});
            m_arcs.insert({source, target});

            m_nodes[source].remove_neighbor(target);
            m_nodes[source].add_children(target);

            m_nodes[target].remove_neighbor(source);
            m_nodes[target].add_parent(source);
        } else if (has_arc_unsafe(target, source)) {
            m_arcs.insert({source, target});

            m_nodes[source].add_children(target);
            m_nodes[target].add_parent(source);
        }
    }

    void PartiallyDirectedGraph::undirect_unsafe(int source, int target) {
        if (has_arc_unsafe(source, target)) {
            m_arcs.erase({source, target});
            m_nodes[source].remove_children(target);

            m_nodes[target].remove_parent(source);
        }

        if (!has_arc_unsafe(target, source)) {
            m_edges.insert({source, target});
            m_nodes[source].add_neighbor(target);
            m_nodes[target].add_neighbor(source);
        }
    }

    // DirectedGraph PartiallyDirectedGraph::random_direct(long unsigned int seed) const {

    // }

}