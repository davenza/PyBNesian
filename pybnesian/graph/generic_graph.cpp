#include <graph/generic_graph.hpp>
#include <util/vector.hpp>

namespace graph {

PartiallyDirectedGraph PartiallyDirectedGraph::CompleteUndirected(const std::vector<std::string>& nodes) {
    PartiallyDirectedGraph pdag(nodes);

    for (int i = 0, limit = nodes.size() - 1; i < limit; ++i) {
        for (int j = i + 1, size = nodes.size(); j < size; ++j) {
            pdag.add_edge_unsafe(i, j);
        }
    }

    return pdag;
}

ConditionalPartiallyDirectedGraph ConditionalPartiallyDirectedGraph::CompleteUndirected(
    const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) {
    ConditionalPartiallyDirectedGraph cpdag(nodes, interface_nodes);

    for (int i = 0, limit = nodes.size() - 1; i < limit; ++i) {
        auto node_index = cpdag.index(nodes[i]);
        for (int j = i + 1, size = nodes.size(); j < size; ++j) {
            auto other_node_index = cpdag.index(nodes[j]);
            cpdag.add_edge_unsafe(node_index, other_node_index);
        }
    }

    for (const auto& node : nodes) {
        auto node_index = cpdag.index(node);
        for (const auto& inode : interface_nodes) {
            auto inode_index = cpdag.index(inode);
            cpdag.add_edge_unsafe(node_index, inode_index);
        }
    }

    return cpdag;
}

void ConditionalPartiallyDirectedGraph::direct_interface_edges() {
    for (const auto& inode : this->interface_nodes()) {
        auto iindex = this->index(inode);

        auto nbr_set = this->neighbor_set(iindex);

        for (auto nbr : nbr_set) {
            this->direct_unsafe(iindex, nbr);
        }
    }
}

UndirectedGraph UndirectedGraph::Complete(const std::vector<std::string>& nodes) {
    UndirectedGraph un(nodes);

    for (int i = 0, limit = nodes.size() - 1; i < limit; ++i) {
        for (int j = i + 1, size = nodes.size(); j < size; ++j) {
            un.add_edge_unsafe(i, j);
        }
    }

    return un;
}

ConditionalUndirectedGraph ConditionalUndirectedGraph::Complete(const std::vector<std::string>& nodes,
                                                                const std::vector<std::string>& interface_nodes) {
    ConditionalUndirectedGraph un(nodes, interface_nodes);

    for (int i = 0, limit = nodes.size() - 1; i < limit; ++i) {
        auto node_index = un.index(nodes[i]);
        for (int j = i + 1, size = nodes.size(); j < size; ++j) {
            auto other_node_index = un.index(nodes[j]);
            un.add_edge_unsafe(node_index, other_node_index);
        }
    }

    for (const auto& node : nodes) {
        auto node_index = un.index(node);
        for (const auto& inode : interface_nodes) {
            auto inode_index = un.index(inode);
            un.add_edge_unsafe(node_index, inode_index);
        }
    }

    return un;
}

}  // namespace graph
