#include <graph/generic_graph.hpp>
#include <boost/dynamic_bitset.hpp>

using boost::dynamic_bitset;

namespace graph {


    UndirectedGraph UndirectedGraph::Complete(const std::vector<std::string>& nodes) {
        UndirectedGraph un;

        std::unordered_set<int> neighbors;
        for(int i = 0, size = nodes.size(); i < size; ++i) {
            neighbors.insert(i);
        }

        un.m_nodes.reserve(nodes.size());
        for (int i = 0, size = nodes.size(); i < size; ++i) {
            neighbors.erase(i);
            UNode n(i, nodes[i], neighbors);
            un.m_nodes.push_back(n);
            un.m_indices.insert(std::make_pair(nodes[i], i));
            neighbors.insert(i);
        }

        for (int i = 0, limit = nodes.size() - 1; i < limit; ++i) {
            for (int j = i + 1, size = nodes.size(); j < size; j++) {
                un.m_edges.insert({i, j});
            }
        }

        return un;
    }

    bool UndirectedGraph::has_path_unsafe(int source, int target) const {
        if (has_edge_unsafe(source, target)) {
            return true;
        } else {
            dynamic_bitset in_stack(m_nodes.size());
            in_stack.reset(0, m_nodes.size());

            for (auto free : free_indices) {
                in_stack.set(free);
            }

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

    PartiallyDirectedGraph::Graph(Graph<Undirected>&& g) : GraphBase<PartiallyDirectedGraph>(),
                                                           ArcGraph<PartiallyDirectedGraph>(), 
                                                           EdgeGraph<PartiallyDirectedGraph>() {
        m_nodes.reserve(g.m_nodes.size());

        for (auto& unnode : g.m_nodes) {
            m_nodes.push_back(std::move(unnode));
        }

        m_edges = std::move(g.m_edges);
        m_indices = std::move(g.m_indices);
        free_indices = std::move(g.free_indices);
    }

    PartiallyDirectedGraph::Graph(Graph<Directed>&& g) : GraphBase<PartiallyDirectedGraph>(),
                            ArcGraph<PartiallyDirectedGraph>(), 
                            EdgeGraph<PartiallyDirectedGraph>() {    
        m_nodes.reserve(g.m_nodes.size());

        for (auto& dnode : g.m_nodes) {
            m_nodes.push_back(std::move(dnode));
        }

        m_arcs = std::move(g.m_arcs);
        m_indices = std::move(g.m_indices);
        free_indices = std::move(g.free_indices);
    }

    void PartiallyDirectedGraph::direct_unsafe(int source, int target) {
        if (has_edge_unsafe(source, target)) {
            remove_edge_unsafe(source, target);
            add_arc_unsafe(source, target);
        } else if (has_arc_unsafe(target, source)) {
            add_arc_unsafe(source, target);
        }
    }

    void PartiallyDirectedGraph::undirect_unsafe(int source, int target) {
        if (has_arc_unsafe(source, target))
            remove_arc_unsafe(source, target);

        if (!has_arc_unsafe(target, source)) {
            add_edge_unsafe(source, target);
        }
    }

    bool is_new_vstructure(const PartiallyDirectedGraph& g, int source, int target) {
        const auto& parents_target = g.node(target).parents();

        for (auto p : parents_target) {
            if (!g.has_connection(source, target))
                return true;
        }

        return false;
    }

    DirectedGraph PartiallyDirectedGraph::random_direct() const {
        DirectedGraph directed(nodes());

        for (const auto& arc : arc_indices()) {
            directed.add_arc_unsafe(arc.first, arc.second);
        }

        if (num_edges() > 0) {
            PartiallyDirectedGraph copy(*this);

            while (copy.num_nodes() > 0) {

            }

        }

        return directed;
    }

    std::vector<std::string> Dag::topological_sort() const {
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

        std::vector<int> stack{roots().begin(), roots().end()};

        while (!stack.empty()) {
            auto idx = stack.back();
            stack.pop_back();

            top_sort.push_back(name(idx));
            
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

    bool Dag::can_add_arc_unsafe(int source, int target) const {
        if (num_parents_unsafe(source) == 0 || 
            num_children_unsafe(target) == 0 || 
            !has_path_unsafe(target, source)) {
            return true;
        }
        return false;
    }

    bool Dag::can_flip_arc_unsafe(int source, int target) {
        if (has_arc_unsafe(source, target)) {
            if (num_parents_unsafe(target) == 1 || num_children_unsafe(source) == 1)
                return true;

            remove_arc_unsafe(source, target);
            bool thereis_path = has_path_unsafe(source, target);
            add_arc_unsafe(source, target);
            if (thereis_path) {
                return false;
            } else {
                return true;
            }
        } else {
            if (num_parents_unsafe(target) == 0 || num_children_unsafe(source) == 0)
                return true;

            bool thereis_path = has_path_unsafe(source, target);
            if (thereis_path) {
                return false;
            } else {
                return true;
            }
        }
    }
}