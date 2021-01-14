#include <graph/generic_graph.hpp>
#include <util/vector.hpp>

namespace graph {

    PartiallyDirectedGraph PartiallyDirectedGraph::CompleteUndirected(const std::vector<std::string>& nodes) {
        PartiallyDirectedGraph pdag;

        std::unordered_set<int> neighbors;
        for(int i = 0, size = nodes.size(); i < size; ++i) {
            neighbors.insert(i);
        }
        
        // Roots and leaves are all the nodes because it is undirected.
        pdag.m_roots = neighbors;
        pdag.m_leaves = neighbors;
        pdag.m_nodes.reserve(nodes.size());
        for (int i = 0, size = nodes.size(); i < size; ++i) {
            neighbors.erase(i);
            PDNode n(i, nodes[i], {}, {}, neighbors);
            pdag.m_nodes.push_back(n);
            pdag.m_indices.insert(std::make_pair(nodes[i], i));
            neighbors.insert(i);
        }

        for (int i = 0, limit = nodes.size() - 1; i < limit; ++i) {
            for (int j = i + 1, size = nodes.size(); j < size; ++j) {
                pdag.m_edges.insert({i, j});
            }
        }

        return pdag;
    }

    ArcStringVector ConditionalPartiallyDirectedGraph::compelled_arcs() const {
        ArcStringVector res;

        for (const auto& edge : edge_indices()) {
            if (is_interface(edge.first))
                res.push_back({name(edge.first), name(edge.second)});
            else if (is_interface(edge.second))
                res.push_back({name(edge.second), name(edge.first)});
        }

        return res;
    }

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
            for (int j = i + 1, size = nodes.size(); j < size; ++j) {
                un.m_edges.insert({i, j});
            }
        }

        return un;
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

    bool Dag::can_add_arc_unsafe(int source, int target) const {
        if (num_parents_unsafe(source) == 0 || 
            num_children_unsafe(target) == 0 || 
            !has_path_unsafe(target, source)) {
            return true;
        }
        return false;
    }
    
    //  This method should be const, but it has to remove and re-add an arc to check.
    bool Dag::can_flip_arc_unsafe(int source, int target) const {
        if (has_arc_unsafe(source, target)) {
            if (num_parents_unsafe(target) == 1 || num_children_unsafe(source) == 1)
                return true;

            bool thereis_path = has_path_unsafe_no_direct_arc(source, target);
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

    std::vector<Arc> sort_arcs(const Dag& g) {
        auto top_sort = g.topological_sort();
        std::vector<int> top_rank(top_sort.size());

        for (size_t i = 0; i < top_sort.size(); ++i) {
            top_rank[g.index(top_sort[i])] = i;
        }

        std::vector<Arc> res;
        res.reserve(g.num_arcs());

        int included_arcs = 0;
        for (size_t i = 0; i < top_sort.size() && included_arcs < g.num_arcs(); ++i) {
            auto p = g.parent_indices(top_sort[i]);

            std::sort(p.begin(), p.end(), [&top_rank](int a, int b) {
                return top_rank[a] < top_rank[b];
            });

            auto x_name = g.index(top_sort[i]);
            for(size_t j = 0; j < p.size(); ++j) {
                res.push_back({p[j], x_name});
            }
        }

        return res;
    }


    PartiallyDirectedGraph Dag::to_pdag() const {
        // DAG-to-PDAG by Chickering (2002). Learning Equivalence Classes of Bayesian-Network Structures.
        std::vector<Arc> sorted_arcs = sort_arcs(*this);
        PartiallyDirectedGraph pdag(nodes());

        for (size_t i = 0; i < sorted_arcs.size() && (pdag.num_arcs() + pdag.num_edges()) < num_arcs(); ++i) {
            auto x = sorted_arcs[i].first;
            auto y = sorted_arcs[i].second;
            // Use name because Dag could have removed/invalidated nodes.
            const auto& x_name = name(x);
            const auto& y_name = name(y);
            if (!pdag.has_arc(x_name, y_name) && !pdag.has_edge(x_name, y_name)) {
                bool done = false;
                for (auto w : pdag.parent_set(x_name)) {
                    const auto& w_name = pdag.name(w);
                    if (!has_arc(w_name, y_name)) {
                        for (auto z : parent_set(y_name)) {
                            const auto& z_name = name(z);
                            pdag.add_arc(z_name, y_name);
                        }

                        done = true;
                        break;
                    } else {
                        pdag.add_arc(w_name, y_name);
                    }
                }

                if (!done) {
                    bool compelled = false;

                    for (auto z : parent_set(y)) {
                        if (z != x && !has_arc(z, x)) {
                            compelled = true;
                            break;
                        }
                    }

                    if (compelled) {
                        for (auto z : parent_set(y)) {
                            const auto& z_name = name(z);
                            pdag.add_arc(z_name, y_name);
                        }
                    } else {
                        for (auto z : parent_set(y)) {
                            const auto& z_name = name(z);
                            pdag.add_edge(z_name, y_name);
                        }
                    }
                }
            }
        }

        return pdag;
    }

    py::object load_graph(const std::string& name) {
        auto open = py::module::import("io").attr("open");
        auto file = open(name, "rb");
        auto graph = py::module::import("pickle").attr("load")(file);
        file.attr("close")();
        return graph;
    }
}
