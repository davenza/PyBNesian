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

            for (auto free : m_free_indices) {
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
        m_roots = std::unordered_set<int>();
        m_leaves = std::unordered_set<int>();

        for (size_t i = 0; i < g.m_nodes.size(); ++i) {
            if (g.is_valid(i)) {
                m_roots.insert(i);
                m_leaves.insert(i);
            }
        }

        m_indices = std::move(g.m_indices);
        m_free_indices = std::move(g.m_free_indices);
    }

    PartiallyDirectedGraph::Graph(Graph<Directed>&& g) : GraphBase<PartiallyDirectedGraph>(),
                            ArcGraph<PartiallyDirectedGraph>(), 
                            EdgeGraph<PartiallyDirectedGraph>() {    
        m_nodes.reserve(g.m_nodes.size());

        for (auto& dnode : g.m_nodes) {
            m_nodes.push_back(std::move(dnode));
        }

        m_arcs = std::move(g.m_arcs);
        m_roots = std::move(g.m_roots);
        m_leaves = std::move(g.m_leaves);
        m_indices = std::move(g.m_indices);
        m_free_indices = std::move(g.m_free_indices);
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

    bool pdag2dag_adjacent_node(const PartiallyDirectedGraph& g,
                                const std::vector<int>& x_neighbors, 
                                const std::vector<int>& x_parents) {
                               
        for (auto y : x_neighbors) {
            for (auto x_adj : x_neighbors) {
                if (y != x_adj && !g.has_connection_unsafe(y, x_adj))
                    return false;
            }

            for (auto x_adj : x_parents) {
                if (y != x_adj && !g.has_connection_unsafe(y, x_adj))
                    return false;
            }
        }

        return true;
    }

    Dag PartiallyDirectedGraph::to_dag() const {
        // PDAG-TO-DAG by D.Dor, M.Tarsi (1992). A simple algorithm to construct a consistent extension of a partially oriented graph.
        Dag directed(nodes());

        for (const auto& arc : arc_indices()) {
            directed.add_arc_unsafe(arc.first, arc.second);
        }
        
        if (num_edges() > 0) {
            PartiallyDirectedGraph copy(*this);

            while (copy.num_edges() > 0) {
                bool ok = false;
                for (auto x : copy.leaves()) {
                    auto x_neighbors = copy.neighbor_indices(x);
                    auto x_parents = copy.parent_indices(x);

                    if (pdag2dag_adjacent_node(copy, x_neighbors, x_parents)) {
                        for (auto neighbor : x_neighbors) {
                            directed.add_arc(neighbor, x);
                        }

                        copy.remove_node(x);
                        ok = true;
                        break;
                    }
                }

                if (!ok) {
                    throw std::invalid_argument("PDAG do not allow a valid DAG extension.");
                }
            }
        }

        return directed;
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

    std::vector<Arc> sort_arcs(const Dag& g) {
        auto top_sort = g.topological_sort();
        std::vector<int> top_rank(top_sort.size());

        for (auto i = 0; i < top_sort.size(); ++i) {
            top_rank[g.index(top_sort[i])] = i;
        }

        std::vector<Arc> res;
        res.reserve(g.num_arcs());

        int included_arcs = 0;
        for (auto i = 0; i < top_sort.size() && included_arcs < g.num_arcs(); ++i) {
            auto p = g.parent_indices(top_sort[i]);

            std::sort(p.begin(), p.end(), [&top_rank](int a, int b) {
                return top_rank[a] < top_rank[b];
            });

            auto x_name = g.index(top_sort[i]);
            for(auto j = 0; j < p.size(); ++j) {
                res.push_back({p[j], x_name});
            }
        }

        return res;
    }


    PartiallyDirectedGraph Dag::to_pdag() const {
        // DAG-to-PDAG by Chickering (2002). Learning Equivalence Classes of Bayesian-Network Structures.
        std::vector<Arc> sorted_arcs = sort_arcs(*this);
        PartiallyDirectedGraph pdag(nodes());

        for (auto i = 0; i < sorted_arcs.size() && (pdag.num_arcs() + pdag.num_edges()) < num_arcs(); ++i) {
            auto x = sorted_arcs[i].first;
            auto y = sorted_arcs[i].second;
            if (!pdag.has_arc(x, y) && !pdag.has_edge(x, y)) {
                bool done = false;
                for (auto w : pdag.parent_set(x)) {
                    if (!has_arc(w, y)) {
                        for (auto z : parent_set(y)) {
                            pdag.add_arc(z, y);
                        }

                        done = true;
                        break;
                    } else {
                        pdag.add_arc(w, y);
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
                            pdag.add_arc(z, y);
                        }
                    } else {
                        for (auto z : parent_set(y)) {
                            pdag.add_edge(z, y);
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