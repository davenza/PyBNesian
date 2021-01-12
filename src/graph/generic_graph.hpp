#ifndef PYBNESIAN_GRAPH_GENERIC_GRAPH_HPP
#define PYBNESIAN_GRAPH_GENERIC_GRAPH_HPP

#include <pybind11/pybind11.h>
#include <graph/graph_types.hpp>
#include <util/util_types.hpp>
#include <util/vector.hpp>

namespace py = pybind11;

using graph::DNode, graph::UNode, graph::PDNode;
using util::ArcStringVector, util::EdgeStringVector, util::ArcSet, util::EdgeSet;

namespace graph {

    enum GraphType {
        Directed,
        Undirected,
        PartiallyDirected
    };

    template<GraphType Type>
    class Graph;
    class Dag;

    using DirectedGraph = Graph<Directed>;
    using UndirectedGraph = Graph<Undirected>;
    using PartiallyDirectedGraph = Graph<PartiallyDirected>;

    template<typename G>
    struct GraphTraits;

    template<>
    struct GraphTraits<DirectedGraph> {
        using NodeType = DNode;
        inline static constexpr bool has_arcs = true;
        inline static constexpr bool has_edges = false;
    };

    template<>
    struct GraphTraits<Dag> {
        using NodeType = DNode;
        inline static constexpr bool has_arcs = true;
        inline static constexpr bool has_edges = false;
    };

    template<>
    struct GraphTraits<UndirectedGraph> {
        using NodeType = UNode;
        inline static constexpr bool has_arcs = false;
        inline static constexpr bool has_edges = true;
    };

    template<>
    struct GraphTraits<PartiallyDirectedGraph> {
        using NodeType = PDNode;
        inline static constexpr bool has_arcs = true;
        inline static constexpr bool has_edges = true;
    };

    template<typename Derived>
    class ArcGraph;

    template<typename Derived>
    class GraphBase {
    public:
        using NodeType = typename GraphTraits<Derived>::NodeType;

        template<typename G>
        friend class ArcGraph;
        template<typename G>
        friend class EdgeGraph;
        template<GraphType Type>
        friend class Graph;
        friend class Dag;
        
        GraphBase() : m_nodes(), m_indices(), m_string_nodes(), m_collapsed_indices(), m_free_indices() {}
        GraphBase(const std::vector<std::string>& nodes) : m_nodes(),
                                                           m_indices(),
                                                           m_string_nodes(nodes),
                                                           m_collapsed_indices(),
                                                           m_free_indices() {
            m_nodes.reserve(nodes.size());

            for (size_t i = 0; i < nodes.size(); ++i) {
                NodeType n(i, nodes[i]);
                m_nodes.push_back(n);
                m_indices.insert(std::make_pair(nodes[i], i));
                m_collapsed_indices.insert(std::make_pair(nodes[i], i));
            }

            if (m_indices.size() != m_nodes.size()) {
                throw std::invalid_argument("Graph cannot be created with repeated names.");
            }
        }

        int num_nodes() const {
            return m_nodes.size() - m_free_indices.size();
        }

        template<typename V>
        const NodeType& node(const V& idx) const {
            return m_nodes[check_index(idx)]; 
        }

        const std::vector<std::string>& nodes() const;
        const std::vector<NodeType>& node_indices() const { return m_nodes; }


        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
        }

        bool contains_node(const std::string& name) const {
            return m_indices.count(name) > 0;
        }
        
        size_t add_node(const std::string& node);

        template<typename V>
        void remove_node(const V& idx) {
            remove_node_unsafe(check_index(idx));
        }

        void remove_node_unsafe(int index);

        const std::string& name(int idx) const {
            return m_nodes[check_index(idx)].name();
        }

        int index(const std::string& node) const {
            return check_index(node);
        }

        int collapsed_index(const std::string& node) const {
            auto f = m_collapsed_indices.find(node);
            if (f == m_collapsed_indices.end()) {
                throw std::invalid_argument("Node " + node + " not present in the graph.");
            }

            return f->second;
        }

        int index_from_collapsed(int collapsed_index) const {
            if (collapsed_index < 0 || collapsed_index >= num_nodes()) {
                throw std::invalid_argument("Wrong collapsed index (" + std::to_string(collapsed_index) + 
                                            ") for a graph with " + std::to_string(num_nodes()) + " nodes");
            }

            return index(m_string_nodes[collapsed_index]);
        }

        int collapsed_from_index(int index) const {
            return collapsed_index(m_nodes[check_index(index)].name());
        }

        const std::unordered_map<std::string, int>& collapsed_indices() const {
            return m_collapsed_indices;
        }

        const std::string& collapsed_name(int collapsed_index) const {
            if (collapsed_index < 0 || collapsed_index >= num_nodes()) {
                throw std::invalid_argument("Wrong collapsed index (" + std::to_string(collapsed_index) + 
                                            ") for a graph with " + std::to_string(num_nodes()) + " nodes");
            }

            return m_string_nodes[collapsed_index];
        }

        const std::vector<size_t>& free_indices() const {
            return m_free_indices;
        }

        bool is_valid(int idx) const {
            return idx >= 0 && static_cast<size_t>(idx) < m_nodes.size() && m_nodes[idx].is_valid();
        }
    private:
        int check_index(int idx) const {
            if (!is_valid(idx)) {
                throw std::invalid_argument("Node index " + std::to_string(idx) + " invalid.");
            }

            return idx;
        }

        int check_index(const std::string& name) const {
            auto f = m_indices.find(name);
            if (f == m_indices.end()) {
                throw std::invalid_argument("Node " + name + " not present in the graph.");
            }

            return f->second;
        }

        std::vector<NodeType> m_nodes;
        std::unordered_map<std::string, int> m_indices;
        std::vector<std::string> m_string_nodes;
        std::unordered_map<std::string, int> m_collapsed_indices;
        std::vector<size_t> m_free_indices;
    };

    template<typename Derived>
    const std::vector<std::string>& GraphBase<Derived>::nodes() const {
        return m_string_nodes;
    }

    template<typename Derived>
    size_t GraphBase<Derived>::add_node(const std::string& node) {
        if (m_indices.find(node) != m_indices.end()) {
            throw std::invalid_argument("Cannot add node " + node + " because a node with the same name already exists.");
        }

        size_t idx = [this, &node]() {
            if (!m_free_indices.empty()) {
                size_t idx = m_free_indices.back();
                m_free_indices.pop_back();
                NodeType n(idx, node);
                m_nodes[idx] = n;
                return idx;
            }
            else {
                size_t idx = m_nodes.size();
                NodeType n(idx, node);
                m_nodes.push_back(n);
                return idx;
            }
        }();

        m_indices.insert(std::make_pair(node, idx));
        m_string_nodes.push_back(node);
        m_collapsed_indices.insert(std::make_pair(node, m_string_nodes.size()-1));

        if constexpr (GraphTraits<Derived>::has_arcs) {
            auto& arcg = static_cast<ArcGraph<Derived>&>(static_cast<Derived&>(*this));
            arcg.add_root(idx);
            arcg.add_leaf(idx);
        }

        return idx;
    }

    template<typename Derived>
    void GraphBase<Derived>::remove_node_unsafe(int index) {
        if constexpr (GraphTraits<Derived>::has_edges) {
            auto& derived = static_cast<Derived&>(*this);
            for (auto neighbor : derived.neighbor_indices(index)) {
                derived.remove_edge_unsafe(index, neighbor);
            }
        }

        if constexpr (GraphTraits<Derived>::has_arcs) {
            if (m_nodes[index].is_root()) {
                auto& arcg = static_cast<ArcGraph<Derived>&>(static_cast<Derived&>(*this));
                arcg.remove_root(index);
            }

            if (m_nodes[index].is_leaf()) {
                auto& arcg = static_cast<ArcGraph<Derived>&>(static_cast<Derived&>(*this));
                arcg.remove_leaf(index);
            }

            auto& derived = static_cast<Derived&>(*this);
            for (auto p : derived.parent_indices(index)) {
                derived.remove_arc_unsafe(p, index);
            }

            for (auto ch : derived.children_indices(index)) {
                derived.remove_arc_unsafe(index, ch);
            }
        }

        m_indices.erase(m_nodes[index].name());

        auto collapsed = m_collapsed_indices.extract(m_nodes[index].name());
        auto collapsed_index = collapsed.mapped();
        util::swap_remove(m_string_nodes, collapsed_index);
        m_collapsed_indices[m_string_nodes[collapsed_index]] = collapsed_index;

        m_nodes[index].invalidate();
        m_free_indices.push_back(index);
    }

    template<typename Derived>
    class ArcGraph {
    public:
        using Base = GraphBase<Derived>;
        using NodeType = typename GraphTraits<Derived>::NodeType;
        inline Base& base() { return static_cast<Base&>(static_cast<Derived&>(*this)); }
        inline const Base& base() const { return static_cast<const Base&>(static_cast<const Derived&>(*this)); }
        inline Derived& derived() { return static_cast<Derived&>(*this); }
        inline const Derived& derived() const { return static_cast<const Derived&>(*this); }

        ArcGraph() = default;
        ArcGraph(const std::vector<std::string>& nodes) : m_arcs(),
                                                          m_roots(),
                                                          m_leaves() {
            for (size_t i = 0; i < nodes.size(); ++i) {
                m_roots.insert(i);
                m_leaves.insert(i);
            }
        }

        friend class Graph<Directed>;
        friend class Graph<PartiallyDirected>;
        friend class Dag;

        int num_arcs() const {
            return m_arcs.size();
        }

        template<typename V>
        int num_parents(const V& idx) const {
            return num_parents_unsafe(base().check_index(idx));
        }

        int num_parents_unsafe(int idx) const {
            return base().m_nodes[idx].parents().size();
        }

        template<typename V>
        int num_children(const V& idx) const {
            return num_children_unsafe(base().check_index(idx));
        }

        int num_children_unsafe(int idx) const {
            return base().m_nodes[idx].children().size();
        }

        ArcStringVector arcs() const;
        const ArcSet& arc_indices() const { return m_arcs; }

        template<typename V>
        std::vector<std::string> parents(const V& idx) const {
            return parents(base().m_nodes[base().check_index(idx)]);
        }

        template<typename V>
        std::vector<int> parent_indices(const V& idx) const {
            auto& p = base().m_nodes[base().check_index(idx)].parents();
            return { p.begin(), p.end() };
        }

        template<typename V>
        const std::unordered_set<int>& parent_set(const V& idx) const {
            return base().m_nodes[base().check_index(idx)].parents();
        }

        template<typename V>
        std::string parents_to_string(const V& idx) const {
            return parents_to_string(base().m_nodes[base().check_index(idx)]);
        }

        template<typename V>
        std::vector<std::string> children(const V& idx) const {
            return children(base().m_nodes[base().check_index(idx)]);
        }

        template<typename V>
        std::vector<int> children_indices(const V& idx) const {
            auto& p = base().m_nodes[base().check_index(idx)].children();
            return { p.begin(), p.end() };
        }

        template<typename V>
        const std::unordered_set<int>& children_set(const V& idx) const {
            return base().m_nodes[base().check_index(idx)].children();
        }

        template<typename V>
        void add_arc(const V& source, const V& target) {
            auto s = base().check_index(source);
            auto t = base().check_index(target);
            if (!has_arc_unsafe(s, t))
                derived().add_arc_unsafe(s, t);
        }

        void add_arc_unsafe(int source, int target);

        template<typename V>
        bool has_arc(const V& source, const V& target) const {
            auto s = base().check_index(source);
            auto t = base().check_index(target);
            return has_arc_unsafe(s, t);
        }

        bool has_arc_unsafe(int source, int target) const {
            const auto& p = base().m_nodes[target].parents();
            return p.find(source) != p.end();
        }

        template<typename V>
        void remove_arc(const V& source, const V& target) {
            auto s = base().check_index(source);
            auto t = base().check_index(target);
            if (has_arc_unsafe(s, t))
                derived().remove_arc_unsafe(s, t);
        }

        void remove_arc_unsafe(int source, int target);

        template<typename V>
        void flip_arc(const V& source, const V& target) {
            auto s = base().check_index(source);
            auto t = base().check_index(target);
            if (has_arc_unsafe(s, t))
                derived().flip_arc_unsafe(s, t);
        }

        void flip_arc_unsafe(int source, int target);

        const std::unordered_set<int>& roots() const {
            return m_roots;
        }

        const std::unordered_set<int>& leaves() const {
            return m_leaves;
        }
    private:
        friend size_t GraphBase<Derived>::add_node(const std::string& node);
        friend void GraphBase<Derived>::remove_node_unsafe(int index);

        void add_root(int idx) {
            m_roots.insert(idx);
        }

        void remove_root(int idx) {
            m_roots.erase(idx);
        }

        void add_leaf(int idx) {
            m_leaves.insert(idx);
        }

        void remove_leaf(int idx) {
            m_leaves.erase(idx);
        }

        std::vector<std::string> parents(const NodeType& n) const;
        std::vector<std::string> children(const NodeType& n) const;
        std::string parents_to_string(const NodeType& n) const;

        ArcSet m_arcs;
        std::unordered_set<int> m_roots;
        std::unordered_set<int> m_leaves;
    };

    template<typename Derived>
    ArcStringVector ArcGraph<Derived>::arcs() const {
        ArcStringVector res;
        res.reserve(m_arcs.size());
        
        for (auto& arc : m_arcs) {
            res.push_back({base().m_nodes[arc.first].name(), 
                           base().m_nodes[arc.second].name()});
        }
        return res;
    }

    template<typename Derived>
    void ArcGraph<Derived>::add_arc_unsafe(int source, int target) {
        if (base().m_nodes[target].is_root()) {
            m_roots.erase(target);
        }

        if (base().m_nodes[source].is_leaf()) {
            m_leaves.erase(source);
        }

        m_arcs.insert({source, target});
        base().m_nodes[target].add_parent(source);
        base().m_nodes[source].add_children(target);
    }

    template<typename Derived>
    void ArcGraph<Derived>::remove_arc_unsafe(int source, int target) {
        m_arcs.erase({source, target});
        base().m_nodes[target].remove_parent(source);
        base().m_nodes[source].remove_children(target);

        if (base().m_nodes[target].is_root()) {
            m_roots.insert(target);
        }

        if (base().m_nodes[source].is_leaf()) {
            m_leaves.insert(source);
        }
    }

    template<typename Derived>
    void ArcGraph<Derived>::flip_arc_unsafe(int source, int target) {
        m_arcs.erase({source, target});
        m_arcs.insert({target, source});

        base().m_nodes[target].remove_parent(source);
        base().m_nodes[source].remove_children(target);

        base().m_nodes[target].add_children(source);
        base().m_nodes[source].add_parent(target);
    }

    template<typename Derived>
    std::vector<std::string> ArcGraph<Derived>::parents(const NodeType& n) const {
        std::vector<std::string> res;

        const auto& parent_indices = n.parents();
        res.reserve(parent_indices.size());

        for (auto node : parent_indices) {
            res.push_back(base().m_nodes[node].name());
        }

        return res;
    }

    template<typename Derived>
    std::vector<std::string> ArcGraph<Derived>::children(const NodeType& n) const {
        std::vector<std::string> res;

        const auto& parent_indices = n.children();
        res.reserve(parent_indices.size());

        for (auto node : parent_indices) {
            res.push_back(base().m_nodes[node].name());
        }

        return res;
    }

    template<typename Derived>
    std::string ArcGraph<Derived>::parents_to_string(const NodeType& n) const {
        const auto& pa = n.parents();
        if (!pa.empty()) {
            std::string str = "[" + base().m_nodes[*pa.begin()].name();
            for (auto it = ++pa.begin(); it != pa.end(); ++it) {
                str += ", " + base().m_nodes[*it].name();
            }
            str += "]";
            return str;
        } else {
            return "[]";
        }
    }

    template<typename Derived>
    class EdgeGraph {
    public:
        using Base = GraphBase<Derived>;
        using NodeType = typename GraphTraits<Derived>::NodeType;

        friend class Graph<Undirected>;
        friend class Graph<PartiallyDirected>;

        inline Base& base() { 
            return static_cast<Base&>(static_cast<Derived&>(*this)); 
        }
        inline const Base& base() const { 
            return static_cast<const Base&>(static_cast<const Derived&>(*this)); 
        }

        int num_edges() const {
            return m_edges.size();
        }

        template<typename V>
        int num_neighbors(const V& idx) const {
            return num_neighbors_unsafe(base().check_index(idx));
        }

        int num_neighbors_unsafe(int idx) const {
            return base().m_nodes[idx].neighbors().size();
        }

        EdgeStringVector edges() const;
        const EdgeSet& edge_indices() const { return m_edges; }

        template<typename V>
        std::vector<std::string> neighbors(const V& idx) const {
            return neighbors(base().m_nodes[base().check_index(idx)]);
        }

        template<typename V>
        std::vector<int> neighbor_indices(const V& idx) const {
            const auto& n = base().m_nodes[base().check_index(idx)].neighbors();
            return { n.begin(), n.end() };
        }

        template<typename V>
        const std::unordered_set<int>& neighbor_set(const V& idx) const {
            return base().m_nodes[base().check_index(idx)].neighbors();
        }

        template<typename V>
        void add_edge(const V& source, const V& target) {
            auto s = base().check_index(source);
            auto t = base().check_index(target);
            if (!has_edge_unsafe(s, t))
                add_edge_unsafe(s, t);
        }

        void add_edge_unsafe(int source, int target);

        template<typename V>
        bool has_edge(const V& source, const V& target) const {
            auto s = base().check_index(source);
            auto t = base().check_index(target);
            return has_edge_unsafe(s, t);
        }

        bool has_edge_unsafe(int source, int target) const {
            const auto& p = base().m_nodes[target].neighbors();
            return p.find(source) != p.end();
        }

        template<typename V>
        void remove_edge(const V& source, const V& target) {
            auto s = base().check_index(source);
            auto t = base().check_index(target);
            if (has_edge_unsafe(s, t))
                remove_edge_unsafe(s, t);
        }

        void remove_edge_unsafe(int source, int target);
    
    private:
        std::vector<std::string> neighbors(const NodeType& n) const;

        EdgeSet m_edges;
    };

    template<typename Derived>
    EdgeStringVector EdgeGraph<Derived>::edges() const {
        EdgeStringVector res;
        res.reserve(m_edges.size());

        for (auto& edge : m_edges) {
            res.push_back({base().m_nodes[edge.first].name(), 
                           base().m_nodes[edge.second].name()});
        }

        return res;
    }

    template<typename Derived>
    void EdgeGraph<Derived>::add_edge_unsafe(int source, int target) {
        m_edges.insert({source, target});
        base().m_nodes[source].add_neighbor(target);
        base().m_nodes[target].add_neighbor(source);
    }

    template<typename Derived>
    void EdgeGraph<Derived>::remove_edge_unsafe(int source, int target) {
        m_edges.erase({source, target});
        base().m_nodes[source].remove_neighbor(target);
        base().m_nodes[target].remove_neighbor(source);
    }

    template<typename Derived>
    std::vector<std::string> EdgeGraph<Derived>::neighbors(const NodeType& n) const {
        std::vector<std::string> res;

        const auto& neighbors_indices = n.neighbors();
        res.reserve(neighbors_indices.size());

        for (auto node : neighbors_indices) {
            res.push_back(base().m_nodes[node].name());
        }

        return res;
    }

    template<typename G>
    py::tuple __getstate__(const G& g) {
        std::vector<std::string> nodes;
        nodes.reserve(g.num_nodes());
        std::vector<Arc> arcs;
        if constexpr (GraphTraits<G>::has_arcs) arcs.reserve(g.num_arcs());
        std::vector<Edge> edges;
        if constexpr (GraphTraits<G>::has_edges) edges.reserve(g.num_edges());


        if (g.free_indices().empty()) {
            for (auto& n : g.node_indices()) {
                nodes.push_back(n.name());
            }

            if constexpr (GraphTraits<G>::has_arcs) {
                for (auto& arc : g.arc_indices()) {
                    arcs.push_back(arc);
                }
            }

            if constexpr (GraphTraits<G>::has_edges) {
                for (auto& edge : g.edge_indices()) {
                    edges.push_back(edge);
                }
            }
        } else {
            std::unordered_map<int, int> new_indices;

            for (int i = 0, j = 0; i < g.num_nodes(); ++i) {
                if (g.node(i).is_valid()) {
                    nodes.push_back(g.name(i));
                    new_indices.insert({i, j++});
                }
            }

            if constexpr (GraphTraits<G>::has_arcs) {
                for (auto& arc : g.arc_indices()) {
                    arcs.push_back({new_indices[arc.first], new_indices[arc.second]});
                }
            }

            if constexpr (GraphTraits<G>::has_edges) {
                for (auto& edge : g.edge_indices()) {
                    edges.push_back({new_indices[edge.first], new_indices[edge.second]});
                }
            }
        }

        if constexpr (GraphTraits<G>::has_arcs) {
            if constexpr (GraphTraits<G>::has_edges) {
                return py::make_tuple(nodes, arcs, edges);
            } else {
                return py::make_tuple(nodes, arcs);
            }
        } else {
            return py::make_tuple(nodes, edges);
        }
    }

    template<typename G>
    py::tuple __getstate__(const G&& g) {
        return graph::__getstate__(g);
    }

    template<typename G>
    G __setstate__(py::tuple& t) {
        if (t.size() != (1 + GraphTraits<G>::has_arcs + GraphTraits<G>::has_edges))
            throw std::runtime_error("Not valid Graph.");

        G g(t[0].cast<std::vector<std::string>>());

        if constexpr (GraphTraits<G>::has_arcs) {
            auto arcs = t[1].cast<std::vector<Arc>>();

            for (auto& arc : arcs) {
                g.add_arc(arc.first, arc.second);
            }

            if constexpr (GraphTraits<G>::has_edges) {
                auto edges = t[2].cast<std::vector<Edge>>();

                for (auto& edge : edges) {
                    g.add_edge(edge.first, edge.second);
                }
            }
        } else {
            auto edges = t[1].cast<std::vector<Edge>>();

            for (auto& edge : edges) {
                g.add_edge(edge.first, edge.second);
            }
        }
        
        return g;
    }

    template<typename G>
    G __setstate__(py::tuple&& t) {
        return graph::__setstate__<G>(t);
    }

    template<typename G>
    void save_graph(G& graph, const std::string& name) {
        auto open = py::module::import("io").attr("open");
        auto file = open(name, "wb");
        py::module::import("pickle").attr("dump")(py::cast(&graph), file, 2);
        file.attr("close")();
    }

    py::object load_graph(const std::string& name);

    template<>
    class Graph<PartiallyDirected> : public GraphBase<PartiallyDirectedGraph>,
                                     public ArcGraph<PartiallyDirectedGraph>,
                                     public EdgeGraph<PartiallyDirectedGraph> 
    {
    public:
        Graph() : GraphBase<PartiallyDirectedGraph>(),
                  ArcGraph<PartiallyDirectedGraph>(),
                  EdgeGraph<PartiallyDirectedGraph>() {}
        Graph(const std::vector<std::string>& nodes) : GraphBase<PartiallyDirectedGraph>(nodes),
                                                       ArcGraph<PartiallyDirectedGraph>(nodes),
                                                       EdgeGraph<PartiallyDirectedGraph>() {}
        Graph(const ArcStringVector& arcs, const EdgeStringVector& edges) : GraphBase<PartiallyDirectedGraph>(),
                                                                ArcGraph<PartiallyDirectedGraph>(),
                                                                EdgeGraph<PartiallyDirectedGraph>() {

            for (auto& arc : arcs) {
                if (m_indices.count(arc.first) == 0) {
                    add_node(arc.first);
                }

                if (m_indices.count(arc.second) == 0) {
                    add_node(arc.second);
                }

                add_arc(arc.first, arc.second);
            }

            for (auto& edge : edges) {
                if (m_indices.count(edge.first) == 0) {
                    add_node(edge.first);
                }

                if (m_indices.count(edge.second) == 0) {
                    add_node(edge.second);
                }

                add_edge(edge.first, edge.second);
            }

        }

        Graph(const std::vector<std::string>& nodes,
              const ArcStringVector& arcs,
              const EdgeStringVector& edges) : GraphBase<PartiallyDirectedGraph>(nodes),
                                         ArcGraph<PartiallyDirectedGraph>(nodes),
                                         EdgeGraph<PartiallyDirectedGraph>() {
                              
            for (auto& arc : arcs) {
                if (m_indices.count(arc.first) == 0) throw pybind11::index_error(
                    "Node \"" + arc.first + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");
            
                if (m_indices.count(arc.second) == 0) throw pybind11::index_error(
                    "Node \"" + arc.second + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");

                add_arc(arc.first, arc.second);
            }

            for (auto& edge : edges) {
                if (m_indices.count(edge.first) == 0) throw pybind11::index_error(
                    "Node \"" + edge.first + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");
            
                if (m_indices.count(edge.second) == 0) throw pybind11::index_error(
                    "Node \"" + edge.second + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");

                add_edge(edge.first, edge.second);
            }
        }

        Graph(Graph<Undirected>&& g);
        Graph(Graph<Directed>&& g);

        static PartiallyDirectedGraph CompleteUndirected(const std::vector<std::string>& nodes);

        template<typename V>
        void direct(const V& source, const V& target) {
            auto s = this->check_index(source);
            auto t = this->check_index(target);
            direct_unsafe(s, t);
        }

        void direct_unsafe(int source, int target);

        template<typename V>
        void undirect(const V& source, const V& target) {
            auto s = this->check_index(source);
            auto t = this->check_index(target);
            undirect_unsafe(s, t);
        }

        void undirect_unsafe(int source, int target);

        template<typename V>
        bool has_connection(const V& source, const V& target) const {
            auto s = this->check_index(source);
            auto t = this->check_index(target);
            return has_connection_unsafe(s, t);
        }

        bool has_connection_unsafe(int source, int target) const {
            return has_edge_unsafe(source, target) || has_arc_unsafe(source, target) || has_arc_unsafe(target, source);
        }

        Dag to_dag() const;

        py::tuple __getstate__() const {
            return graph::__getstate__(*this);
        }

        static PartiallyDirectedGraph __setstate__(py::tuple& t) {
            return graph::__setstate__<PartiallyDirectedGraph>(t);
        }

        static PartiallyDirectedGraph __setstate__(py::tuple&& t) {
            return graph::__setstate__<PartiallyDirectedGraph>(t);
        }

        void save(const std::string& name) const {
            save_graph(*this, name);
        }
    };

    template<>
    class Graph<Undirected> : public GraphBase<UndirectedGraph>,
                              public EdgeGraph<UndirectedGraph> {
    public:

        Graph() : GraphBase<UndirectedGraph>(), EdgeGraph<UndirectedGraph>() {}
        Graph(const std::vector<std::string>& nodes) : GraphBase<UndirectedGraph>(nodes),
                                                       EdgeGraph<UndirectedGraph>() {}
        Graph(const EdgeStringVector& edges) : GraphBase<UndirectedGraph>(), 
                                         EdgeGraph<UndirectedGraph>() {
            for (auto& edge : edges) {
                if (m_indices.count(edge.first) == 0) {
                    add_node(edge.first);
                }

                if (m_indices.count(edge.second) == 0) {
                    add_node(edge.second);
                }

                add_edge(edge.first, edge.second);
            }
        }

        Graph(const std::vector<std::string>& nodes, const EdgeStringVector& edges) 
                    : GraphBase<UndirectedGraph>(nodes), EdgeGraph<UndirectedGraph>() {

            for (auto& edge : edges) {
                if (m_indices.count(edge.first) == 0) throw pybind11::index_error(
                    "Node \"" + edge.first + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");
            
                if (m_indices.count(edge.second) == 0) throw pybind11::index_error(
                    "Node \"" + edge.second + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");

                add_edge(edge.first, edge.second);
            }   
        }

        static UndirectedGraph Complete(const std::vector<std::string>& nodes);

        template<typename V>
        bool has_path(const V& source, const V& target) const {
            auto s = this->check_index(source);
            auto t = this->check_index(target);
            return has_path_unsafe(s, t);
        }

        bool has_path_unsafe(int source, int target) const;

        py::tuple __getstate__() const {
            return graph::__getstate__(*this);
        }

        static UndirectedGraph __setstate__(py::tuple& t) {
            return graph::__setstate__<UndirectedGraph>(t);
        }

        static UndirectedGraph __setstate__(py::tuple&& t) {
            return graph::__setstate__<UndirectedGraph>(t);
        }

        void save(const std::string& name) const {
            save_graph(*this, name);
        }
    };

    template<>
    class Graph<Directed> : public GraphBase<DirectedGraph>,
                            public ArcGraph<DirectedGraph> {
    public:
        Graph() : GraphBase<DirectedGraph>(), ArcGraph<DirectedGraph>() {}
        Graph(const std::vector<std::string>& nodes) : GraphBase<DirectedGraph>(nodes), 
                                                       ArcGraph<DirectedGraph>(nodes) {}

        Graph(const ArcStringVector& arcs) : GraphBase<DirectedGraph>(),
                                       ArcGraph<DirectedGraph>() {
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

        Graph(const std::vector<std::string>& nodes, 
              const ArcStringVector& arcs) : GraphBase<DirectedGraph>(nodes),
                                       ArcGraph<DirectedGraph>(nodes) {
            for (auto& arc : arcs) {
                if (m_indices.count(arc.first) == 0) throw pybind11::index_error(
                    "Node \"" + arc.first + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");
            
                if (m_indices.count(arc.second) == 0) throw pybind11::index_error(
                    "Node \"" + arc.second + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");

                add_arc(arc.first, arc.second);
            }
        }

        template<typename V>
        bool has_path(const V& source, const V& target) const {
            auto s = this->check_index(source);
            auto t = this->check_index(target);
            return has_path_unsafe(s, t);
        }

        bool has_path_unsafe(int source, int target) const;
        bool has_path_unsafe_no_direct_arc(int source, int target) const;

        py::tuple __getstate__() const {
            return graph::__getstate__(*this);
        }

        static DirectedGraph __setstate__(py::tuple& t) {
            return graph::__setstate__<DirectedGraph>(t);
        }

        static DirectedGraph __setstate__(py::tuple&& t) {
            return graph::__setstate__<DirectedGraph>(t);
        }

        void save(const std::string& name) const {
            save_graph(*this, name);
        }
    };

    class Dag : public DirectedGraph {
    public:
        Dag() : DirectedGraph() {}
        Dag(const std::vector<std::string>& nodes) : DirectedGraph(nodes) {}
        Dag(const ArcStringVector& arcs) : DirectedGraph(arcs) {
            topological_sort();
        }
        Dag(const std::vector<std::string>& nodes, const ArcStringVector& arcs) : DirectedGraph(nodes, arcs) {
            topological_sort();
        }

        std::vector<std::string> topological_sort() const;

        template<typename V>
        bool can_add_arc(const V& source, const V& target) const {
            auto s = this->check_index(source);
            auto t = this->check_index(target);
            return can_add_arc_unsafe(s, t);
        }

        bool can_add_arc_unsafe(int source, int target) const;

        template<typename V>
        bool can_flip_arc(const V& source, const V& target) const {
            auto s = this->check_index(source);
            auto t = this->check_index(target);
            return can_flip_arc_unsafe(s, t);
        }

        bool can_flip_arc_unsafe(int source, int target) const;

        template<typename V>
        void add_arc(const V& source, const V& target) {
            auto s = check_index(source);
            auto t = check_index(target);

            if (!can_add_arc_unsafe(s, t)) {
                throw std::runtime_error("Arc " + name(s) + " -> " + name(t) + " addition would break acyclity.");
            }

            ArcGraph<DirectedGraph>::add_arc_unsafe(s, t);
        }

        template<typename V>
        void flip_arc(const V& source, const V& target) {
            auto s = check_index(source);
            auto t = check_index(target);

            if (!can_flip_arc_unsafe(s, t)) {
                throw std::runtime_error("Arc " + name(s) + " -> " + name(t) + " flip would break acyclity.");
            }

            ArcGraph<DirectedGraph>::flip_arc_unsafe(s, t);
        }

        PartiallyDirectedGraph to_pdag() const;

        bool is_dag() const { 
            try {
                topological_sort();
                return true;
            } catch (std::invalid_argument&) {
                return false;
            }
        }

        py::tuple __getstate__() const {
            return graph::__getstate__(*this);
        }

        static Dag __setstate__(py::tuple& t) {
            return graph::__setstate__<Dag>(t);
        }

        static Dag __setstate__(py::tuple&& t) {
            return graph::__setstate__<Dag>(t);
        }

        void save(const std::string& name) const {
            save_graph(*this, name);
        }
    };

}

#endif //PYBNESIAN_GRAPH_GENERIC_GRAPH_HPP
