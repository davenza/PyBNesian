#ifndef PGM_DATASET_GENERIC_GRAPH_HPP
#define PGM_DATASET_GENERIC_GRAPH_HPP

#include <pybind11/pybind11.h>
#include <graph/graph_types.hpp>
#include <util/util_types.hpp>

#include <iostream>

using graph::DNode, graph::UNode, graph::PDNode;
using util::ArcVector, util::EdgeVector;

namespace graph {


    template<typename Type>
    class Graph;

    struct Directed;
    struct Undirected;
    struct PartiallyDirected;

    using DirectedGraph = Graph<Directed>;
    using UndirectedGraph = Graph<Undirected>;
    using PartiallyDirectedGraph = Graph<PartiallyDirected>;

    template<typename GraphType>
    struct GraphTraits;

    template<>
    struct GraphTraits<DirectedGraph> {
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
        template<typename GraphType>
        friend class Graph;
        friend class Dag;
        
        GraphBase() : m_nodes(), m_indices(), free_indices() {}
        GraphBase(const std::vector<std::string>& nodes) : m_nodes(), m_indices(), free_indices() {
            m_nodes.reserve(nodes.size());

            for (size_t i = 0; i < nodes.size(); ++i) {
                NodeType n (i, nodes[i]);
                m_nodes.push_back(n);
                m_indices.insert(std::make_pair(nodes[i], i));
            }
        }

        int num_nodes() const {
            return m_nodes.size() - free_indices.size();
        }

        template<typename V>
        const NodeType& node(const V& idx) const {
            return m_nodes[check_index(idx)]; 
        }

        std::vector<std::string> nodes() const;
        const std::vector<NodeType>& node_indices() const { return m_nodes; }

        bool contains_node(const std::string& name) const {
            return m_indices.count(name) > 0;
        }
        
        int add_node(const std::string& node);

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

        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
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
        std::vector<int> free_indices;
    };

    template<typename Derived>
    std::vector<std::string> GraphBase<Derived>::nodes() const {
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

    template<typename Derived>
    int GraphBase<Derived>::add_node(const std::string& node) {
        int idx = [this, &node]() {
            if (!free_indices.empty()) {
                int idx = free_indices.back();
                free_indices.pop_back();
                NodeType n(idx, node);
                m_nodes[idx] = n;
                return idx;
            }
            else {
                int idx = m_nodes.size();
                NodeType n(idx, node);
                m_nodes.push_back(n);
                return idx;
            }
        }();

        m_indices.insert(std::make_pair(node, idx));

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
        m_nodes[index].invalidate();
        free_indices.push_back(index);
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

        ArcVector arcs() const;
        const auto& arc_indices() const { return m_arcs; }

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
        friend int GraphBase<Derived>::add_node(const std::string& node);
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

        std::unordered_set<Arc, ArcHash> m_arcs;
        std::unordered_set<int> m_roots;
        std::unordered_set<int> m_leaves;
    };

    template<typename GraphType>
    ArcVector ArcGraph<GraphType>::arcs() const {
        ArcVector res;
        res.reserve(m_arcs.size());
        
        for (auto& arc : m_arcs) {
            res.push_back({base().m_nodes[arc.first].name(), 
                           base().m_nodes[arc.second].name()});
        }
        return res;
    }

    template<typename GraphType>
    void ArcGraph<GraphType>::add_arc_unsafe(int source, int target) {
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

    template<typename GraphType>
    void ArcGraph<GraphType>::remove_arc_unsafe(int source, int target) {
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

    template<typename GraphType>
    void ArcGraph<GraphType>::flip_arc_unsafe(int source, int target) {
        m_arcs.erase({source, target});
        m_arcs.insert({target, source});

        base().m_nodes[target].remove_parent(source);
        base().m_nodes[source].remove_children(target);

        base().m_nodes[target].add_children(source);
        base().m_nodes[source].add_parent(target);
    }

    template<typename GraphType>
    std::vector<std::string> ArcGraph<GraphType>::parents(const NodeType& n) const {
        std::vector<std::string> res;

        const auto& parent_indices = n.parents();
        res.reserve(parent_indices.size());

        for (auto node : parent_indices) {
            res.push_back(base().m_nodes[node].name());
        }

        return res;
    }

    template<typename GraphType>
    std::vector<std::string> ArcGraph<GraphType>::children(const NodeType& n) const {
        std::vector<std::string> res;

        const auto& parent_indices = n.children();
        res.reserve(parent_indices.size());

        for (auto node : parent_indices) {
            res.push_back(base().m_nodes[node].name());
        }

        return res;
    }

    template<typename GraphType>
    std::string ArcGraph<GraphType>::parents_to_string(const NodeType& n) const {
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

    template<typename GraphType>
    class EdgeGraph {
    public:
        using Base = GraphBase<GraphType>;
        using NodeType = typename GraphTraits<GraphType>::NodeType;

        friend class Graph<Undirected>;
        friend class Graph<PartiallyDirected>;

        inline Base& base() { 
            return static_cast<Base&>(static_cast<GraphType&>(*this)); 
        }
        inline const Base& base() const { 
            return static_cast<const Base&>(static_cast<const GraphType&>(*this)); 
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

        EdgeVector edges() const;
        const auto& edge_indices() const { return m_edges; }

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

        std::unordered_set<Edge, EdgeHash, EdgeEqualTo> m_edges;
    };

    template<typename GraphType>
    EdgeVector EdgeGraph<GraphType>::edges() const {
        EdgeVector res;
        res.reserve(m_edges.size());

        for (auto& edge : m_edges) {
            res.push_back({base().m_nodes[edge.first].name(), 
                           base().m_nodes[edge.second].name()});
        }

        return res;
    }

    template<typename GraphType>
    void EdgeGraph<GraphType>::add_edge_unsafe(int source, int target) {
        m_edges.insert({source, target});
        base().m_nodes[source].add_neighbor(target);
        base().m_nodes[target].add_neighbor(source);
    }

    template<typename GraphType>
    void EdgeGraph<GraphType>::remove_edge_unsafe(int source, int target) {
        m_edges.erase({source, target});
        base().m_nodes[source].remove_neighbor(target);
        base().m_nodes[target].remove_neighbor(source);
    }

    template<typename GraphType>
    std::vector<std::string> EdgeGraph<GraphType>::neighbors(const NodeType& n) const {
        std::vector<std::string> res;

        const auto& neighbors_indices = n.neighbors();
        res.reserve(neighbors_indices.size());

        for (auto node : neighbors_indices) {
            res.push_back(base().m_nodes[node].name());
        }

        return res;
    }

    class Dag;

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
        Graph(const EdgeVector& edges, const ArcVector& arcs) : GraphBase<PartiallyDirectedGraph>(),
                                                                ArcGraph<PartiallyDirectedGraph>(),
                                                                EdgeGraph<PartiallyDirectedGraph>() {

            for (auto& edge : edges) {
                if (m_indices.count(edge.first) == 0) {
                    add_node(edge.first);
                }

                if (m_indices.count(edge.second) == 0) {
                    add_node(edge.second);
                }

                add_edge(edge.first, edge.second);
            }

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
              const EdgeVector& edges, 
              const ArcVector& arcs) : GraphBase<PartiallyDirectedGraph>(nodes),
                                       ArcGraph<PartiallyDirectedGraph>(nodes),
                                       EdgeGraph<PartiallyDirectedGraph>() {
                              
            for (auto& edge : edges) {
                if (m_indices.count(edge.first) == 0) throw pybind11::index_error(
                    "Node \"" + edge.first + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");
            
                if (m_indices.count(edge.second) == 0) throw pybind11::index_error(
                    "Node \"" + edge.second + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");

                add_edge(edge.first, edge.second);
            }

            for (auto& arc : arcs) {
                if (m_indices.count(arc.first) == 0) throw pybind11::index_error(
                    "Node \"" + arc.first + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");
            
                if (m_indices.count(arc.second) == 0) throw pybind11::index_error(
                    "Node \"" + arc.second + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");

                add_arc(arc.first, arc.second);
            }          
        }

        Graph(Graph<Undirected>&& g);
        Graph(Graph<Directed>&& g);

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
    };

    template<>
    class Graph<Undirected> : public GraphBase<UndirectedGraph>,
                              public EdgeGraph<UndirectedGraph> {
    public:

        Graph() : GraphBase<UndirectedGraph>(), EdgeGraph<UndirectedGraph>() {}
        Graph(const std::vector<std::string>& nodes) : GraphBase<UndirectedGraph>(nodes),
                                                       EdgeGraph<UndirectedGraph>() {}
        Graph(const EdgeVector& edges) : GraphBase<UndirectedGraph>(), 
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

        Graph(const std::vector<std::string>& nodes, const EdgeVector& edges) 
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
    };

    template<>
    class Graph<Directed> : public GraphBase<DirectedGraph>,
                            public ArcGraph<DirectedGraph> {
    public:
        Graph() : GraphBase<DirectedGraph>(), ArcGraph<DirectedGraph>() {}
        Graph(const std::vector<std::string>& nodes) : GraphBase<DirectedGraph>(nodes), 
                                                       ArcGraph<DirectedGraph>(nodes) {}

        Graph(const ArcVector& arcs) : GraphBase<DirectedGraph>(),
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
              const ArcVector& arcs) : GraphBase<DirectedGraph>(nodes),
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
    };

    class Dag : public DirectedGraph {
    public:
        Dag() : DirectedGraph() {}
        Dag(const std::vector<std::string>& nodes) : DirectedGraph(nodes) {}
        Dag(const ArcVector& arcs) : DirectedGraph(arcs) {
            topological_sort();
        }
        Dag(const std::vector<std::string>& nodes, const ArcVector& arcs) : DirectedGraph(nodes, arcs) {
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
        bool can_flip_arc(const V& source, const V& target) {
            auto s = this->check_index(source);
            auto t = this->check_index(target);
            return can_flip_arc_unsafe(s, t);
        }

        bool can_flip_arc_unsafe(int source, int target);

        PartiallyDirectedGraph to_pdag() const;
    };



}

#endif //PGM_DATASET_GENERIC_GRAPH_HPP
