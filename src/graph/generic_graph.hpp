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
    class GraphBase {
    public:
        using NodeType = typename GraphTraits<Derived>::NodeType;
        
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

        const NodeType& node(int idx) const {
            check_valid_indices(idx);
            return m_nodes[idx]; 
        }
        const NodeType& node(const std::string& name) const {
            auto idx = check_names(name);
            return m_nodes[idx]; 
        }

        std::vector<std::string> nodes() const;
        const std::vector<NodeType>& node_indices() const { return m_nodes; }

        bool contains_node(const std::string& name) const {
            return m_indices.count(name) > 0;
        }
        
        int add_node(const std::string& node);

        void remove_node(int idx) {
            check_valid_indices(idx);
            remove_node_unsafe(idx);
        }

        void remove_node(const std::string& node) {
            auto idx = check_names(node);
            remove_node_unsafe(idx);
        }

        void remove_node_unsafe(int index);

        const std::string& name(int idx) const {
            check_valid_indices(idx);
            return m_nodes[idx].name();
        }


        int index(const std::string& node) const {
            return check_names(node);
        }

        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
        }

        bool is_valid(int idx) const {
            return idx >= 0 && static_cast<size_t>(idx) < m_nodes.size() && m_nodes[idx].is_valid();
        }

    private:
        template<typename G>
        friend class ArcGraph;
        template<typename G>
        friend class EdgeGraph;
        template<typename GraphType>
        friend class Graph;
        friend class Dag;

        void check_valid_indices(int idx) const {
            if (!is_valid(idx)) {
                throw std::invalid_argument("Node index " + std::to_string(idx) + " invalid.");
            }
        }

        void check_valid_indices(int idx1, int idx2) const {
            if (!is_valid(idx1)) {
                throw std::invalid_argument("Node index " + std::to_string(idx1) + " invalid.");
            }

            if (!is_valid(idx2)) {
                throw std::invalid_argument("Node index " + std::to_string(idx2) + " invalid.");
            }
        }

        int check_names(const std::string& name) const {
            auto f = m_indices.find(name);
            if (f == m_indices.end()) {
                throw std::invalid_argument("Node " + name + " not present in the graph.");
            }

            return f->second;
        }

        std::pair<int, int> check_names(const std::string& v1, const std::string& v2) const {
            auto f = m_indices.find(v1);
            if (f == m_indices.end()) {
                throw std::invalid_argument("Node " + v1 + " not present in the graph.");
            }

            auto f2 = m_indices.find(v2);
            if (f2 == m_indices.end()) {
                throw std::invalid_argument("Node " + v2 + " not present in the graph.");
            }

            return std::make_pair(f->second, f2->second);
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
        return idx;
    }

    template<typename Derived>
    void GraphBase<Derived>::remove_node_unsafe(int index) {
        if constexpr (GraphTraits<Derived>::has_edges) {
            auto& derived = static_cast<Derived&>(*this);
            for (auto neighbor : m_nodes[index].neighbors()) {
                derived.remove_edge_unsafe(index, neighbor);
            }
        }

        if constexpr (GraphTraits<Derived>::has_arcs) {
            auto& derived = static_cast<Derived&>(*this);
            for (auto p : m_nodes[index].parents()) {
                derived.remove_arc_unsafe(p, index);
            }

            for (auto ch : m_nodes[index].children()) {
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

        friend class Graph<Directed>;
        friend class Graph<PartiallyDirected>;
        friend class Dag;

        int num_arcs() const {
            return m_arcs.size();
        }

        int num_parents(int idx) const {
            base().check_valid_indices(idx);
            return num_parents_unsafe(idx);
        }

        int num_parents(const std::string& node) const {
            auto idx = base().check_names(node);
            return num_parents_unsafe(idx);
        }

        int num_parents_unsafe(int idx) const {
            return base().m_nodes[idx].parents().size();
        }

        int num_children(int idx) const {
            base().check_valid_indices(idx);
            return num_children_unsafe(idx);
        }

        int num_children(const std::string& node) const {
            auto idx = base().check_names(node);
            return num_children_unsafe(idx);
        }

        int num_children_unsafe(int idx) const {
            return base().m_nodes[idx].children().size();
        }

        ArcVector arcs() const;
        const auto& arc_indices() const { return m_arcs; }

        std::vector<std::string> parents(int idx) const {
            base().check_valid_indices(idx);
            return parents(base().m_nodes[idx]);
        }

        std::vector<std::string> parents(const std::string& node) const {
            auto idx = base().check_names(node);
            return parents(base().m_nodes[idx]);
        }

        std::vector<int> parent_indices(int idx) const {
            base().check_valid_indices(idx);
            auto& p = base().m_nodes[idx].parents();
            return { p.begin(), p.end() };
        }

        std::vector<int> parent_indices(const std::string& node) const {
            auto idx = base().check_names(node);
            auto& p = base().m_nodes[idx].parents();
            return { p.begin(), p.end() };
        }

        std::string parents_to_string(int idx) const {
            base().check_valid_indices(idx);
            return parents_to_string(base().m_nodes[idx]);
        }

        std::string parents_to_string(const std::string& node) const {
            auto idx = base().check_names(node);
            return parents_to_string(base().m_nodes[idx]);
        }

        std::vector<std::string> children(int idx) const {
            base().check_valid_indices(idx);
            return children(base().m_nodes[idx]);
        }

        std::vector<std::string> children(const std::string& node) const {
            auto idx = base().check_names(node);
            return children(base().m_nodes[idx]);
        }

        std::vector<int> children_indices(int idx) const {
            base().check_valid_indices(idx);
            auto& p = base().m_nodes[idx].children();
            return { p.begin(), p.end() };
        }

        std::vector<int> children_indices(const std::string& node) const {
            auto idx = base().check_names(node);
            auto& p = base().m_nodes[idx].children();
            return { p.begin(), p.end() };
        }

        void add_arc(int source, int target) {
            if (!has_arc(source, target))
                derived().add_arc_unsafe(source, target);
        }

        void add_arc(const std::string& source, const std::string& target) {
            auto [s, t] = base().check_names(source, target);
            if (!has_arc_unsafe(s, t))
                derived().add_arc_unsafe(s, t);
        }

        void add_arc_unsafe(int source, int target);

        bool has_arc(int source, int target) const {
            base().check_valid_indices(source, target);
            return has_arc_unsafe(source, target);
        }

        bool has_arc(const std::string& source, const std::string& target) const {
            auto [s, t] = base().check_names(source, target);
            return has_arc_unsafe(s, t);
        }

        bool has_arc_unsafe(int source, int target) const {
            const auto& p = base().m_nodes[target].parents();
            return p.find(source) != p.end();
        }

        void remove_arc(int source, int target) {
            if (has_arc(source, target))
                derived().remove_arc_unsafe(source, target);
        }

        void remove_arc(const std::string& source, const std::string& target) {
            auto [s, t] = base().check_names(source, target);
            if (has_arc_unsafe(s, t))
                derived().remove_arc_unsafe(s, t);
        }

        void remove_arc_unsafe(int source, int target);

        void flip_arc(int source, int target) {
            if (has_arc(source, target))
                derived().flip_arc_unsafe(source, target);
        }

        void flip_arc(const std::string& source, const std::string& target) {
            auto [s, t] = base().check_names(source, target);
            if (has_arc_unsafe(s, t))
                derived().flip_arc_unsafe(s, t);
        }

        void flip_arc_unsafe(int source, int target);
    private:
        std::vector<std::string> parents(const NodeType& n) const;
        std::vector<std::string> children(const NodeType& n) const;
        std::string parents_to_string(const NodeType& n) const;

        std::unordered_set<Arc, ArcHash> m_arcs;
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
        m_arcs.insert({source, target});
        base().m_nodes[target].add_parent(source);
        base().m_nodes[source].add_children(target);
    }

    template<typename GraphType>
    void ArcGraph<GraphType>::remove_arc_unsafe(int source, int target) {
        m_arcs.erase({source, target});
        base().m_nodes[target].remove_parent(source);
        base().m_nodes[source].remove_children(target);
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

        int num_neighbors(int idx) const {
            base().check_valid_indices(idx);
            return num_neighbors_unsafe(idx);
        }

        int num_neighbors(const std::string& node) const {
            auto idx = base().check_names(node);
            return num_neighbors_unsafe(idx);
        }

        int num_neighbors_unsafe(int idx) const {
            return base().m_nodes[idx].neighbors().size();
        }

        EdgeVector edges() const;
        const auto& edge_indices() const { return m_edges; }

        std::vector<std::string> neighbors(int idx) const {
            base().check_valid_indices(idx);
            return neighbors(base().m_nodes[idx]);
        }

        std::vector<std::string> neighbors(const std::string& node) const {
            auto idx = base().check_names(node);
            return neighbors(idx);
        }

        const std::unordered_set<int>& neighbor_indices(int idx) const {
            base().check_valid_indices(idx);
            return base().m_nodes[idx].neighbors();
        }

        const std::unordered_set<int>& neighbor_indices(const std::string& node) const {
            auto idx = base().check_names(node);
            return base().m_nodes[idx].neighbors();
        }

        void add_edge(int source, int target) {
            if (!has_edge(source, target))
                add_edge_unsafe(source, target);
        }

        void add_edge(const std::string& source, const std::string& target) {
            auto [s, t] = base().check_names(source, target);
            if (!has_edge_unsafe(s, t))
                add_edge_unsafe(s, t);
        }

        void add_edge_unsafe(int source, int target);

        bool has_edge(int source, int target) const {
            base().check_valid_indices(source, target);
            return has_edge_unsafe(source, target);
        }

        bool has_edge(const std::string& source, const std::string& target) const {
            auto [s, t] = base().check_names(source, target);
            return has_edge_unsafe(s, t);
        }

        bool has_edge_unsafe(int source, int target) const {
            const auto& p = base().m_nodes[target].neighbors();
            return p.find(source) != p.end();
        }

        void remove_edge(int source, int target) {
            if (has_edge(source, target))
                remove_edge_unsafe(source, target);
        }

        void remove_edge(const std::string& source, const std::string& target) {
            auto [s, t] = base().check_names(source, target);
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
                                                       ArcGraph<PartiallyDirectedGraph>(),
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
                                       ArcGraph<PartiallyDirectedGraph>(),
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

        void direct(int source, int target) {
            check_valid_indices(source, target);
            direct_unsafe(source, target);
        }

        void direct(const std::string& source, const std::string& target) {
            auto [s, t] = check_names(source, target);
            direct_unsafe(s, t);
        }

        void direct_unsafe(int source, int target);

        void undirect(int source, int target) {
            check_valid_indices(source, target);
            undirect_unsafe(source, target);
        }

        void undirect(const std::string& source, const std::string& target) {
            auto [s, t] = check_names(source, target);
            undirect_unsafe(s, t);
        }

        void undirect_unsafe(int source, int target);

        bool has_connection(int source, int target) const {
            check_valid_indices(source, target);
            return has_connection_unsafe(source, target);
        }

        bool has_connection(const std::string& source, const std::string& target) const {
            auto [s, t] = check_names(source, target);
            return has_connection_unsafe(s, t);
        }

        bool has_connection_unsafe(int source, int target) const {
            return has_edge_unsafe(source, target) || has_arc_unsafe(source, target) || has_arc_unsafe(target, source);
        }

        DirectedGraph random_direct() const;
    private:
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

        bool has_path(int source, int target) const {
            check_valid_indices(source, target);
            return has_path_unsafe(source, target);
        }

        bool has_path(const std::string& source, const std::string& target) const {
            auto [s, t] = check_names(source, target);
            return has_path_unsafe(s, t);
        }

        bool has_path_unsafe(int source, int target) const;
    };

    template<>
    class Graph<Directed> : public GraphBase<DirectedGraph>,
                            public ArcGraph<DirectedGraph> {
    public:
        Graph() : GraphBase<DirectedGraph>(), 
                  ArcGraph<DirectedGraph>(), 
                  m_roots(), 
                  m_leaves() {}
        Graph(const std::vector<std::string>& nodes) : GraphBase<DirectedGraph>(nodes), 
                                                       ArcGraph<DirectedGraph>(), 
                                                       m_roots(),
                                                       m_leaves() {
            m_roots.reserve(nodes.size());
            m_leaves.reserve(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                m_roots.insert(i);
                m_leaves.insert(i);
            }
        }

        Graph(const ArcVector& arcs) : GraphBase<DirectedGraph>(),
                                       ArcGraph<DirectedGraph>(),
                                       m_roots(),
                                       m_leaves() {
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
                                       ArcGraph<DirectedGraph>() {
            
            m_roots.reserve(nodes.size());
            m_leaves.reserve(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                m_roots.insert(i);
                m_leaves.insert(i);
            }

            for (auto& arc : arcs) {
                if (m_indices.count(arc.first) == 0) throw pybind11::index_error(
                    "Node \"" + arc.first + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");
            
                if (m_indices.count(arc.second) == 0) throw pybind11::index_error(
                    "Node \"" + arc.second + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");

                add_arc(arc.first, arc.second);
            }
        }

        void add_node(const std::string& node) {
            auto idx = GraphBase<DirectedGraph>::add_node(node);
            m_roots.insert(idx);
            m_leaves.insert(idx);
        }

        void remove_node_unsafe(int index) {
            if (m_nodes[index].is_root()) {
                m_roots.erase(index);
            }

            if (m_nodes[index].is_leaf()) {
                m_leaves.erase(index);
            }

            GraphBase<DirectedGraph>::remove_node_unsafe(index);
        }

        void add_arc_unsafe(int source, int target) {
            if (m_nodes[target].is_root()) {
                m_roots.erase(target);
            }

            if (m_nodes[source].is_leaf()) {
                m_leaves.erase(source);
            }

            ArcGraph<DirectedGraph>::add_arc_unsafe(source, target);
        }

        void remove_arc_unsafe(int source, int target) {
            ArcGraph<DirectedGraph>::remove_arc_unsafe(source, target);
            
            if (m_nodes[target].is_root()) {
                m_roots.insert(target);
            }

            if (m_nodes[source].is_leaf()) {
                m_leaves.insert(source);
            }
        }

        bool has_path(int source, int target) const {
            check_valid_indices(source, target);
            return has_path_unsafe(source, target);
        }

        bool has_path(const std::string& source, const std::string& target) const {
            auto [s, t] = check_names(source, target);
            return has_path_unsafe(s, t);
        }

        bool has_path_unsafe(int source, int target) const;

        const std::unordered_set<int>& roots() const {
            return m_roots;
        }

        const std::unordered_set<int>& leaves() const {
            return m_leaves;
        }

    private:
        std::unordered_set<int> m_roots;
        std::unordered_set<int> m_leaves;
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

        bool can_add_arc(int source, int target) const {
            this->check_valid_indices(source, target);
            return can_add_arc_unsafe(source, target);
        }

        bool can_add_arc(const std::string& source, const std::string& target) const {
            auto [s, t] = this->check_names(source, target);
            return can_add_arc_unsafe(s, t);
        }

        bool can_add_arc_unsafe(int source, int target) const;

        bool can_flip_arc(int source, int target) {
            check_valid_indices(source, target);
            return can_flip_arc_unsafe(source, target);
        }

        bool can_flip_arc(const std::string& source, const std::string& target) {
            auto [s, t] = check_names(source, target);
            return can_flip_arc_unsafe(s, t);
        }

        bool can_flip_arc_unsafe(int source, int target);
    };



}

#endif //PGM_DATASET_GENERIC_GRAPH_HPP
