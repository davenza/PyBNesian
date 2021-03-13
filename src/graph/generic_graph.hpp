#ifndef PYBNESIAN_GRAPH_GENERIC_GRAPH_HPP
#define PYBNESIAN_GRAPH_GENERIC_GRAPH_HPP

#include <pybind11/pybind11.h>
#include <boost/dynamic_bitset.hpp>
#include <graph/graph_types.hpp>
#include <util/util_types.hpp>
#include <util/bidirectionalmap_index.hpp>
#include <util/vector.hpp>
#include <util/parameter_traits.hpp>
#include <util/temporal.hpp>

#include <iostream>

namespace py = pybind11;

using graph::DNode, graph::UNode, graph::PDNode;
using boost::dynamic_bitset;
using util::ArcStringVector, util::EdgeStringVector, util::ArcSet, util::EdgeSet, util::BidirectionalMapIndex;
using util::TemporalIndex;

namespace graph {

    enum GraphType {
        Directed,
        DirectedAcyclic,
        Undirected,
        PartiallyDirected
    };

    template<GraphType Type>
    class Graph;
    template<GraphType Type>
    class ConditionalGraph;
    template<GraphType Type>
    class TemporalGraph;

    using DirectedGraph = Graph<Directed>;
    using Dag = Graph<DirectedAcyclic>;
    using UndirectedGraph = Graph<Undirected>;
    using PartiallyDirectedGraph = Graph<PartiallyDirected>;

    using ConditionalDirectedGraph = ConditionalGraph<Directed>;
    using ConditionalDag = ConditionalGraph<DirectedAcyclic>;
    using ConditionalUndirectedGraph = ConditionalGraph<Undirected>;
    using ConditionalPartiallyDirectedGraph = ConditionalGraph<PartiallyDirected>;

    using TemporalDirectedGraph = TemporalGraph<Directed>;
    using TemporalDag = TemporalGraph<DirectedAcyclic>;
    using TemporalUndirectedGraph = TemporalGraph<Undirected>;
    using TemporalPartiallyDirectedGraph = TemporalGraph<PartiallyDirected>;

    template<typename G>
    struct GraphTraits;

    template<template<GraphType> typename _GraphClass>
    struct GraphTraits<_GraphClass<Directed>> {
        using NodeType = DNode;
        template<GraphType Type>
        using GraphClass = _GraphClass<Type>;
        inline static constexpr bool has_arcs = true;
        inline static constexpr bool has_edges = false;
    };

    template<template<GraphType> typename _GraphClass>
    struct GraphTraits<_GraphClass<DirectedAcyclic>> {
        using NodeType = DNode;
        template<GraphType Type>
        using GraphClass = _GraphClass<Type>;
        inline static constexpr bool has_arcs = true;
        inline static constexpr bool has_edges = false;
    };

    template<template<GraphType> typename _GraphClass>
    struct GraphTraits<_GraphClass<Undirected>> {
        using NodeType = UNode;
        template<GraphType Type>
        using GraphClass = _GraphClass<Type>;
        inline static constexpr bool has_arcs = false;
        inline static constexpr bool has_edges = true;
    };

    template<template<GraphType> typename _GraphClass>
    struct GraphTraits<_GraphClass<PartiallyDirected>> {
        using NodeType = PDNode;
        template<GraphType Type>
        using GraphClass = _GraphClass<Type>;
        inline static constexpr bool has_arcs = true;
        inline static constexpr bool has_edges = true;
    };

    template<typename G, typename _ = void>
    struct is_unconditional_graph : public std::false_type {};

    template<typename G>
    struct is_unconditional_graph<G,
                                std::void_t<
                                    util::GenericInstantation<GraphType>::enable_if_template_instantation_t<Graph, G>
                                >
    > : public std::true_type {};

    template<typename G>
    inline constexpr auto is_unconditional_graph_v = is_unconditional_graph<G>::value;

    template<typename G, typename R = void>
    using enable_if_unconditional_graph_t = std::enable_if_t<is_unconditional_graph_v<G>, R>;

    template<typename G, typename _ = void>
    struct is_conditional_graph : public std::false_type {};

    template<typename G>
    struct is_conditional_graph<G,
                                std::void_t<
                                    util::GenericInstantation<GraphType>::enable_if_template_instantation_t<ConditionalGraph, G>
                                >
    > : public std::true_type {};

    template<typename G>
    inline constexpr auto is_conditional_graph_v = is_conditional_graph<G>::value;

    template<typename G, typename R = void>
    using enable_if_conditional_graph_t = std::enable_if_t<is_conditional_graph_v<G>, R>;

    static_assert(util::GenericInstantation<GraphType>::is_template_instantation_v<Graph, Dag>,
                    "GenericInstantation failed for Graph, Dag.");
    static_assert(is_unconditional_graph_v<Dag>, "Dag is not unconditional graph");

    template<GraphType Type, template<GraphType> typename GraphClass>
    ConditionalGraph<Type> to_conditional_graph(const GraphClass<Type>& g,
                                                const std::vector<std::string>& nodes,
                                                const std::vector<std::string>& interface_nodes) {

        ConditionalGraph<Type> cgraph(nodes, interface_nodes);

        int num_total_nodes;
        if constexpr (is_unconditional_graph_v<GraphClass<Type>>) {
            num_total_nodes = g.num_nodes();
        } else if constexpr (is_conditional_graph_v<GraphClass<Type>>) {
            num_total_nodes = g.num_total_nodes();
        }

        if (cgraph.num_total_nodes() != num_total_nodes) {
            throw std::invalid_argument("The graph has " + std::to_string(g.num_nodes()) + " nodes, but " + 
                                        std::to_string(cgraph.num_total_nodes()) + 
                                        " nodes have been specified in the nodes/interface_nodes lists.");
        }

        for (const auto& node : cgraph.nodes()) {
            if constexpr (is_unconditional_graph_v<GraphClass<Type>>) {
                if (!g.contains_node(node))
                    throw std::invalid_argument("Node " + node + "in node list, not present in the graph");

            } else if constexpr (is_conditional_graph_v<GraphClass<Type>>) {
                if (!g.contains_total_node(node))
                    throw std::invalid_argument("Node " + node + "in node list, not present in the graph");
            } else
                static_assert(util::always_false<GraphClass<Type>>, "Wrong GraphType.");
        }

        for (const auto& node : cgraph.interface_nodes()) {
            if constexpr (is_unconditional_graph_v<GraphClass<Type>>) {
                if (!g.contains_node(node))
                    throw std::invalid_argument("Node " + node + "in interface_node list, not present in the graph");

            } else if constexpr (is_conditional_graph_v<GraphClass<Type>>) {
                if (!g.contains_total_node(node))
                    throw std::invalid_argument("Node " + node + "in interface_node list, not present in the graph");
            } else
                static_assert(util::always_false<GraphClass<Type>>, "Wrong GraphType.");
        }

        if constexpr (GraphTraits<Graph<Type>>::has_arcs) {
            for (const auto& arc : g.arc_indices()) {
                cgraph.add_arc(g.name(arc.first), g.name(arc.second));
            }
        }

        if constexpr (GraphTraits<Graph<Type>>::has_edges) {
            for (const auto& edge : g.edge_indices()) {
                cgraph.add_edge(g.name(edge.first), g.name(edge.second));
            }
        }

        return cgraph;
    }

    template<GraphType Type, template<GraphType> typename GraphClass>
    Graph<Type> to_unconditional_graph(const GraphClass<Type>& g) {
        if constexpr (is_unconditional_graph_v<GraphClass<Type>>) {
            return g;
        } else {
            std::vector<std::string> nodes;
            nodes.reserve(g.num_total_nodes());
            nodes.insert(nodes.end(), g.nodes().begin(), g.nodes().end());
            nodes.insert(nodes.end(), g.interface_nodes().begin(), g.interface_nodes().end());

            Graph<Type> graph(nodes);

            if constexpr (GraphTraits<ConditionalGraph<Type>>::has_arcs) {
                for (const auto& arc : g.arc_indices()) {
                    graph.add_arc(g.name(arc.first), g.name(arc.second));
                }
            }
            if constexpr (GraphTraits<ConditionalGraph<Type>>::has_edges) {
                for (const auto& edge : g.edge_indices()) {
                    graph.add_edge(g.name(edge.first), g.name(edge.second));
                }
            }

            return graph;
        }

    }

    template<typename G, enable_if_unconditional_graph_t<G, int> = 0>
    py::tuple __getstate__(const G& g) {
        std::vector<std::string> nodes;
        nodes.reserve(g.num_nodes());
        std::vector<Arc> arcs;
        if constexpr (GraphTraits<G>::has_arcs) arcs.reserve(g.num_arcs());
        std::vector<Edge> edges;
        if constexpr (GraphTraits<G>::has_edges) edges.reserve(g.num_edges());


        if (g.free_indices().empty()) {
            for (auto& n : g.raw_nodes()) {
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

            for (int i = 0, j = 0; i < g.num_raw_nodes(); ++i) {
                if (g.is_valid(i)) {
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

    template<typename G, enable_if_unconditional_graph_t<G, int> = 0>
    py::tuple __getstate__(G&& g) {
        return graph::__getstate__(g);
    }

    template<typename G, enable_if_conditional_graph_t<G, int> = 0>
    py::tuple __getstate__(const G& g) {
        std::vector<std::string> nodes;
        std::vector<std::string> interface_nodes;
        nodes.reserve(g.num_nodes());
        interface_nodes.reserve(g.num_interface_nodes());

        ArcStringVector arcs;
        if constexpr (GraphTraits<G>::has_arcs) arcs.reserve(g.num_arcs());
    
        EdgeStringVector edges;
        if constexpr (GraphTraits<G>::has_edges) edges.reserve(g.num_edges());

        if (g.free_indices().empty()) {
            for (auto& n : g.raw_nodes()) {
                const auto& name  = n.name();
                if (g.is_interface(name))
                    interface_nodes.push_back(name);
                else
                    nodes.push_back(name);
            }
        } else {
            for (int i = 0; i < g.num_raw_nodes(); ++i) {
                if (g.is_valid(i)) {
                    if (g.is_interface(i))
                        interface_nodes.push_back(g.name(i));
                    else
                        nodes.push_back(g.name(i));
                }
            }
        }

        if constexpr (GraphTraits<G>::has_arcs) {
            arcs = g.arcs();
        }

        if constexpr (GraphTraits<G>::has_edges) {
            edges = g.edges();
        }

        if constexpr (GraphTraits<G>::has_arcs) {
            if constexpr (GraphTraits<G>::has_edges) {
                return py::make_tuple(nodes, interface_nodes, arcs, edges);
            } else {
                return py::make_tuple(nodes, interface_nodes, arcs);
            }
        } else {
            return py::make_tuple(nodes, interface_nodes, edges);
        }
    }

    template<typename G, enable_if_conditional_graph_t<G, int> = 0>
    py::tuple __getstate__(G&& g) {
        return graph::__getstate__(g);
    }

    template<typename G, enable_if_unconditional_graph_t<G, int> = 0>
    G __setstate__(py::tuple& t) {
        if (t.size() != (1 + GraphTraits<G>::has_arcs + GraphTraits<G>::has_edges))
            throw std::runtime_error("Not valid Graph.");

        G g(t[0].cast<std::vector<std::string>>());

        if constexpr (GraphTraits<G>::has_arcs) {
            auto arcs = t[1].cast<std::vector<Arc>>();

            for (auto& arc : arcs) {
                g.add_arc_unsafe(arc.first, arc.second);
            }

            if constexpr (GraphTraits<G>::has_edges) {
                auto edges = t[2].cast<std::vector<Edge>>();

                for (auto& edge : edges) {
                    g.add_edge_unsafe(edge.first, edge.second);
                }
            }
        } else {
            auto edges = t[1].cast<std::vector<Edge>>();

            for (auto& edge : edges) {
                g.add_edge_unsafe(edge.first, edge.second);
            }
        }
        
        return g;
    }

    template<typename G, enable_if_unconditional_graph_t<G, int> = 0>
    G __setstate__(py::tuple&& t) {
        return graph::__setstate__<G>(t);
    }

    template<typename G, enable_if_conditional_graph_t<G, int> = 0>
    G __setstate__(py::tuple& t) {
        if (t.size() != (2 + GraphTraits<G>::has_arcs + GraphTraits<G>::has_edges))
            throw std::runtime_error("Not valid Graph.");

        G g(t[0].cast<std::vector<std::string>>(), t[1].cast<std::vector<std::string>>());

        if constexpr (GraphTraits<G>::has_arcs) {
            auto arcs = t[2].cast<ArcStringVector>();

            for (const auto& arc : arcs) {
                g.add_arc_unsafe(g.index(arc.first), g.index(arc.second));
            }

            if constexpr (GraphTraits<G>::has_edges) {
                auto edges = t[3].cast<EdgeStringVector>();

                for (const auto& edge : edges) {
                    g.add_edge_unsafe(g.index(edge.first), g.index(edge.second));
                }
            }
        } else {
            auto edges = t[2].cast<EdgeStringVector>();

            for (const auto& edge : edges) {
                g.add_edge_unsafe(g.index(edge.first), g.index(edge.second));
            }
        }
        
        return g;
    }

    template<typename G, enable_if_conditional_graph_t<G, int> = 0>
    G __setstate__(py::tuple&& t) {
        return graph::__setstate__<G>(t);
    }

    template<typename G>
    void save_graph(const G& graph, std::string name) {
        auto open = py::module::import("io").attr("open");
        
        if (name.size() < 7 || name.substr(name.size()-7) != ".pickle")
            name += ".pickle";

        auto file = open(name, "wb");
        py::module::import("pickle").attr("dump")(py::cast(&graph), file, 2);
        file.attr("close")();
    }

    template<typename Derived, template<typename> typename BaseClass>
    class ArcGraph;

    template<typename _Derived, template<typename> typename BaseClass>
    class PartiallyDirectedImpl;

    template<typename Derived>
    class GraphBase {
    public:
        using NodeType = typename GraphTraits<Derived>::NodeType;

        inline Derived& derived() { return static_cast<Derived&>(*this); }
        inline const Derived& derived() const { return static_cast<const Derived&>(*this); }

        template<typename G, template<typename> typename BaseClass>
        friend class ArcGraph;
        template<typename G, template<typename> typename BaseClass>
        friend class EdgeGraph;
        
        GraphBase() = default;
        GraphBase(const std::vector<std::string>& nodes) : m_nodes(),
                                                           m_indices(),
                                                           m_string_nodes(nodes),
                                                           m_free_indices() {
            m_nodes.reserve(nodes.size());

            for (size_t i = 0; i < nodes.size(); ++i) {
                NodeType n(i, nodes[i]);
                m_nodes.push_back(n);
                m_indices.insert(std::make_pair(nodes[i], i));
            }

            if (m_indices.size() != m_nodes.size()) {
                throw std::invalid_argument("Graph cannot be created with repeated names.");
            }
        }

        int num_nodes() const {
            return m_string_nodes.size();
        }

        int num_raw_nodes() const {
            return m_nodes.size();
        }

        template<typename V>
        const NodeType& raw_node(const V& idx) const {
            return m_nodes[check_index(idx)]; 
        }

        const std::vector<std::string>& nodes() const {
            return m_string_nodes.elements();
        }
        
        const std::vector<NodeType>& raw_nodes() const { return m_nodes; }

        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
        }

        bool contains_node(const std::string& name) const {
            return m_string_nodes.contains(name);
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

        const std::string& collapsed_name(int collapsed_index) const {
            return m_string_nodes.element(collapsed_index);
        }

        int index(const std::string& node) const {
            return check_index(node);
        }

        int collapsed_index(const std::string& node) const {
            return m_string_nodes.index(node);
        }

        int index_from_collapsed(int collapsed_index) const {
            return index(m_string_nodes.element(collapsed_index));
        }

        int collapsed_from_index(int index) const {
            return collapsed_index(name(index));
        }

        const std::unordered_map<std::string, int>& collapsed_indices() const {
            return m_string_nodes.indices();
        }

        const std::vector<int>& free_indices() const {
            return m_free_indices;
        }

        bool is_valid(int idx) const {
            return idx >= 0 && idx < num_raw_nodes() && m_nodes[idx].is_valid();
        }
        
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
    protected:
        int create_node(const std::string& node);
        void remove_node_arcs_edges(int index);

    private:
        std::vector<NodeType> m_nodes;
        std::unordered_map<std::string, int> m_indices;
        BidirectionalMapIndex<std::string> m_string_nodes;
        std::vector<int> m_free_indices;
    };

    template<typename Derived>
    int GraphBase<Derived>::create_node(const std::string& node) {
        if (!m_free_indices.empty()) {
            int idx = m_free_indices.back();
            m_free_indices.pop_back();
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
    }

    template<typename Derived>
    int GraphBase<Derived>::add_node(const std::string& node) {
        if (contains_node(node)) {
            throw std::invalid_argument("Cannot add node " + node + " because a node with the same name already exists.");
        }

        int idx = create_node(node);

        m_indices.insert(std::make_pair(node, idx));
        m_string_nodes.insert(node);

        if constexpr (GraphTraits<Derived>::has_arcs) {
            auto& arcg = static_cast<Derived&>(*this).arc_base();
            arcg.add_root(idx);
            arcg.add_leaf(idx);
        }

        return idx;
    }

    template<typename Derived>
    void GraphBase<Derived>::remove_node_arcs_edges(int index) {
        if constexpr (GraphTraits<Derived>::has_edges) {
            auto& derived = static_cast<Derived&>(*this);
            for (auto neighbor : derived.neighbor_indices(index)) {
                derived.remove_edge_unsafe(index, neighbor);
            }
        }

        if constexpr (GraphTraits<Derived>::has_arcs) {
            if (m_nodes[index].is_root()) {
                auto& arcg = static_cast<Derived&>(*this).arc_base();
                arcg.remove_root(index);
            }

            if (m_nodes[index].is_leaf()) {
                auto& arcg = static_cast<Derived&>(*this).arc_base();
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
    }

    template<typename Derived>
    void GraphBase<Derived>::remove_node_unsafe(int index) {
        remove_node_arcs_edges(index);

        m_string_nodes.remove(m_nodes[index].name());
        m_indices.erase(m_nodes[index].name());
        m_nodes[index].invalidate();
        m_free_indices.push_back(index);
    }

    template<typename Derived>
    class ConditionalGraphBase {
    public:
        using NodeType = typename GraphTraits<Derived>::NodeType;

        inline Derived& derived() { return static_cast<Derived&>(*this); }
        inline const Derived& derived() const { return static_cast<const Derived&>(*this); }

        template<typename G, template<typename> typename BaseClass>
        friend class ArcGraph;
        template<typename G, template<typename> typename BaseClass>
        friend class EdgeGraph;

        ConditionalGraphBase() = default;
        ConditionalGraphBase(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes) : m_nodes(),
                                                                                m_string_nodes(nodes),
                                                                                m_interface_nodes(interface_nodes),
                                                                                m_total_nodes(),
                                                                                m_indices(),
                                                                                m_free_indices() {
            if (nodes.empty())
                throw std::invalid_argument("Nodes can not be empty.");

            m_total_nodes.reserve(nodes.size() + interface_nodes.size());
            m_total_nodes.insert(m_string_nodes.begin(), m_string_nodes.end());
            m_total_nodes.insert(m_interface_nodes.begin(), m_interface_nodes.end());

            m_nodes.reserve(m_total_nodes.size());
            for (size_t i = 0; i < m_total_nodes.size(); ++i) {
                NodeType n(i, m_total_nodes[i]);
                m_nodes.push_back(n);
                
                m_indices.insert({m_total_nodes[i], i});
            }

            if (m_indices.size() != (nodes.size() + interface_nodes.size()))
                throw std::invalid_argument("Nodes and interface nodes contain repeated names.");
        }

        int num_nodes() const {
            return m_string_nodes.size();
        }

        int num_raw_nodes() const {
            return m_nodes.size();
        }

        int num_interface_nodes() const {
            return m_interface_nodes.size();
        }

        int num_total_nodes() const {
            return m_total_nodes.size();
        }

        template<typename V>
        const NodeType& raw_node(const V& idx) const {
            return m_nodes[check_index(idx)]; 
        }

        const std::vector<std::string>& nodes() const {
            return m_string_nodes.elements();
        }

        const std::vector<std::string>& interface_nodes() const {
            return m_interface_nodes.elements();
        }

        const std::vector<std::string>& all_nodes() const {
            return m_total_nodes.elements();
        }

        const std::vector<NodeType>& raw_nodes() const { return m_nodes; }

        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
        }

        const std::unordered_map<std::string, int>& collapsed_indices() const {
            return m_string_nodes.indices();
        }

        const std::unordered_map<std::string, int>& interface_collapsed_indices() const {
            return m_interface_nodes.indices();
        }

        const std::unordered_map<std::string, int>& joint_collapsed_indices() const {
            return m_total_nodes.indices();
        }

        bool contains_node(const std::string& name) const {
            return m_string_nodes.contains(name);
        }

        bool contains_interface_node(const std::string& name) const {
            return contains_total_node(name) && !contains_node(name);   
        }

        bool contains_total_node(const std::string& name) const {
            return m_total_nodes.contains(name);
        }

        int add_node(const std::string& node);
        int add_interface_node(const std::string& node);

        template<typename V>
        void remove_node(const V& idx) {
            remove_node_unsafe(check_index(idx));
        }

        void remove_node_unsafe(int index);

        template<typename V>
        void remove_interface_node(const V& idx) {
            remove_interface_node_unsafe(check_index(idx));
        }

        void remove_interface_node_unsafe(int index);

        const std::string& name(int idx) const {
            return m_nodes[check_index(idx)].name();
        }

        const std::string& collapsed_name(int collapsed_index) const {
            return m_string_nodes.element(collapsed_index);
        }

        const std::string& interface_collapsed_name(int interface_collapsed_index) const {
            return m_interface_nodes.element(interface_collapsed_index);
        }

        const std::string& joint_collapsed_name(int joint_collapsed_index) const {
            return m_total_nodes.element(joint_collapsed_index);
        }

        int index(const std::string& node) const {
            return check_index(node);
        }

        int collapsed_index(const std::string& node) const {
            return m_string_nodes.index(node);
        }

        int interface_collapsed_index(const std::string& node) const {
            return m_interface_nodes.index(node);
        }

        int joint_collapsed_index(const std::string& node) const {
            return m_total_nodes.index(node);
        }

        int index_from_collapsed(int collapsed_index) const {
            return index(collapsed_name(collapsed_index));
        }

        int index_from_interface_collapsed(int interface_collapsed_index) const {
            return index(interface_collapsed_name(interface_collapsed_index));
        }

        int index_from_joint_collapsed(int joint_collapsed_index) const {
            return index(joint_collapsed_name(joint_collapsed_index));
        }

        int collapsed_from_index(int index) const {
            return m_string_nodes[name(index)];
        }

        int interface_collapsed_from_index(int index) const {
            return m_interface_nodes[name(index)];
        }

        int joint_collapsed_from_index(int index) const {
            return m_total_nodes[name(index)];
        }

        bool is_interface(int index) const {
            return contains_interface_node(m_nodes[check_index(index)].name());   
        }
        
        bool is_interface(const std::string& name) const {
            return contains_interface_node(m_nodes[check_index(name)].name());
        }

        void set_interface(int index) {
            if (!is_interface(index)) {
                m_string_nodes.remove(m_nodes[index].name());
                m_interface_nodes.insert(m_nodes[index].name());
            }
        }

        void set_interface(const std::string& node) {
            if (!is_interface(node)) {
                m_string_nodes.remove(node);
                m_interface_nodes.insert(node);
            }
        }

        void set_node(int index) {
            if (is_interface(index)) {
                const auto& node_name = name(index);
                m_string_nodes.insert(node_name);
                m_interface_nodes.remove(node_name);
            }
        }

        void set_node(const std::string& node) {
            if (is_interface(node)) {
                m_string_nodes.insert(node);
                m_interface_nodes.remove(node);
            }
        }

        const std::vector<int> free_indices() const {
            return m_free_indices;
        }

        bool is_valid(int idx) const {
            return idx >= 0 && static_cast<size_t>(idx) < m_nodes.size() && m_nodes[idx].is_valid();
        }

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
    private:
        int create_node(const std::string& node);
        void remove_node_arcs_edges(int index);


        std::vector<NodeType> m_nodes;
        // std::vector<std::string> m_string_nodes;
        BidirectionalMapIndex<std::string> m_string_nodes;
        BidirectionalMapIndex<std::string> m_interface_nodes;
        BidirectionalMapIndex<std::string> m_total_nodes;
        
        // all nodes -> index
        std::unordered_map<std::string, int> m_indices;
        std::vector<int> m_free_indices;
    };

    template<typename Derived>
    int ConditionalGraphBase<Derived>::create_node(const std::string& node) {
        if (!m_free_indices.empty()) {
            int idx = m_free_indices.back();
            m_free_indices.pop_back();
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
    }

    template<typename Derived>
    int ConditionalGraphBase<Derived>::add_node(const std::string& node) {
        if (contains_total_node(node)) {
            throw std::invalid_argument("Cannot add node " + node + " because a node with the same name already exists.");
        }

        int idx = create_node(node);
        m_indices.insert({node, idx});

        m_string_nodes.insert(node);
        m_total_nodes.insert(node);

        if constexpr (GraphTraits<Derived>::has_arcs) {
            auto& arcg = static_cast<Derived&>(*this).arc_base();
            arcg.add_root(idx);
            arcg.add_leaf(idx);
        }

        return idx;
    }

    template<typename Derived>
    int ConditionalGraphBase<Derived>::add_interface_node(const std::string& node) {
        if (contains_total_node(node)) {
            throw std::invalid_argument("Cannot add node " + node + " because a node with the same name already exists.");
        }

        int idx = create_node(node);
        m_indices.insert({node, idx});
        m_interface_nodes.insert(node);
        m_total_nodes.insert(node);

        if constexpr (GraphTraits<Derived>::has_arcs) {
            auto& arcg = static_cast<Derived&>(*this).arc_base();
            arcg.add_root(idx);
            arcg.add_leaf(idx);
        }

        return idx;
    }

    template<typename Derived>
    void ConditionalGraphBase<Derived>::remove_node_arcs_edges(int index) {
        if constexpr (GraphTraits<Derived>::has_edges) {
            auto& derived = static_cast<Derived&>(*this);
            for (auto neighbor : derived.neighbor_indices(index)) {
                derived.remove_edge_unsafe(index, neighbor);
            }
        }

        if constexpr (GraphTraits<Derived>::has_arcs) {
            if (m_nodes[index].is_root()) {
                auto& arcg = static_cast<Derived&>(*this).arc_base();
                arcg.remove_root(index);
            }

            if (m_nodes[index].is_leaf()) {
                auto& arcg = static_cast<Derived&>(*this).arc_base();
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
    }

    template<typename Derived>
    void ConditionalGraphBase<Derived>::remove_node_unsafe(int index) {
        remove_node_arcs_edges(index);

        m_indices.erase(m_nodes[index].name());
        
        m_string_nodes.remove(m_nodes[index].name());
        m_total_nodes.remove(m_nodes[index].name());

        m_nodes[index].invalidate();
        m_free_indices.push_back(index);
    }

    template<typename Derived>
    void ConditionalGraphBase<Derived>::remove_interface_node_unsafe(int index) {
        remove_node_arcs_edges(index);

        m_indices.erase(m_nodes[index].name());
        
        m_interface_nodes.remove(m_nodes[index].name());
        m_total_nodes.remove(m_nodes[index].name());

        m_nodes[index].invalidate();
        m_free_indices.push_back(index);
    }

    template<typename Derived>
    class TemporalGraphBase {
    public:
        using NodeType = typename GraphTraits<Derived>::NodeType;

        TemporalGraphBase() = default;
        TemporalGraphBase(const std::vector<std::string>& variables,
                          int markovian_order,
                          const std::vector<std::string>& free_nodes) : m_nodes(),
                                                                        m_indices(),
                                                                        m_slice_indices(),
                                                                        m_string_nodes(),
                                                                        m_variables(variables),
                                                                        m_free_indices(),
                                                                        m_markovian(markovian_order) {
            m_nodes.reserve(variables.size() * (markovian_order + 1) + free_nodes.size());

            int nindex = 0;
            for (int i = 0; i <= markovian_order; ++i) {
                for (const auto& v : m_variables) {
                    auto name = util::temporal_name(v, i);
                    NodeType n(nindex, name);
                    m_nodes.push_back(n);
                    m_indices.insert({name, nindex});
                    m_slice_indices.insert({m_variables.index(v), i}, nindex++);
                    m_string_nodes.insert(name);
                }
            }

            for (auto& name : free_nodes) {
                NodeType n(nindex, name);
                m_nodes.push_back(n);
                m_indices.insert({name, nindex});
                m_string_nodes.insert(name);
            }
        }

        int num_nodes() const {
            return m_nodes.size() - m_free_indices.size();
        }

        int num_variables() const {
            return m_variables.size();
        }

        int num_raw_nodes() const {
            return m_nodes.size();
        }

        template<typename V>
        const NodeType& raw_node(const V& idx) const {
            return m_nodes[check_index(idx)]; 
        }

        const std::vector<std::string>& nodes() const {
            return m_string_nodes.elements();
        }

        const std::vector<NodeType>& raw_nodes() const { return m_nodes; }

        const std::unordered_map<std::string, int>& indices() const {
            return m_indices;
        }

        bool contains_node(const std::string& name) const {
            return m_string_nodes.contains(name);
        }

        template<typename Index>
        bool contains_node(const TemporalIndex<Index>& name) const {
            return contains_node(name.temporal_name());
        }

        bool contains_variable(const std::string& variable) const {
            return m_variables.contains(variable);
        }

        bool is_free_node(const std::string& name) const {
            auto tindex = TemporalIndex<std::string>::from_string(name);

            if (tindex && contains_variable(tindex->variable) && tindex->temporal_slice >= 0 && tindex->temporal_slice <= m_markovian) {
                return true;
            }

            return false;
        }

        int add_node(const std::string& node);
        int add_variable(const std::string& variable);

        template<typename V>
        void remove_node(const V& idx) {
            remove_node_unsafe(check_index(idx));
        }

        void remove_node_unsafe(int index);

        void remove_variable(const std::string& variable);


        const std::string& name(int global_idx) const {
            return m_nodes[check_index(global_idx)].name();
        }

        const std::string& collapsed_name(int collapsed_index) const {
            return m_string_nodes.element(collapsed_index);
        }

        const std::string& collapsed_variable(int collapsed_variable) const {
            return m_variables.element(collapsed_variable);
        }

        template<typename Index>
        int index(const Index& index) const {
            return check_index(index);
        }

        int collapsed_index(const std::string& node) const {
            return m_string_nodes.index(node);
        }

        template<typename Index>
        int collapsed_index(const TemporalIndex<Index>& node) const {
            return collapsed_index(node.temporal_name());
        }
        
        int collapsed_variable_index(const std::string& variable) const {
            return m_variables.index(variable);
        }

        int index_from_collapsed(int collapsed_index) const {
            return index(m_string_nodes.element(collapsed_index));
        }

        int collapsed_from_index(int index) const {
            return collapsed_index(name(index));
        }

        const std::unordered_map<std::string, int>& collapsed_indices() const {
            return m_string_nodes.indices();
        }

        const std::unordered_map<std::string, int>& collapsed_variable_indices() const {
            return m_variables.indices();
        }

        const std::vector<int>& free_indices() const {
            return m_free_indices;
        }

        bool is_valid(int global_idx) const {
            return global_idx >= 0 && global_idx < num_raw_nodes() && m_nodes[global_idx].is_valid();
        }

        int check_index(int global_idx) const {
            if (!is_valid(global_idx)) {
                throw std::invalid_argument("Node index " + std::to_string(global_idx) + " invalid.");
            }

            return global_idx;
        }

        int check_index(const std::string& name) const {
            auto f = m_indices.find(name);
            if (f == m_indices.end()) {
                throw std::invalid_argument("Node " + name + " not present in the graph.");
            }

            return f->second;
        }

        int check_index(const TemporalIndex<int>& ti) const {
            auto f = m_slice_indices.find(ti);
            if (f == m_slice_indices.end()) {
                throw std::invalid_argument("Node " + ti.temporal_name() + " not present in the graph.");
            }

            return f->second;
        }

        int check_index(const TemporalIndex<std::string>& ti) const {
            return check_index({collapsed_variable_index(ti.variable), temporal_slice})
        }
    protected:
        int create_node(const std::string& node);
        int create_variable(const std::string& variable);
        void remove_node_arcs_edges(int index);
        void remove_variable_arcs_edges(int index);

    private:
        std::vector<NodeType> m_nodes;
        std::unordered_map<std::string, int> m_indices;
        std::unordered_map<TemporalIndex<int>, int> m_slice_indices;
        BidirectionalMapIndex<std::string> m_string_nodes;
        BidirectionalMapIndex<std::string> m_variables;
        std::vector<int> m_free_indices;
        int m_markovian;
    };

    template<typename Derived>
    int TemporalGraphBase<Derived>::create_node(const std::string& node) {
        if (!m_free_indices.empty()) {
            int idx = m_free_indices.back();
            m_free_indices.pop_back();
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
    }


    template<typename Derived>
    int TemporalGraphBase<Derived>::add_node(const std::string& node) {
        if (contains_node(node)) {
            throw std::invalid_argument("Cannot add node " + node + " because a node with the same name already exists.");
        }

        int idx = create_node(node);

        m_indices.insert(std::make_pair(node, idx));
        m_string_nodes.insert(node);

        if constexpr (GraphTraits<Derived>::has_arcs) {
            auto& arcg = static_cast<Derived&>(*this).arc_base();
            arcg.add_root(idx);
            arcg.add_leaf(idx);
        }

        return idx;
    }
    
    template<typename Derived>
    int TemporalGraphBase<Derived>::add_variable(const std::string& variable) {
        if (contains_variable(variable)) {
            throw std::invalid_argument("Can not add variable " + variable + " because a variable with the same name already exists.");
        }

        for (int i = 0; i <= m_markovian; ++i) {
            auto tindex = TemporalIndex(variable, i);
            if (contains_node(tindex)) {
                throw std::invalid_argument("Can not add variable " + variable + " because a free node " + tindex.temporal_name() 
                                            + " in the graph collides with the variable node.");
            }
        }

        for (int i = 0; i <= m_markovian; ++i) {
            auto tindex = TemporalIndex(variable, i);

            int idx = create_node(tindex.temporal_name());

            m_indices.insert({tindex.temporal_name(), idx});
            m_slice_indices.insert({TemporalIndex{index(tindex.variable), tindex.temporal_slice}, idx});
            m_string_nodes.insert(tindex.temporal_name());
        }

        m_variables.insert(variable);

        return m_variables.index(variable);
    }

    template<typename Derived>
    void TemporalGraphBase<Derived>::remove_node_arcs_edges(int index) {
        if constexpr (GraphTraits<Derived>::has_edges) {
            auto& derived = static_cast<Derived&>(*this);
            for (auto neighbor : derived.neighbor_indices(index)) {
                derived.remove_edge_unsafe(index, neighbor);
            }
        }

        if constexpr (GraphTraits<Derived>::has_arcs) {
            if (m_nodes[index].is_root()) {
                auto& arcg = static_cast<Derived&>(*this).arc_base();
                arcg.remove_root(index);
            }

            if (m_nodes[index].is_leaf()) {
                auto& arcg = static_cast<Derived&>(*this).arc_base();
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
    }
    
    template<typename Derived>
    void TemporalGraphBase<Derived>::remove_node_unsafe(int index) {
        remove_node_arcs_edges(index);

        m_string_nodes.remove(m_nodes[index].name());
        m_indices.erase(m_nodes[index].name());
        m_nodes[index].invalidate();
        m_free_indices.push_back(index);
    }

    template<typename Derived>
    void TemporalGraphBase<Derived>::remove_variable(const std::string& variable) {
        if (!contains_variable(variable)) {
            throw std::invalid_argument("Variable " + variable + " not present in the graph.");
        }

        auto variable_index = collapsed_variable_index(variable);
        for (int i = 0; i <= m_markovian; ++i) {
            remove_node({variable_index, i});
            m_slice_indices.erase({variable_index, i});
        }

        m_variables.remove(variable);
    }

    template<typename Derived>
    bool can_exist_arc(const GraphBase<Derived>&, int, int) { return true; }
    template<typename Derived>
    bool can_exist_arc(const ConditionalGraphBase<Derived>& g, int, int target) {
        if (g.is_interface(target)) {
            return false;
        }

        return true;
    }
    template<typename Derived>
    bool can_exist_arc(const TemporalGraphBase<Derived>& g, int source, int target) {
        if (!is_free_node(source) && !is_free_node(target)) {

        }

        return true;
    }

    template<typename Derived>
    bool can_exist_edge(const GraphBase<Derived>&, int, int) { return true; }
    template<typename Derived>
    bool can_exist_edge(const ConditionalGraphBase<Derived>&g, int source, int target) {
        if (g.is_interface(source) && g.is_interface(target)) {
            return false;
        }

        return true;
    }

    template<typename Derived>
    void check_can_exist_arc(const GraphBase<Derived>&, int, int) {}
    template<typename Derived>
    void check_can_exist_arc(const ConditionalGraphBase<Derived>& g, int source, int target) {
        if (!can_exist_arc(g, source, target)) {
            throw std::invalid_argument("Interface node can not have parents.");
        }
    }
    template<typename Derived>
    void check_can_exist_arc(const TemporalGraphBase<Derived>& g, int source, int target) {
        if (!can_exist_arc(g, source, target)) {
            throw std::invalid_argument("Cannot add arc " + g.name(source) + " -> " + g.name(target));
        }
    }

    template<typename Derived>
    void check_can_exist_edge(const GraphBase<Derived>&, int, int) {}
    template<typename Derived>
    void check_can_exist_edge(const ConditionalGraphBase<Derived>&g, int source, int target) {
        if (!can_exist_edge(g, source, target)) {
            throw std::invalid_argument("An edge cannot exist between interface nodes.");
        }
    }

    template<typename Derived, template<typename> typename BaseClass>
    class ArcGraph {
    public:
        using Base = BaseClass<Derived>;
        using NodeType = typename GraphTraits<Derived>::NodeType;
        inline Base& base() { return static_cast<Base&>(static_cast<Derived&>(*this)); }
        inline const Base& base() const { return static_cast<const Base&>(static_cast<const Derived&>(*this)); }
        inline ArcGraph<Derived, BaseClass>& arc_base() { return *this; }
        inline const ArcGraph<Derived, BaseClass>& arc_base() const { return *this; }

        friend class BaseClass<Derived>;

        ArcGraph() = default;
        ArcGraph(const std::vector<std::string>& nodes) : m_arcs(),
                                                          m_roots(),
                                                          m_leaves() {
            for (const auto& name : nodes) {
                m_roots.insert(base().index(name));
                m_leaves.insert(base().index(name));
            }
        }


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
            return base().raw_nodes()[idx].children().size();
        }

        ArcStringVector arcs() const;

        template<typename D = Derived, enable_if_conditional_graph_t<D, int> = 0>
        ArcStringVector interface_arcs() const;
        
        const ArcSet& arc_indices() const { return m_arcs; }

        template<typename V>
        std::vector<std::string> parents(const V& idx) const {
            return parents(base().raw_node(idx));
        }

        template<typename V>
        std::vector<int> parent_indices(const V& idx) const {
            auto& p = base().raw_node(idx).parents();
            return { p.begin(), p.end() };
        }

        template<typename V>
        const std::unordered_set<int>& parent_set(const V& idx) const {
            return base().raw_node(idx).parents();
        }

        template<typename V>
        std::string parents_to_string(const V& idx) const {
            return parents_to_string(base().raw_node(idx));
        }

        template<typename V>
        std::vector<std::string> children(const V& idx) const {
            return children(base().raw_node(idx));
        }

        template<typename V>
        std::vector<int> children_indices(const V& idx) const {
            auto& p = base().raw_node(idx).children();
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
            if (!has_arc_unsafe(s, t)) {
                check_can_exist_arc(base(), s, t);
                add_arc_unsafe(s, t);
            }
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
                remove_arc_unsafe(s, t);
        }

        void remove_arc_unsafe(int source, int target);

        template<typename V>
        void flip_arc(const V& source, const V& target) {
            auto s = base().check_index(source);
            auto t = base().check_index(target);
            if (has_arc_unsafe(s, t)) {
                check_can_exist_arc(base(), t, s);
                flip_arc_unsafe(s, t);
            }
        }

        void flip_arc_unsafe(int source, int target);

        const std::unordered_set<int>& roots() const {
            return m_roots;
        }

        const std::unordered_set<int>& leaves() const {
            return m_leaves;
        }
    private:
        // friend int GraphBase<Derived>::add_node(const std::string& node);
        // friend void GraphBase<Derived>::remove_node_unsafe(int index);

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

    template<typename Derived, template<typename> typename BaseClass>
    ArcStringVector ArcGraph<Derived, BaseClass>::arcs() const {
        ArcStringVector res;
        res.reserve(m_arcs.size());
        
        for (auto& arc : m_arcs) {
            res.push_back({base().m_nodes[arc.first].name(), 
                           base().m_nodes[arc.second].name()});
        }
        return res;
    }

    template<typename Derived, template<typename> typename BaseClass>
    template<typename D, enable_if_conditional_graph_t<D, int>>
    ArcStringVector ArcGraph<Derived, BaseClass>::interface_arcs() const {
        ArcStringVector res;

        for (const auto& inode : base().interface_nodes()) {
            const auto& child_set = children_set(inode);

            for (auto children : child_set) {
                res.push_back({inode, base().name(children)});
            }
        }

        return res;
    }

    template<typename Derived, template<typename> typename BaseClass>
    void ArcGraph<Derived, BaseClass>::add_arc_unsafe(int source, int target) {
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

    template<typename Derived, template<typename> typename BaseClass>
    void ArcGraph<Derived, BaseClass>::remove_arc_unsafe(int source, int target) {
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

    template<typename Derived, template<typename> typename BaseClass>
    void ArcGraph<Derived, BaseClass>::flip_arc_unsafe(int source, int target) {
        remove_arc_unsafe(source, target);
        add_arc_unsafe(target, source);
    }

    template<typename Derived, template<typename> typename BaseClass>
    std::vector<std::string> ArcGraph<Derived, BaseClass>::parents(const NodeType& n) const {
        std::vector<std::string> res;

        const auto& parent_indices = n.parents();
        res.reserve(parent_indices.size());

        for (auto node : parent_indices) {
            res.push_back(base().m_nodes[node].name());
        }

        return res;
    }

    template<typename Derived, template<typename> typename BaseClass>
    std::vector<std::string> ArcGraph<Derived, BaseClass>::children(const NodeType& n) const {
        std::vector<std::string> res;

        const auto& parent_indices = n.children();
        res.reserve(parent_indices.size());

        for (auto node : parent_indices) {
            res.push_back(base().m_nodes[node].name());
        }

        return res;
    }

    template<typename Derived, template<typename> typename BaseClass>
    std::string ArcGraph<Derived, BaseClass>::parents_to_string(const NodeType& n) const {
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

    template<typename Derived, template<typename> typename BaseClass>
    class EdgeGraph {
    public:
        using Base = BaseClass<Derived>;
        using NodeType = typename GraphTraits<Derived>::NodeType;

        inline Base& base() { return static_cast<Base&>(static_cast<Derived&>(*this)); }
        inline const Base& base() const { return static_cast<const Base&>(static_cast<const Derived&>(*this)); }
        inline EdgeGraph<Derived, BaseClass>& edge_base() { return *this; }
        inline const EdgeGraph<Derived, BaseClass>& edge_base() const { return *this; }

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
        template<typename D = Derived, enable_if_conditional_graph_t<D, int> = 0>
        ArcStringVector interface_edges() const;
        const EdgeSet& edge_indices() const { return m_edges; }

        template<typename V>
        std::vector<std::string> neighbors(const V& idx) const {
            return neighbors(base().raw_node(idx));
        }

        template<typename V>
        std::vector<int> neighbor_indices(const V& idx) const {
            const auto& n = base().raw_node(idx).neighbors();
            return { n.begin(), n.end() };
        }

        template<typename V>
        const std::unordered_set<int>& neighbor_set(const V& idx) const {
            return base().raw_node(idx).neighbors();
        }

        template<typename V>
        void add_edge(const V& source, const V& target) {
            auto s = base().check_index(source);
            auto t = base().check_index(target);
            if (!has_edge_unsafe(s, t)) {
                check_can_exist_edge(base(), s, t);
                add_edge_unsafe(s, t);
            }
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

    template<typename Derived, template<typename> typename BaseClass>
    EdgeStringVector EdgeGraph<Derived, BaseClass>::edges() const {
        EdgeStringVector res;
        res.reserve(m_edges.size());

        for (auto& edge : m_edges) {
            res.push_back({base().m_nodes[edge.first].name(), 
                           base().m_nodes[edge.second].name()});
        }

        return res;
    }

    template<typename Derived, template<typename> typename BaseClass>
    template<typename D, enable_if_conditional_graph_t<D, int>>
    ArcStringVector EdgeGraph<Derived, BaseClass>::interface_edges() const {
        ArcStringVector res;

        for (const auto& inode : base().interface_nodes()) {
            const auto& neigh_set = neighbor_set(inode);

            for (auto neigh : neigh_set) {
                res.push_back({inode, base().name(neigh)});
            }
        }

        return res;
    }

    template<typename Derived, template<typename> typename BaseClass>
    void EdgeGraph<Derived, BaseClass>::add_edge_unsafe(int source, int target) {
        m_edges.insert({source, target});
        base().m_nodes[source].add_neighbor(target);
        base().m_nodes[target].add_neighbor(source);
    }

    template<typename Derived, template<typename> typename BaseClass>
    void EdgeGraph<Derived, BaseClass>::remove_edge_unsafe(int source, int target) {
        m_edges.erase({source, target});
        base().m_nodes[source].remove_neighbor(target);
        base().m_nodes[target].remove_neighbor(source);
    }

    template<typename Derived, template<typename> typename BaseClass>
    std::vector<std::string> EdgeGraph<Derived, BaseClass>::neighbors(const NodeType& n) const {
        std::vector<std::string> res;

        const auto& neighbors_indices = n.neighbors();
        res.reserve(neighbors_indices.size());

        for (auto node : neighbors_indices) {
            res.push_back(base().m_nodes[node].name());
        }

        return res;
    }

    template<typename Derived, template<typename> typename BaseClass>
    class PartiallyDirectedImpl : public BaseClass<Derived>,
                                  public ArcGraph<Derived, BaseClass>,
                                  public EdgeGraph<Derived, BaseClass> {
    public:
        template<GraphType Type>
        using GraphClass = typename GraphTraits<Derived>::template GraphClass<Type>;

        PartiallyDirectedImpl() = default;

        // /////////////////////////////////////
        // GraphBase constructors
        // /////////////////////////////////////
        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        PartiallyDirectedImpl(const std::vector<std::string>& nodes) 
                                        : BaseClass<Derived>(nodes),
                                          ArcGraph<Derived, BaseClass>(nodes),
                                          EdgeGraph<Derived, BaseClass>() {}

        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        PartiallyDirectedImpl(const ArcStringVector& arcs, const EdgeStringVector& edges) 
                                        : BaseClass<Derived>(),
                                          ArcGraph<Derived, BaseClass>(),
                                          EdgeGraph<Derived, BaseClass>() {

            for (auto& arc : arcs) {
                if (!this->contains_node(arc.first)) {
                    this->add_node(arc.first);
                }

                if (!this->contains_node(arc.second)) {
                    this->add_node(arc.second);
                }

                this->add_arc(arc.first, arc.second);
            }

            for (auto& edge : edges) {
                if (!this->contains_node(edge.first)) {
                    this->add_node(edge.first);
                }

                if (!this->contains_node(edge.second)) {
                    this->add_node(edge.second);
                }

                this->add_edge(edge.first, edge.second);
            }

        }

        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        PartiallyDirectedImpl(const std::vector<std::string>& nodes,
                              const ArcStringVector& arcs,
                              const EdgeStringVector& edges) : BaseClass<Derived>(nodes),
                                                               ArcGraph<Derived, BaseClass>(nodes),
                                                               EdgeGraph<Derived, BaseClass>() {
                              
            for (auto& arc : arcs) {
                if (!this->contains_node(arc.first)) throw pybind11::index_error(
                    "Node \"" + arc.first + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");
            
                if (!this->contains_node(arc.second)) throw pybind11::index_error(
                    "Node \"" + arc.second + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");

                this->add_arc(arc.first, arc.second);
            }

            for (auto& edge : edges) {
                if (!this->contains_node(edge.first)) throw pybind11::index_error(
                    "Node \"" + edge.first + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");
            
                if (!this->contains_node(edge.second)) throw pybind11::index_error(
                    "Node \"" + edge.second + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");

                this->add_edge(edge.first, edge.second);
            }
        }

        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        PartiallyDirectedImpl(Graph<Undirected>&& g);
        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        PartiallyDirectedImpl(Graph<Directed>&& g);

        // /////////////////////////////////////
        // ConditionalGraphBase constructors
        // /////////////////////////////////////
        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalGraphBase, BaseClass<D>, int> = 0>
        PartiallyDirectedImpl(const std::vector<std::string>& nodes,
                              const std::vector<std::string>& interface_nodes) 
                                    : BaseClass<Derived>(nodes, interface_nodes),
                                      ArcGraph<Derived, BaseClass>(this->all_nodes()),
                                      EdgeGraph<Derived, BaseClass>() {}

        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalGraphBase, BaseClass<D>, int> = 0>
        PartiallyDirectedImpl(const std::vector<std::string>& nodes,
                              const std::vector<std::string>& interface_nodes,
                              const ArcStringVector& arcs,
                              const EdgeStringVector& edges) 
                                    : BaseClass<Derived>(nodes, interface_nodes),
                                      ArcGraph<Derived, BaseClass>(this->all_nodes()),
                                      EdgeGraph<Derived, BaseClass>() {

            for (auto& arc : arcs) {
                this->add_arc(arc.first, arc.second);
            }

            for (auto& edge : edges) {
                this->add_edge(edge.first, edge.second);
            }
        }

        template<typename V>
        void direct(const V& source, const V& target) {
            auto s = this->check_index(source);
            auto t = this->check_index(target);

            check_can_exist_arc(*this, s, t);
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
            return this->has_edge_unsafe(source, target) || 
                   this->has_arc_unsafe(source, target) || 
                   this->has_arc_unsafe(target, source);
        }

        GraphClass<DirectedAcyclic> to_dag() const;
        GraphClass<DirectedAcyclic> to_approximate_dag() const;
        
        py::tuple __getstate__() const {
            return graph::__getstate__(static_cast<const Derived&>(*this));
        }

        static Derived __setstate__(py::tuple& t) {
            return graph::__setstate__<Derived>(t);
        }

        static Derived __setstate__(py::tuple&& t) {
            return graph::__setstate__<Derived>(t);
        }

        void save(const std::string& name) const {
            save_graph(static_cast<const Derived&>(*this), name);
        }

        ConditionalGraph<PartiallyDirected> conditional_graph(const std::vector<std::string>& nodes,
                                                              const std::vector<std::string>& interface_nodes) const;

        ConditionalGraph<PartiallyDirected> conditional_graph() const;
        Graph<PartiallyDirected> unconditional_graph() const;
    };

    template<>
    class Graph<PartiallyDirected> : public PartiallyDirectedImpl<PartiallyDirectedGraph,
                                                                  GraphBase> {
    public:
        using PartiallyDirectedImpl<PartiallyDirectedGraph, GraphBase>::PartiallyDirectedImpl;

        static PartiallyDirectedGraph CompleteUndirected(const std::vector<std::string>& nodes);
    };

    template<>
    class ConditionalGraph<PartiallyDirected> : public PartiallyDirectedImpl<ConditionalPartiallyDirectedGraph,
                                                                             ConditionalGraphBase> {
    public:
        using PartiallyDirectedImpl<ConditionalPartiallyDirectedGraph, ConditionalGraphBase>::PartiallyDirectedImpl;
        
        static ConditionalPartiallyDirectedGraph CompleteUndirected(const std::vector<std::string>& nodes,
                                                                    const std::vector<std::string>& interface_nodes);

        void direct_interface_edges();
    };

    template<typename Derived, template<typename> typename BaseClass>
    class UndirectedImpl : public BaseClass<Derived>,
                           public EdgeGraph<Derived, BaseClass> {
    public:
        UndirectedImpl() = default;

        // /////////////////////////////////////
        // GraphBase constructors
        // /////////////////////////////////////
        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        UndirectedImpl(const std::vector<std::string>& nodes) : BaseClass<Derived>(nodes),
                                                                EdgeGraph<Derived, BaseClass>() {}
        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        UndirectedImpl(const EdgeStringVector& edges) : BaseClass<Derived>(), 
                                                        EdgeGraph<Derived, BaseClass>() {
            for (auto& edge : edges) {
                if (!this->contains_node(edge.first)) {
                    this->add_node(edge.first);
                }

                if (!this->contains_node(edge.second)) {
                    this->add_node(edge.second);
                }

                this->add_edge(edge.first, edge.second);
            }
        }
        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        UndirectedImpl(const std::vector<std::string>& nodes, const EdgeStringVector& edges) 
                    : BaseClass<Derived>(nodes), 
                      EdgeGraph<Derived, BaseClass>() {

            for (auto& edge : edges) {
                if (!this->contains_node(edge.first)) throw pybind11::index_error(
                    "Node \"" + edge.first + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");
            
                if (!this->contains_node(edge.second)) throw pybind11::index_error(
                    "Node \"" + edge.second + "\" in edge (" + edge.first + ", " + edge.second + ") not present in the graph.");

                this->add_edge(edge.first, edge.second);
            }   
        }

        // /////////////////////////////////////
        // ConditionalGraphBase constructors
        // /////////////////////////////////////
        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalGraphBase, BaseClass<D>, int> = 0>
        UndirectedImpl(const std::vector<std::string>& nodes,
                       const std::vector<std::string>& interface_nodes) 
                                    : BaseClass<Derived>(nodes, interface_nodes),
                                      EdgeGraph<Derived, BaseClass>() {}

        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalGraphBase, BaseClass<D>, int> = 0>
        UndirectedImpl(const std::vector<std::string>& nodes,
                       const std::vector<std::string>& interface_nodes,
                       const EdgeStringVector& edges) 
                                    : BaseClass<Derived>(nodes, interface_nodes),
                                      EdgeGraph<Derived, BaseClass>() {
            for (auto& edge : edges) {
                this->add_edge(edge.first, edge.second);
            }
        }


        template<typename V>
        bool has_path(const V& source, const V& target) const {
            auto s = this->check_index(source);
            auto t = this->check_index(target);
            return has_path_unsafe(s, t);
        }

        bool has_path_unsafe(int source, int target) const;

        py::tuple __getstate__() const {
            return graph::__getstate__(static_cast<const Derived&>(*this));
        }

        static Derived __setstate__(py::tuple& t) {
            return graph::__setstate__<Derived>(t);
        }

        static Derived __setstate__(py::tuple&& t) {
            return graph::__setstate__<Derived>(t);
        }

        void save(const std::string& name) const {
            save_graph(static_cast<const Derived&>(*this), name);
        }

        ConditionalGraph<Undirected> conditional_graph(const std::vector<std::string>& nodes,
                                                       const std::vector<std::string>& interface_nodes) const;
        ConditionalGraph<Undirected> conditional_graph() const;
        Graph<Undirected> unconditional_graph() const;
    };

    template<>
    class Graph<Undirected> : public UndirectedImpl<UndirectedGraph, GraphBase> {
    public:
        using UndirectedImpl<UndirectedGraph, GraphBase>::UndirectedImpl;

        static UndirectedGraph Complete(const std::vector<std::string>& nodes);
    };

    template<>
    class ConditionalGraph<Undirected> : public UndirectedImpl<ConditionalUndirectedGraph, ConditionalGraphBase> {
    public:
        using UndirectedImpl<ConditionalUndirectedGraph, ConditionalGraphBase>::UndirectedImpl;

        static ConditionalUndirectedGraph Complete(const std::vector<std::string>& nodes,
                                                   const std::vector<std::string>& interface_nodes);
    };

    
    template<typename Derived, template<typename> typename BaseClass>
    class DirectedImpl : public BaseClass<Derived>,
                         public ArcGraph<Derived, BaseClass> {
    public:
        DirectedImpl() = default;

        // /////////////////////////////////////
        // GraphBase constructors
        // /////////////////////////////////////
        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        DirectedImpl(const std::vector<std::string>& nodes) : GraphBase<DirectedGraph>(nodes), 
                                                              ArcGraph<DirectedGraph, GraphBase>(nodes) {}

        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        DirectedImpl(const ArcStringVector& arcs) : GraphBase<DirectedGraph>(),
                                                    ArcGraph<DirectedGraph, GraphBase>() {
            for (auto& arc : arcs) {
                if (!this->contains_node(arc.first)) {
                    this->add_node(arc.first);
                }

                if (!this->contains_node(arc.second)) {
                    this->add_node(arc.second);
                }

                this->add_arc(arc.first, arc.second);
            }
        }

        template<typename D = Derived, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int> = 0>
        DirectedImpl(const std::vector<std::string>& nodes, 
                     const ArcStringVector& arcs) : GraphBase<DirectedGraph>(nodes),
                                                    ArcGraph<DirectedGraph, GraphBase>(nodes) {
            for (auto& arc : arcs) {
                if (!this->contains_node(arc.first)) throw pybind11::index_error(
                    "Node \"" + arc.first + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");
            
                if (!this->contains_node(arc.second)) throw pybind11::index_error(
                    "Node \"" + arc.second + "\" in arc (" + arc.first + ", " + arc.second + ") not present in the graph.");

                this->add_arc(arc.first, arc.second);
            }
        }

        // /////////////////////////////////////
        // ConditionalGraphBase constructors
        // /////////////////////////////////////
        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalGraphBase, BaseClass<D>, int> = 0>
        DirectedImpl(const std::vector<std::string>& nodes,
                     const std::vector<std::string>& interface_nodes) 
                                    : BaseClass<Derived>(nodes, interface_nodes),
                                      ArcGraph<Derived, BaseClass>(this->all_nodes()) {}

        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalGraphBase, BaseClass<D>, int> = 0>
        DirectedImpl(const std::vector<std::string>& nodes,
                     const std::vector<std::string>& interface_nodes,
                     const ArcStringVector& arcs) 
                                    : BaseClass<Derived>(nodes, interface_nodes),
                                      ArcGraph<Derived, BaseClass>(this->all_nodes()) {
            for (auto& arc : arcs) {
                this->add_arc(arc.first, arc.second);
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
            return graph::__getstate__(static_cast<const Derived&>(*this));
        }

        static Derived __setstate__(py::tuple& t) {
            return graph::__setstate__<Derived>(t);
        }

        static Derived __setstate__(py::tuple&& t) {
            return graph::__setstate__<Derived>(t);
        }

        void save(const std::string& name) const {
            save_graph(static_cast<const Derived&>(*this), name);
        }

        ConditionalGraph<Directed> conditional_graph(const std::vector<std::string>& nodes,
                                             const std::vector<std::string>& interface_nodes) const;
        ConditionalGraph<Directed> conditional_graph() const;
        Graph<Directed> unconditional_graph() const;
    };

    template<>
    class Graph<Directed> : public DirectedImpl<DirectedGraph, GraphBase> {
    public:
        using DirectedImpl<DirectedGraph, GraphBase>::DirectedImpl;
    };

    template<>
    class ConditionalGraph<Directed> : public DirectedImpl<ConditionalDirectedGraph, ConditionalGraphBase> {
    public:
        using DirectedImpl<ConditionalDirectedGraph, ConditionalGraphBase>::DirectedImpl;
    };

    template<typename Derived, typename BaseClass>
    class DagImpl : public BaseClass {
    public:
        template<GraphType Type>
        using GraphClass = typename GraphTraits<Derived>::template GraphClass<Type>;

        DagImpl() = default;

        // /////////////////////////////////////
        // GraphBase constructors
        // /////////////////////////////////////
        template<typename B = BaseClass, std::enable_if_t<std::is_same_v<DirectedGraph, B>, int> = 0>
        DagImpl(const std::vector<std::string>& nodes) : BaseClass(nodes) {}
        template<typename B = BaseClass, std::enable_if_t<std::is_same_v<DirectedGraph, B>, int> = 0>
        DagImpl(const ArcStringVector& arcs) : BaseClass(arcs) {
            topological_sort();
        }
        template<typename B = BaseClass, std::enable_if_t<std::is_same_v<DirectedGraph, B>, int> = 0>
        DagImpl(const std::vector<std::string>& nodes, const ArcStringVector& arcs) : BaseClass(nodes, arcs) {
            topological_sort();
        }

        // /////////////////////////////////////
        // ConditionalGraphBase constructors
        // /////////////////////////////////////
        template<typename B = BaseClass, std::enable_if_t<std::is_same_v<ConditionalDirectedGraph, B>, int> = 0>
        DagImpl(const std::vector<std::string>& nodes,
                const std::vector<std::string>& interface_nodes) : BaseClass(nodes, interface_nodes) {}
        template<typename B = BaseClass, std::enable_if_t<std::is_same_v<ConditionalDirectedGraph, B>, int> = 0>
        DagImpl(const std::vector<std::string>& nodes,
                const std::vector<std::string>& interface_nodes,
                const ArcStringVector& arcs) : BaseClass(nodes, interface_nodes, arcs) {
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
            auto s = this->check_index(source);
            auto t = this->check_index(target);

            if (!can_add_arc_unsafe(s, t)) {
                throw std::runtime_error("Arc " + this->name(s) + " -> " + this->name(t) + " addition would break acyclity.");
            }

            BaseClass::add_arc_unsafe(s, t);
        }

        template<typename V>
        void flip_arc(const V& source, const V& target) {
            auto s = this->check_index(source);
            auto t = this->check_index(target);

            if (!can_flip_arc_unsafe(s, t)) {
                throw std::runtime_error("Arc " + this->name(s) + " -> " + this->name(t) + " flip would break acyclity.");
            }

            BaseClass::flip_arc_unsafe(s, t);
        }

        GraphClass<PartiallyDirected> to_pdag() const;

        bool is_dag() const { 
            try {
                topological_sort();
                return true;
            } catch (std::invalid_argument&) {
                return false;
            }
        }

        py::tuple __getstate__() const {
            return graph::__getstate__(static_cast<const Derived&>(*this));
        }

        static Derived __setstate__(py::tuple& t) {
            return graph::__setstate__<Derived>(t);
        }

        static Derived __setstate__(py::tuple&& t) {
            return graph::__setstate__<Derived>(t);
        }

        void save(const std::string& name) const {
            save_graph(static_cast<const Derived&>(*this), name);
        }

        ConditionalGraph<DirectedAcyclic> conditional_graph(const std::vector<std::string>& nodes,
                                                            const std::vector<std::string>& interface_nodes) const;
        ConditionalGraph<DirectedAcyclic> conditional_graph() const;
        Graph<DirectedAcyclic> unconditional_graph() const;
    };

    template<>
    class Graph<DirectedAcyclic> : public DagImpl<Dag, DirectedGraph> {
    public:
        using DagImpl<Dag, DirectedGraph>::DagImpl;
    };

    template<>
    class ConditionalGraph<DirectedAcyclic> : public DagImpl<ConditionalDag, ConditionalDirectedGraph> {
    public:
        using DagImpl<ConditionalDag, ConditionalDirectedGraph>::DagImpl;
    };

    template<typename Derived, template<typename> typename BaseClass>
    template<typename D, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int>>
    PartiallyDirectedImpl<Derived, BaseClass>::PartiallyDirectedImpl(Graph<Undirected>&& g) 
                                                : GraphBase<PartiallyDirectedGraph>(std::move(g)),
                                                  ArcGraph<PartiallyDirectedGraph, BaseClass>(this->nodes()), 
                                                  EdgeGraph<PartiallyDirectedGraph, BaseClass>(std::move(g)) {}

    template<typename Derived, template<typename> typename BaseClass>
    template<typename D, util::enable_if_template_instantation_t<GraphBase, BaseClass<D>, int>>
    PartiallyDirectedImpl<Derived, BaseClass>::PartiallyDirectedImpl(Graph<Directed>&& g)
                                                : GraphBase<PartiallyDirectedGraph>(std::move(g)),
                                                  ArcGraph<PartiallyDirectedGraph, GraphBase>(std::move(g)),
                                                  EdgeGraph<PartiallyDirectedGraph, GraphBase>() {}

    template<typename Derived, template<typename> typename BaseClass>
    void PartiallyDirectedImpl<Derived, BaseClass>::direct_unsafe(int source, int target) {
        if (this->has_edge_unsafe(source, target)) {
            this->remove_edge_unsafe(source, target);
            this->add_arc_unsafe(source, target);
        } else if (this->has_arc_unsafe(target, source)) {
            this->add_arc_unsafe(source, target);
        }
    }

    template<typename Derived, template<typename> typename BaseClass>
    void PartiallyDirectedImpl<Derived, BaseClass>::undirect_unsafe(int source, int target) {
        if (this->has_arc_unsafe(source, target))
            this->remove_arc_unsafe(source, target);

        if (!this->has_arc_unsafe(target, source))
            this->add_edge_unsafe(source, target);
    }

    template<typename Derived, template<typename> typename BaseClass>
    bool pdag2dag_adjacent_node(const PartiallyDirectedImpl<Derived, BaseClass>& g,
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


    template<typename Derived, template<typename> typename BaseClass>
    typename PartiallyDirectedImpl<Derived, BaseClass>::template GraphClass<DirectedAcyclic> 
    PartiallyDirectedImpl<Derived, BaseClass>::to_dag() const {
        // PDAG-TO-DAG by D.Dor, M.Tarsi (1992). A simple algorithm to construct a consistent extension of a partially oriented graph.
        GraphClass<DirectedAcyclic> directed;

        if constexpr (util::is_template_instantation_v<GraphBase, BaseClass<Derived>>)
            directed = GraphClass<DirectedAcyclic>(this->nodes());
        else if constexpr (util::is_template_instantation_v<ConditionalGraphBase, BaseClass<Derived>>)
            directed = GraphClass<DirectedAcyclic>(this->nodes(), this->interface_nodes());
        else
            static_assert(util::always_false<Derived>, "Wrong BaseClass for PartiallyDirectedImpl");
        
        for (const auto& arc : this->arcs()) {
            directed.add_arc_unsafe(directed.index(arc.first), 
                                    directed.index(arc.second));
        }

        ArcStringVector interface_edges;
        if constexpr (util::is_template_instantation_v<ConditionalGraphBase, BaseClass<Derived>>) {
            interface_edges = this->derived().interface_edges();
            for (const auto& arc : interface_edges) {
                directed.add_arc_unsafe(directed.index(arc.first),
                                        directed.index(arc.second));
            }
        }

        if (!directed.is_dag()) {
            throw std::invalid_argument("PDAG contains directed cycles.");
        }
        
        if (this->num_edges() > 0) {
            PartiallyDirectedImpl copy(*this);

            // Remove compelled arcs, as they are already included.
            if constexpr (util::is_template_instantation_v<ConditionalGraphBase, BaseClass<Derived>>) {
                for (const auto& arc : interface_edges) {
                    copy.remove_edge_unsafe(copy.index(arc.first),
                                            copy.index(arc.second));
                }
            }

            while (copy.num_edges() > 0) {
                bool ok = false;
                for (auto x : copy.leaves()) {
                    auto x_neighbors = copy.neighbor_indices(x);
                    auto x_parents = copy.parent_indices(x);

                    if (pdag2dag_adjacent_node(copy, x_neighbors, x_parents)) {
                        for (auto neighbor : x_neighbors) {
                            // Use names because pdag could have removed/invalidated nodes.
                            directed.add_arc(copy.name(neighbor), copy.name(x));
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

    template<typename Derived, template<typename> typename BaseClass>
    typename PartiallyDirectedImpl<Derived, BaseClass>::template GraphClass<DirectedAcyclic> 
    PartiallyDirectedImpl<Derived, BaseClass>::to_approximate_dag() const {
        GraphClass<DirectedAcyclic> directed;

        if constexpr (util::is_template_instantation_v<GraphBase, BaseClass<Derived>>)
            directed = GraphClass<DirectedAcyclic>(this->nodes());
        else if constexpr (util::is_template_instantation_v<ConditionalGraphBase, BaseClass<Derived>>)
            directed = GraphClass<DirectedAcyclic>(this->nodes(), this->interface_nodes());
        else
            static_assert(util::always_false<Derived>, "Wrong BaseClass for PartiallyDirectedImpl");
        
        for (const auto& arc : this->arcs()) {
            directed.add_arc_unsafe(directed.index(arc.first), 
                                    directed.index(arc.second));
        }

        if constexpr (util::is_template_instantation_v<ConditionalGraphBase, BaseClass<Derived>>) {
            auto interface_edges = this->derived().interface_edges();
            for (const auto& arc : interface_edges) {
                directed.add_arc_unsafe(directed.index(arc.first),
                                        directed.index(arc.second));
            }
        }
        
        std::vector<int> incoming_arcs;
        incoming_arcs.reserve(this->num_raw_nodes());

        for (const auto& raw_node : this->raw_nodes()) {
            if (raw_node.is_valid()) {
                incoming_arcs.push_back(raw_node.parents().size());
            } else {
                incoming_arcs.push_back(std::numeric_limits<int>::max());
            }
        }

        // Create a pseudo topological sort.
        std::vector<std::string> top_sort;
        dynamic_bitset in_top_sort(static_cast<size_t>(this->num_raw_nodes()));
        in_top_sort.reset(0, this->num_raw_nodes());
        for (auto free : this->free_indices()) {
            in_top_sort.set(free);
        }

        if constexpr (is_conditional_graph_v<Derived>)
            top_sort.reserve(this->num_total_nodes());
        else
            top_sort.reserve(this->num_nodes());

        std::vector<int> stack{this->roots().begin(), this->roots().end()};

        size_t expected_num_nodes;
        if constexpr (is_conditional_graph_v<Derived>)
            expected_num_nodes = static_cast<size_t>(this->num_total_nodes());
        else
            expected_num_nodes = static_cast<size_t>(this->num_nodes());


        while (top_sort.size() != expected_num_nodes) {
            // Possible cycle found. This would have not happened in a DAG.
            // Find the next node among the children of the already explored nodes.
            if (stack.empty()) {
                auto min_cardinality = std::numeric_limits<int>::max();
                auto min_cardinality_index = std::numeric_limits<int>::max();
                for (const auto& explored : top_sort) {
                    for (auto ch : directed.children_set(explored)) {
                        const auto& ch_name = directed.name(ch);
                        auto ch_this_index = this->index(ch_name);
                        if (!in_top_sort[ch_this_index] && directed.num_parents(ch) < min_cardinality) {
                            min_cardinality = directed.num_parents(ch);
                            min_cardinality_index = ch_this_index;
                        }
                    }
                }

                if (min_cardinality_index == std::numeric_limits<int>::max()) {
                    // Find the node without less parents
                    for (int i = 0; i < this->num_raw_nodes(); ++i) {
                        if (!in_top_sort[i] && directed.num_parents(this->name(i)) < min_cardinality) {
                            min_cardinality = directed.num_parents(this->name(i));
                            min_cardinality_index = i;
                        }
                    }
                }

                stack.push_back(min_cardinality_index);
            }

            auto idx = stack.back();
            stack.pop_back();

            top_sort.push_back(this->name(idx));
            in_top_sort.set(idx);

            for (const auto& children : this->children_set(idx)) {
                --incoming_arcs[children];

                if (in_top_sort[children]) {
                    const auto& idx_name = this->name(idx);
                    const auto& children_name = this->name(children);
                    directed.flip_arc_unsafe(directed.index(idx_name), directed.index(children_name));
                } else if (incoming_arcs[children] == 0) {
                    stack.push_back(children);
                }
            }
        }

        // directed is DAG now, with topological sort equal to top_sort
        if (this->num_edges() > 0) {
            std::unordered_map<std::string, int> top_sort_index;
            for (int i = 0, end = top_sort.size(); i < end; ++i) {
                top_sort_index.insert({top_sort[i], i});
            }

            for (const auto& edge : this->edges()) {
                auto first_top_sort_idx = top_sort_index.at(edge.first);
                auto second_top_sort_idx = top_sort_index.at(edge.second);
                if (first_top_sort_idx < second_top_sort_idx) {
                    directed.add_arc_unsafe(directed.index(edge.first), directed.index(edge.second));
                } else {
                    directed.add_arc_unsafe(directed.index(edge.second), directed.index(edge.first));
                }
            }
        }

        return directed;
    }


    template<typename Derived, template<typename> typename BaseClass>
    inline ConditionalGraph<PartiallyDirected>
    PartiallyDirectedImpl<Derived, BaseClass>::conditional_graph(const std::vector<std::string>& nodes,
                                                                 const std::vector<std::string>& interface_nodes) const {
        return to_conditional_graph(static_cast<const Derived&>(*this), nodes, interface_nodes);
    }

    template<typename Derived, template<typename> typename BaseClass>
    inline ConditionalGraph<PartiallyDirected>
    PartiallyDirectedImpl<Derived, BaseClass>::conditional_graph() const {
        if constexpr (is_conditional_graph_v<Derived>) {
            return static_cast<const Derived&>(*this);
        } else {
            std::vector<std::string> v;
            return to_conditional_graph(static_cast<const Derived&>(*this), this->nodes(), v);
        }
    }

    template<typename Derived, template<typename> typename BaseClass>
    inline Graph<PartiallyDirected>
    PartiallyDirectedImpl<Derived, BaseClass>::unconditional_graph() const {
        return to_unconditional_graph(static_cast<const Derived&>(*this));
    }

    template<typename Derived, template<typename> typename BaseClass>
    bool UndirectedImpl<Derived, BaseClass>::has_path_unsafe(int source, int target) const {
        if (this->has_edge_unsafe(source, target)) {
            return true;
        } else {
            dynamic_bitset in_stack(static_cast<size_t>(this->num_raw_nodes()));
            in_stack.reset(0, this->num_raw_nodes());

            for (auto free : this->free_indices()) {
                in_stack.set(free);
            }

            in_stack.set(source);

            const auto& neighbors = this->neighbor_set(source);
            std::vector<int> stack {neighbors.begin(), neighbors.end()};

            for (auto neighbor : stack) {
                in_stack.set(neighbor);
            }

            while(!stack.empty()) {
                auto v = stack.back();
                stack.pop_back();

                const auto& neighbors = this->neighbor_set(v);

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

    template<typename Derived, template<typename> typename BaseClass>
    inline ConditionalGraph<Undirected>
    UndirectedImpl<Derived, BaseClass>::conditional_graph(const std::vector<std::string>& nodes,
                                                          const std::vector<std::string>& interface_nodes) const {
        return to_conditional_graph(static_cast<const Derived&>(*this), nodes, interface_nodes);
    }

    template<typename Derived, template<typename> typename BaseClass>
    inline ConditionalGraph<Undirected>
    UndirectedImpl<Derived, BaseClass>::conditional_graph() const {
        if constexpr (is_conditional_graph_v<Derived>) {
            return static_cast<const Derived&>(*this);
        } else {
            std::vector<std::string> v;
            return to_conditional_graph(static_cast<const Derived&>(*this), this->nodes(), v);
        }
    }

    template<typename Derived, template<typename> typename BaseClass>
    inline Graph<Undirected>
    UndirectedImpl<Derived, BaseClass>::unconditional_graph() const {
        return to_unconditional_graph(static_cast<const Derived&>(*this));
    }

    template<typename Derived, template<typename> typename BaseClass>
    bool DirectedImpl<Derived, BaseClass>::has_path_unsafe(int source, int target) const {
        if (this->has_arc_unsafe(source, target))
            return true;
        else {
            dynamic_bitset in_stack(static_cast<size_t>(this->num_raw_nodes()));
            in_stack.reset(0, this->num_raw_nodes());

            for (auto free : this->free_indices()) {
                in_stack.set(free);
            }

            in_stack.set(source);

            const auto& children = this->children_set(source);
            std::vector<int> stack {children.begin(), children.end()};

            for (auto children : stack) {
                in_stack.set(children);
            }

            while(!stack.empty()) {
                auto v = stack.back();
                stack.pop_back();

                const auto& children = this->children_set(v);

                if (children.find(target) != children.end())
                    return true;

                for (auto ch : children) {
                    if (!in_stack[ch]) {
                        stack.push_back(ch);
                        in_stack.set(ch);
                    }
                }
            }

            return false;
        }
    }

    // Checks if there is a path between source and target without taking into account the possible arc source -> target.
    template<typename Derived, template<typename> typename BaseClass>
    bool DirectedImpl<Derived, BaseClass>::has_path_unsafe_no_direct_arc(int source, int target) const {
        dynamic_bitset in_stack(static_cast<size_t>(this->num_raw_nodes()));
        in_stack.reset(0, this->num_raw_nodes());

        for (auto free : this->free_indices()) {
            in_stack.set(free);
        }

        in_stack.set(source);

        const auto& children = this->children_set(source);
        std::vector<int> stack;
        for (auto ch : children) {
            if (ch != target) {
                stack.push_back(ch);
                in_stack.set(ch);
            }
        }

        while(!stack.empty()) {
            auto v = stack.back();
            stack.pop_back();

            const auto& children = this->children_set(v);

            if (children.find(target) != children.end())
                return true;

            for (auto ch : children) {
                if (!in_stack[ch]) {
                    stack.push_back(ch);
                    in_stack.set(ch);
                }
            }
        }

        return false;
    }

    template<typename Derived, template<typename> typename BaseClass>
    inline ConditionalGraph<Directed>
    DirectedImpl<Derived, BaseClass>::conditional_graph(const std::vector<std::string>& nodes,
                                                        const std::vector<std::string>& interface_nodes) const {
        return to_conditional_graph(static_cast<const Derived&>(*this), nodes, interface_nodes);
    }

    template<typename Derived, template<typename> typename BaseClass>
    inline ConditionalGraph<Directed>
    DirectedImpl<Derived, BaseClass>::conditional_graph() const {
        if constexpr (is_conditional_graph_v<Derived>) {
            return static_cast<const Derived&>(*this);
        } else {
            std::vector<std::string> v;
            return to_conditional_graph(static_cast<const Derived&>(*this), this->nodes(), v);
        }
    }

    template<typename Derived, template<typename> typename BaseClass>
    inline Graph<Directed>
    DirectedImpl<Derived, BaseClass>::unconditional_graph() const {
        return to_unconditional_graph(static_cast<const Derived&>(*this));
    }

    template<typename Derived, typename BaseClass>
    std::vector<std::string> DagImpl<Derived, BaseClass>::topological_sort() const {
        std::vector<int> incoming_edges;
        incoming_edges.reserve(this->num_raw_nodes());

        for (const auto& raw_node : this->raw_nodes()) {
            if (raw_node.is_valid()) {
                incoming_edges.push_back(raw_node.parents().size());
            } else {
                incoming_edges.push_back(-1);
            }
        }

        std::vector<std::string> top_sort;
        if constexpr (is_conditional_graph_v<Derived>)
            top_sort.reserve(this->num_total_nodes());
        else
            top_sort.reserve(this->num_nodes());

        std::vector<int> stack{this->roots().begin(), this->roots().end()};

        while (!stack.empty()) {
            auto idx = stack.back();
            stack.pop_back();

            top_sort.push_back(this->name(idx));
            
            for (const auto& children : this->children_set(idx)) {
                --incoming_edges[children];
                if (incoming_edges[children] == 0) {
                    stack.push_back(children);
                }
            }
        }
        
        for (auto in : incoming_edges) {
            if (in > 0) {
                throw std::invalid_argument("Graph must be a DAG to obtain a topological sort.");
            }
        }

        return top_sort;
    }

    template<typename Derived, typename BaseClass>
    bool DagImpl<Derived, BaseClass>::can_add_arc_unsafe(int source, int target) const {
        if (can_exist_arc(*this, source, target) && 
                (this->num_parents_unsafe(source) == 0 || 
                 this->num_children_unsafe(target) == 0 || 
                 !this->has_path_unsafe(target, source))) {
            return true;
        }
        return false;
    }
    
    template<typename Derived, typename BaseClass>
    bool DagImpl<Derived, BaseClass>::can_flip_arc_unsafe(int source, int target) const {
        if (!can_exist_arc(*this, target, source))
            return false;

        if (this->has_arc_unsafe(source, target)) {
            if (this->num_parents_unsafe(target) == 1 || this->num_children_unsafe(source) == 1)
                return true;

            bool thereis_path = this->has_path_unsafe_no_direct_arc(source, target);
            if (thereis_path) {
                return false;
            } else {
                return true;
            }
        } else {
            if (this->num_parents_unsafe(target) == 0 || this->num_children_unsafe(source) == 0)
                return true;

            bool thereis_path = this->has_path_unsafe(source, target);
            if (thereis_path) {
                return false;
            } else {
                return true;
            }
        }
    }

    template<typename Derived, typename BaseClass>
    std::vector<Arc> sort_arcs(const DagImpl<Derived, BaseClass>& g) {
        auto top_sort = g.topological_sort();
        std::vector<int> top_rank(g.num_raw_nodes());

        for (size_t i = 0; i < top_sort.size(); ++i) {
            top_rank[g.index(top_sort[i])] = i;
        }

        std::vector<Arc> res;
        res.reserve(g.num_arcs());

        int included_arcs = 0;
        for (size_t i = 0; i < top_sort.size() && included_arcs < g.num_arcs(); ++i) {
            auto p = g.parent_indices(top_sort[i]);

            std::sort(p.begin(), p.end(), [&top_rank](int a, int b) {
                return top_rank[a] > top_rank[b];
            });

            auto y_index = g.index(top_sort[i]);
            for(auto x_index : p) {
                
                // Do not include interface_arcs, as they are known to be compelled.
                if constexpr (is_conditional_graph_v<Derived>) {
                    if (!g.is_interface(x_index))
                        res.push_back({x_index, y_index});
                } else {
                    res.push_back({x_index, y_index});
                }
            }
        }

        return res;
    }

    template<typename Derived, typename BaseClass>
    typename DagImpl<Derived, BaseClass>::template GraphClass<PartiallyDirected> 
    DagImpl<Derived, BaseClass>::to_pdag() const {
        // DAG-to-PDAG by Chickering (2002). Learning Equivalence Classes of Bayesian-Network Structures.
        // See Also: Chickering (1995). A Transformational Characterization of Equivalent Bayesian Network Structures
        std::vector<Arc> sorted_arcs = sort_arcs(*this);
        GraphClass<PartiallyDirected> pdag;

        if constexpr (std::is_same_v<DirectedGraph, BaseClass>)
            pdag = GraphClass<PartiallyDirected>(this->nodes());
        else if constexpr (std::is_same_v<ConditionalDirectedGraph, BaseClass>) {
            pdag = GraphClass<PartiallyDirected>(this->nodes(), this->interface_nodes());

            for (const auto& compelled : this->interface_arcs()) {
                pdag.add_arc(compelled.first, compelled.second);
            }
        }
        else
            static_assert(util::always_false<Derived>, "Wrong BaseClass for DagImpl");

        for (size_t i = 0; i < sorted_arcs.size() && (pdag.num_arcs() + pdag.num_edges()) < this->num_arcs(); ++i) {
            auto x = sorted_arcs[i].first;
            auto y = sorted_arcs[i].second;
            // Use name because Dag could have removed/invalidated nodes.
            const auto& x_name = this->name(x);
            const auto& y_name = this->name(y);
            if (!pdag.has_arc(x_name, y_name) && !pdag.has_edge(x_name, y_name)) {
                bool done = false;
                for (auto w : pdag.parent_set(x_name)) {
                    const auto& w_name = pdag.name(w);
                    if (!this->has_arc(w_name, y_name)) {
                        for (auto z : this->parent_set(y_name)) {
                            const auto& z_name = this->name(z);
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

                    for (auto z : this->parent_set(y)) {
                        if (z != x && !this->has_arc(z, x)) {
                            compelled = true;
                            break;
                        }
                    }

                    if (compelled) {
                        for (auto z : this->parent_set(y)) {
                            const auto& z_name = this->name(z);
                            pdag.add_arc(z_name, y_name);
                        }
                    } else {
                        for (auto z : this->parent_set(y)) {
                            const auto& z_name = this->name(z);
                            pdag.add_edge(z_name, y_name);
                        }
                    }
                }
            }
        }

        return pdag;
    }

    template<typename Derived, typename BaseClass>
    inline ConditionalGraph<DirectedAcyclic>
    DagImpl<Derived, BaseClass>::conditional_graph(const std::vector<std::string>& nodes,
                                                   const std::vector<std::string>& interface_nodes) const {
        return to_conditional_graph(static_cast<const Derived&>(*this), nodes, interface_nodes);
    }

    template<typename Derived, typename BaseClass>
    inline ConditionalGraph<DirectedAcyclic>
    DagImpl<Derived, BaseClass>::conditional_graph() const {
        if constexpr (is_conditional_graph_v<Derived>) {
            return static_cast<const Derived&>(*this);
        } else {
            std::vector<std::string> v;
            return to_conditional_graph(static_cast<const Derived&>(*this), this->nodes(), v);
        }
    }

    template<typename Derived, typename BaseClass>
    inline Graph<DirectedAcyclic>
    DagImpl<Derived, BaseClass>::unconditional_graph() const {
        return to_unconditional_graph(static_cast<const Derived&>(*this));
    }

}

#endif //PYBNESIAN_GRAPH_GENERIC_GRAPH_HPP
