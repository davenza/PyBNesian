#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <graph/generic_graph.hpp>

namespace py = pybind11;

using graph::ConditionalDirectedGraph, graph::ConditionalDag, graph::ConditionalUndirectedGraph,
    graph::ConditionalPartiallyDirectedGraph;
using graph::DirectedGraph, graph::Dag, graph::UndirectedGraph, graph::PartiallyDirectedGraph;

template <typename CppClass, typename PyClass>
void add_graphbase_methods(PyClass& c, std::string class_name) {
    c.def("num_nodes", &CppClass::num_nodes, R"doc(
Gets the number of nodes.

:returns: Number of nodes.
)doc")
        .def("nodes",
             &CppClass::nodes,
             R"doc(
Gets the nodes of the graph.

:returns: Nodes of the graph.
)doc",
             py::return_value_policy::reference_internal)
        .def("indices", &CppClass::indices, R"doc(
Gets all the indices in the graph.

:returns: A dictionary with the index of each node.
)doc")
        .def("collapsed_indices", &CppClass::collapsed_indices, R"doc(
Gets the collapsed indices in the graph.

:returns: A dictionary with the collapsed index of each node.
)doc")
        .def("contains_node", &CppClass::contains_node, py::arg("node"), R"doc(
Tests whether the node is in the graph or not.

:param node: Name of the node.
:returns: True if the graph contains the node, False otherwise.
)doc")
        .def("add_node", &CppClass::add_node, py::arg("node"), R"doc(
Adds a node to the graph and returns its index.

:param node: Name of the new node.
:returns: Index of the new node.
)doc")
        .def("name", &CppClass::name, py::arg("index"), R"doc(
Gets the name of a node from its index.

:param index: Index of the node.
:returns: Name of the node.
)doc")
        .def("collapsed_name", &CppClass::collapsed_name, py::arg("collapsed_index"), R"doc(
Gets the name of a node from its collapsed index.

:param collapsed_index: Collapsed index of the node.
:returns: Name of the node.
)doc")
        .def("index", &CppClass::index, py::arg("node"), R"doc(
Gets the index of a node from its name.

:param node: Name of the node.
:returns: Index of the node.
)doc")
        .def("collapsed_index", &CppClass::collapsed_index, py::arg("node"), R"doc(
Gets the collapsed index of a node from its name.

:param node: Name of the node.
:returns: Collapsed index of the node.
)doc")
        .def("index_from_collapsed", &CppClass::index_from_collapsed, py::arg("collapsed_index"), R"doc(
Gets the index of a node from its collapsed index.

:param collapsed_index: Collapsed index of the node.
:returns: Index of the node.
)doc")
        .def("collapsed_from_index", &CppClass::collapsed_from_index, py::arg("index"), R"doc(
Gets the collapsed index of a node from its index.

:param index: Index of the node.
:returns: Collapsed index of the node.
)doc")
        .def("is_valid", &CppClass::is_valid, py::arg("index"), R"doc(
Checks whether a index is a valid index (the node is not removed). All the valid indices are always returned by
``indices()``.

:param index: Index of the node.
:returns: True if the index is valid, False otherwise.
)doc");

    {
        py::options options;
        options.disable_function_signatures();

        std::stringstream signature;
        signature << "remove_node(self: pybnesian." << class_name << ", node: int or str) -> None";

        auto doc = signature.str() + R"doc(

Removes a node.

:param node: A node name or index.
)doc";

        c.def(
             "remove_node", [](CppClass& self, int index) { self.remove_node(index); }, py::arg("node"))
            .def(
                "remove_node",
                [](CppClass& self, const std::string& name) { self.remove_node(name); },
                py::arg("node"),
                doc.c_str());
    }
}

template <typename CppClass, typename PyClass>
void add_conditionalgraphbase_methods(PyClass& c, std::string class_name) {
    c.def("num_nodes", &CppClass::num_nodes, R"doc(
Gets the number of nodes.

:returns: Number of nodes.
)doc")
        .def("num_interface_nodes", &CppClass::num_interface_nodes, R"doc(
Gets the number of interface nodes.

:returns: Number of interface nodes.
)doc")
        .def("num_joint_nodes", &CppClass::num_joint_nodes, R"doc(
Gets the number of joint nodes. That is, ``num_nodes() + num_interface_nodes()``

:returns: Number of joint nodes.
)doc")
        .def("nodes",
             &CppClass::nodes,
             R"doc(
Gets the nodes of the graph.

:returns: Nodes of the graph.
)doc",
             py::return_value_policy::reference_internal)
        .def("interface_nodes", &CppClass::interface_nodes, R"doc(
Gets the interface nodes of the graph.

:returns: Interface nodes of the graph.
)doc")
        .def("joint_nodes", &CppClass::joint_nodes, R"doc(
Gets the joint set of nodes of the graph.

:returns: Joint set of nodes of the graph.
)doc")
        .def("indices", &CppClass::indices, R"doc(
Gets all the indices for the nodes in the graph.

:returns: A dictionary with the index of each node.
)doc")
        .def("collapsed_indices", &CppClass::collapsed_indices, R"doc(
Gets all the collapsed indices for the nodes in the graph.

:returns: A dictionary with the collapsed index of each node.
)doc")
        .def("interface_collapsed_indices", &CppClass::interface_collapsed_indices, R"doc(
Gets all the interface collapsed indices for the interface nodes in the graph.

:returns: A dictionary with the interface collapsed index of each interface node.
)doc")
        .def("joint_collapsed_indices", &CppClass::joint_collapsed_indices, R"doc(
Gets all the joint collapsed indices for the joint set of nodes in the graph.

:returns: A dictionary with the joint collapsed index of each joint node.
)doc")
        .def("contains_node", &CppClass::contains_node, py::arg("node"), R"doc(
Tests whether the node is in the graph or not.

:param node: Name of the node.
:returns: True if the graph contains the node, False otherwise.
)doc")
        .def("contains_interface_node", &CppClass::contains_interface_node, py::arg("node"), R"doc(
Tests whether the interface node is in the graph or not.

:param node: Name of the node.
:returns: True if the graph contains the interface node, False otherwise.
)doc")
        .def("contains_joint_node", &CppClass::contains_joint_node, py::arg("node"), R"doc(
Tests whether the node is in the joint set of nodes or not.

:param node: Name of the node.
:returns: True if the node is in the joint set of nodes, False otherwise.
)doc")
        .def("add_node", &CppClass::add_node, py::arg("node"), R"doc(
Adds a node to the graph and returns its index.

:param node: Name of the new node.
:returns: Index of the new node.
)doc")
        .def("add_interface_node", &CppClass::add_interface_node, py::arg("node"), R"doc(
Adds an interface node to the graph and returns its index.

:param node: Name of the new interface node.
:returns: Index of the new interface node.
)doc")
        .def("name", &CppClass::name, py::arg("index"), R"doc(
Gets the name of a node from its index.

:param index: Index of the node.
:returns: Name of the node.
)doc")
        .def("collapsed_name", &CppClass::collapsed_name, py::arg("collapsed_index"), R"doc(
Gets the name of a node from its collapsed index.

:param collapsed_index: Collapsed index of the node.
:returns: Name of the node.
)doc")
        .def("interface_collapsed_name", &CppClass::interface_collapsed_name, py::arg("collapsed_index"), R"doc(
Gets the name of an interface node from its collapsed index.

:param collapsed_index: Collapsed index of the interface node.
:returns: Name of the interface node.
)doc")
        .def("joint_collapsed_name", &CppClass::joint_collapsed_name, py::arg("collapsed_index"), R"doc(
Gets the name of a node from its joint collapsed index.

:param collapsed_index: Joint collapsed index of the node.
:returns: Name of the node.
)doc")
        .def("index", &CppClass::index, py::arg("node"), R"doc(
Gets the index of a node from its name.

:param node: Name of the node.
:returns: Index of the node.
)doc")
        .def("collapsed_index", &CppClass::collapsed_index, py::arg("node"), R"doc(
Gets the collapsed index of a node from its name.

:param node: Name of the node.
:returns: Collapsed index of the node.
)doc")
        .def("interface_collapsed_index", &CppClass::interface_collapsed_index, py::arg("node"), R"doc(
Gets the interface collapsed index of an interface node from its name.

:param node: Name of the interface node.
:returns: Interface collapsed index of the interface node.
)doc")
        .def("joint_collapsed_index", &CppClass::joint_collapsed_index, py::arg("node"), R"doc(
Gets the joint collapsed index of a node from its name.

:param node: Name of the node.
:returns: Joint collapsed index of the node.
)doc")
        .def("index_from_collapsed", &CppClass::index_from_collapsed, py::arg("collapsed_index"), R"doc(
Gets the index of a node from its collapsed index.

:param collapsed_index: Collapsed index of the node.
:returns: Index of the node.
)doc")
        .def("index_from_interface_collapsed",
             &CppClass::index_from_interface_collapsed,
             py::arg("collapsed_index"),
             R"doc(
Gets the index of a node from the interface collapsed index.

:param collapsed_index: Interface collapsed index of the node.
:returns: Index of the node.
)doc")
        .def("index_from_joint_collapsed",
             &CppClass::index_from_joint_collapsed,
             py::arg("collapsed_index"),
             R"doc(
Gets the index of a node from the joint collapsed index.

:param collapsed_index: Joint collapsed index of the node.
:returns: Index of the node.
)doc")
        .def("collapsed_from_index", &CppClass::collapsed_from_index, py::arg("index"), R"doc(
Gets the collapsed index of a node from its index.

:param index: Index of the node.
:returns: Collapsed index of the node.
)doc")
        .def("interface_collapsed_from_index", &CppClass::interface_collapsed_from_index, py::arg("index"), R"doc(
Gets the interface collapsed index of a node from its index.

:param index: Index of the node.
:returns: Interface collapsed index of the node.
)doc")
        .def("joint_collapsed_from_index", &CppClass::joint_collapsed_from_index, py::arg("index"), R"doc(
Gets the joint collapsed index of a node from its index.

:param index: Index of the node.
:returns: Joint collapsed index of the node.
)doc")
        .def("is_valid", &CppClass::is_valid, py::arg("index"), R"doc(
Checks whether a index is a valid index (the node is not removed). All the valid indices are always returned by
``indices()``.

:param index: Index of the node.
:returns: True if the index is valid, False otherwise.
)doc");

    {
        py::options options;
        options.disable_function_signatures();

        auto remove_node_doc = "remove_node(self: pybnesian." + class_name + R"doc(, node: int or str) -> None

Removes a node.

:param node: A node name or index.
)doc";

        auto remove_interface_node_doc = "remove_interface_node(self: pybnesian." + class_name +
                                         R"doc(, node: int or str) -> None

Removes an interface node.

:param node: A node name or index.
)doc";

        auto is_interface_doc = "is_interface(self: pybnesian." + class_name + R"doc(, node: int or str) -> bool

Checks whether the ``node`` is an interface node.

:param node: A node name or index.
:returns: True if ``node`` is interface node, False, otherwise.
)doc";

        auto set_interface_doc = "set_interface(self: pybnesian." + class_name + R"doc(, node: int or str) -> None

Converts a normal node into an interface node.

:param node: A node name or index.
)doc";

        auto set_node_doc = "set_node(self: pybnesian." + class_name + R"doc(, node: int or str) -> None

Converts an interface node into a normal node.

:param node: A node name or index.
)doc";

        c.def(
             "remove_node", [](CppClass& self, int index) { self.remove_node(index); }, py::arg("node"))
            .def(
                "remove_node",
                [](CppClass& self, const std::string& name) { self.remove_node(name); },
                py::arg("node"),
                remove_node_doc.c_str())
            .def(
                "remove_interface_node",
                [](CppClass& self, int index) { self.remove_interface_node(index); },
                py::arg("node"))
            .def(
                "remove_interface_node",
                [](CppClass& self, const std::string& name) { self.remove_interface_node(name); },
                py::arg("node"),
                remove_interface_node_doc.c_str())
            .def(
                "is_interface", [](CppClass& self, int index) { return self.is_interface(index); }, py::arg("node"))
            .def(
                "is_interface",
                [](CppClass& self, const std::string& name) { return self.is_interface(name); },
                py::arg("node"),
                is_interface_doc.c_str())
            .def(
                "set_interface", [](CppClass& self, int index) { self.is_interface(index); }, py::arg("node"))
            .def(
                "set_interface",
                [](CppClass& self, const std::string& name) { self.set_interface(name); },
                py::arg("node"),
                set_interface_doc.c_str())
            .def(
                "set_node", [](CppClass& self, int index) { self.set_node(index); }, py::arg("node"))
            .def(
                "set_node",
                [](CppClass& self, const std::string& name) { self.set_node(name); },
                py::arg("node"),
                set_node_doc.c_str());
    }
}

template <typename CppClass, typename PyClass>
void add_arcgraph_methods(PyClass& c, std::string class_name) {
    c.def("num_arcs", &CppClass::num_arcs, R"doc(
Gets the number of arcs.

:returns: Number of arcs.
)doc")
        .def("arcs", &CppClass::arcs, R"doc(
Gets the list of arcs.

:returns: A list of tuples (source, target) representing an arc source -> target.
)doc");

    std::string roots_doc;
    std::string leaves_doc;

    if constexpr (graph::is_unconditional_graph_v<CppClass>) {
        roots_doc = R"doc(
Gets the root nodes of the graph. A root node do not have parent nodes.

:returns: The set of root nodes.
)doc";

        leaves_doc = R"doc(
Gets the leaf nodes of the graph. A leaf node do not have children nodes.

:returns: The set of leaf nodes.
)doc";
    } else if constexpr (graph::is_conditional_graph_v<CppClass>) {
        roots_doc = R"doc(
Gets the root nodes of the graph. A root node do not have parent nodes.

This implementation do not include the interface nodes in the result. Also, do not take into account the
interface arcs. That is, if a node only have interface nodes as parents, it is considered a root. Thus, this returns
the same result as an unconditional graph without the interface nodes.

:returns: The set of root nodes.
)doc";

        leaves_doc = R"doc(
Gets the leaf nodes of the graph. A leaf node do not have children nodes.

This implementation do not include the interface nodes in the result. Thus, this returns the same result as
an unconditional graph without the interface nodes.

:returns: The set of leaf nodes.
)doc";

    } else {
        static_assert(util::always_false<CppClass>, "Wrong CppClass!");
    }

    c.def(
         "roots",
         [](CppClass& self) {
             std::unordered_set<std::string> roots;
             auto rr = self.roots();
             for (const auto& r : rr) {
                 roots.insert(self.name(r));
             }
             return roots;
         },
         roots_doc.c_str())
        .def(
            "leaves",
            [](CppClass& self) {
                std::unordered_set<std::string> leaves;
                for (const auto& lv : self.leaves()) {
                    leaves.insert(self.name(lv));
                }
                return leaves;
            },
            leaves_doc.c_str());

    std::string signature_type = "pybnesian." + class_name;
    std::string num_parents_doc = "num_parents(self: " + signature_type + R"doc(, node: int or str) -> int

Gets the number of parent nodes of a node.

:param node: A node name or index.
:returns: Number of parent nodes.
)doc";

    std::string num_children_doc = "num_children(self: " + signature_type + R"doc(, node: int or str) -> int

Gets the number of children nodes of a node.

:param node: A node name or index.
:returns: Number of children nodes.
)doc";

    std::string parents_doc = "parents(self: " + signature_type + R"doc(, node: int or str) -> List[str]

Gets the parent nodes of a node.

:param node: A node name or index.
:returns: Parent node names.
)doc";

    std::string children_doc = "children(self: " + signature_type + R"doc(, node: int or str) -> List[str]

Gets the children nodes of a node.

:param node: A node name or index.
:returns: Children node names.
)doc";

    std::string add_arc_doc = "add_arc(self: " + signature_type +
                              R"doc(, source: int or str, target: int or str) -> None

Adds an arc between the nodes ``source`` and ``target``. If the arc already exists, the graph is left unaffected.

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
)doc";

    std::string has_arc_doc = "has_arc(self: " + signature_type +
                              R"doc(, source: int or str, target: int or str) -> bool

Checks whether an arc between the nodes ``source`` and ``target`` exists. 

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
:returns: True if the arc exists, False otherwise.
)doc";

    std::string remove_arc_doc = "remove_arc(self: " + signature_type +
                                 R"doc(, source: int or str, target: int or str) -> None

Removes an arc between the nodes ``source`` and ``target``. If the arc do not exist, the graph is left unaffected.

``source`` and ``target`` can be the name or the index, but **the type of source and target must be the same**.

:param source: A node name or index.
:param target: A node name or index.
)doc";

    std::string flip_arc_doc = "flip_arc(self: " + signature_type +
                               R"doc(, source: int or str, target: int or str) -> None

Flips (reverses) an arc between the nodes ``source`` and ``target``. If the arc do not exist, the graph is left
unaffected.

``source`` and ``target`` can be the name or the index, but **the type of source and target must be the same**.

:param source: A node name or index.
:param target: A node name or index.
)doc";

    std::string is_root_doc;
    std::string is_leaf_doc = "is_leaf(self: " + signature_type + R"doc(, node: int or str) -> bool

Checks whether  ``node`` is a leaf node. A root node do not have children nodes.

:param node: A node name or index.
:returns: True if ``node`` is leaf, False otherwise.
)doc";

    if constexpr (graph::is_unconditional_graph_v<CppClass>) {
        is_root_doc = "is_root(self: " + signature_type + R"doc(, node: int or str) -> bool

Checks whether  ``node`` is a root node. A root node do not have parent nodes.

:param node: A node name or index.
:returns: True if ``node`` is root, False otherwise.
)doc";

    } else if constexpr (graph::is_conditional_graph_v<CppClass>) {
        is_root_doc = "is_root(self: " + signature_type + R"doc(, node: int or str) -> bool

Checks whether  ``node`` is a root node. A root node do not have parent nodes.

This implementation do not take into account the interface arcs. That is, if a node only have interface nodes as
parents, it is considered a root.

:param node: A node name or index.
:returns: True if ``node`` is root, False otherwise.
)doc";
    } else {
        static_assert(util::always_false<CppClass>, "Wrong CppClass!");
    }

    {
        py::options options;
        options.disable_function_signatures();

        c.def(
             "num_parents", [](CppClass& self, int index) { return self.num_parents(index); }, py::arg("node"))
            .def(
                "num_parents",
                [](CppClass& self, const std::string& name) { return self.num_parents(name); },
                py::arg("node"),
                num_parents_doc.c_str())
            .def(
                "num_children", [](CppClass& self, int index) { return self.num_parents(index); }, py::arg("node"))
            .def(
                "num_children",
                [](CppClass& self, const std::string& name) { return self.num_parents(name); },
                py::arg("node"),
                num_children_doc.c_str())
            .def(
                "parents", [](CppClass& self, int index) { return self.parents(index); }, py::arg("node"))
            .def(
                "parents",
                [](CppClass& self, const std::string& name) { return self.parents(name); },
                py::arg("node"),
                parents_doc.c_str())
            .def(
                "children", [](CppClass& self, int index) { return self.children(index); }, py::arg("node"))
            .def(
                "children",
                [](CppClass& self, const std::string& name) { return self.children(name); },
                py::arg("node"),
                children_doc.c_str())
            .def(
                "add_arc",
                [](CppClass& self, int source, int target) { self.add_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "add_arc",
                [](CppClass& self, const std::string& source, const std::string& target) {
                    self.add_arc(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                add_arc_doc.c_str())
            .def(
                "has_arc",
                [](CppClass& self, int source, int target) { return self.has_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "has_arc",
                [](CppClass& self, const std::string& source, const std::string& target) {
                    return self.has_arc(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                has_arc_doc.c_str())
            .def(
                "remove_arc",
                [](CppClass& self, int source, int target) { self.remove_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "remove_arc",
                [](CppClass& self, const std::string& source, const std::string& target) {
                    self.remove_arc(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                remove_arc_doc.c_str())
            .def(
                "flip_arc",
                [](CppClass& self, int source, int target) { self.flip_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "flip_arc",
                [](CppClass& self, const std::string& source, const std::string& target) {
                    self.flip_arc(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                flip_arc_doc.c_str())
            .def(
                "is_root", [](CppClass& self, int node) { return self.is_root(node); }, py::arg("node"))
            .def(
                "is_root",
                [](CppClass& self, const std::string& node) { return self.is_root(node); },
                py::arg("node"),
                is_root_doc.c_str())
            .def(
                "is_leaf", [](CppClass& self, int node) { return self.is_leaf(node); }, py::arg("node"))
            .def(
                "is_leaf",
                [](CppClass& self, const std::string& node) { return self.is_leaf(node); },
                py::arg("node"),
                is_leaf_doc.c_str());
    }

    if constexpr (graph::is_conditional_graph_v<CppClass>) {
        c.def("interface_arcs", &CppClass::template interface_arcs<>, R"doc(
Gets the arcs where the source node is an interface node.

:returns: arcs with an interface node as source node.
)doc");
    }
}

template <typename CppClass, typename PyClass>
void add_edgegraph_methods(PyClass& c, std::string class_name) {
    c.def("num_edges", &CppClass::num_edges, R"doc(
Gets the number of edges.

:returns: Number of edges.
)doc")
        .def("edges", &CppClass::edges, R"doc(
Gets the list of edges.

:returns: A list of tuples (n1, n2) representing an edge between n1 and n2.
)doc");

    std::string signature_type = "pybnesian." + class_name;
    std::string num_neighbors_doc = "num_neighbors(self: " + signature_type + R"doc(, node: int or str) -> int

Gets the number of neighbors (adjacent nodes by an edge) of a node.

:param node: A node name or index.
:returns: Number of neighbors.
)doc";

    std::string neighbors_doc = "neighbors(self: " + signature_type + R"doc(, node: int or str) -> List[str]

Gets the neighbors (adjacent nodes by an edge) of a node.

:param node: A node name or index.
:returns: Neighbor names.
)doc";

    std::string add_edge_doc = "add_edge(self: " + signature_type + R"doc(, n1: int or str, n2: int or str) -> None

Adds an edge between the nodes ``n1`` and ``n2``.

``n1`` and ``n2`` can be the name or the index, **but the type of n1 and n2 must be the same.**

:param n1: A node name or index.
:param n2: A node name or index.
)doc";

    std::string has_edge_doc = "has_edge(self: " + signature_type + R"doc(, n1: int or str, n2: int or str) -> bool

Checks whether an edge between the nodes ``n1`` and ``n2`` exists.

``n1`` and ``n2`` can be the name or the index, **but the type of n1 and n2 must be the same.**

:param n1: A node name or index.
:param n2: A node name or index.
:returns: True if the edge exists, False otherwise.
)doc";

    std::string remove_edge_doc =
        "remove_edge(self: " + signature_type + R"doc(, n1: int or str, n2: int or str) -> None

Removes an edge between the nodes ``n1`` and ``n2``.

``n1`` and ``n2`` can be the name or the index, but **the type of n1 and n2 must be the same**.

:param n1: A node name or index.
:param n2: A node name or index.
)doc";

    {
        py::options options;
        options.disable_function_signatures();

        c.def(
             "num_neighbors", [](CppClass& self, int index) { return self.num_neighbors(index); }, py::arg("node"))
            .def(
                "num_neighbors",
                [](CppClass& self, const std::string& name) { return self.num_neighbors(name); },
                py::arg("node"),
                num_neighbors_doc.c_str())
            .def(
                "neighbors", [](CppClass& self, int index) { return self.neighbors(index); }, py::arg("node"))
            .def(
                "neighbors",
                [](CppClass& self, const std::string& name) { return self.neighbors(name); },
                py::arg("node"),
                neighbors_doc.c_str())
            .def(
                "add_edge",
                [](CppClass& self, int source, int target) { self.add_edge(source, target); },
                py::arg("n1"),
                py::arg("n2"))
            .def(
                "add_edge",
                [](CppClass& self, const std::string& source, const std::string& target) {
                    self.add_edge(source, target);
                },
                py::arg("n1"),
                py::arg("n2"),
                add_edge_doc.c_str())
            .def(
                "has_edge",
                [](CppClass& self, int source, int target) { return self.has_edge(source, target); },
                py::arg("n1"),
                py::arg("n2"))
            .def(
                "has_edge",
                [](CppClass& self, const std::string& source, const std::string& target) {
                    return self.has_edge(source, target);
                },
                py::arg("n1"),
                py::arg("n2"),
                has_edge_doc.c_str())
            .def(
                "remove_edge",
                [](CppClass& self, int source, int target) { self.remove_edge(source, target); },
                py::arg("n1"),
                py::arg("n2"))
            .def(
                "remove_edge",
                [](CppClass& self, const std::string& source, const std::string& target) {
                    self.remove_edge(source, target);
                },
                py::arg("n1"),
                py::arg("n2"),
                remove_edge_doc.c_str());
    }

    if constexpr (graph::is_conditional_graph_v<CppClass>) {
        c.def("interface_edges", &CppClass::template interface_edges<>, R"doc(
Gets the edges where one of the nodes is an interface node.

:returns: edges as a list of tuples (inode, node), where ``inode`` is an interface node and ``node`` is a normal node.
)doc");
    }
}

template <typename CppClass, typename PyClass>
void add_pdag_methods(PyClass& c, std::string class_name) {
    c.def("to_dag", [](const CppClass& self) { return self.to_dag(); }, R"doc(
Gets a :class:`Dag` which belongs to the equivalence class of ``self``.

It implements the algorithm in [pdag2dag]_.

:returns: A :class:`Dag` which belongs to the equivalence class of ``self``.
:raises ValueError: If ``self`` do not have a valid DAG extension.
)doc");

    auto approximate_dag_doc = R"doc(
Gets a :class:`Dag` approximate extension of ``self``. This method can be useful
when :func:`)doc" + class_name +
                               R"doc(.to_dag` cannot return a valid extension.

The algorithm is based on generating a topological sort which tries to preserve a similar structure.

:returns: A :class:`Dag` approximate extension of ``self``.
)doc";

    c.def(
        "to_approximate_dag",
        [](const CppClass& self) { return self.to_approximate_dag(); },
        approximate_dag_doc.c_str());

    std::string signature_type = "pybnesian." + class_name;
    std::string direct_doc = "direct(self: " + signature_type + R"doc(, source: int or str, target: int or str) -> None

Transformation to create the arc ``source`` -> ``target`` when possible.

- If there is an edge ``source`` -- ``target``, it is transformed into an arc ``source`` -> ``target``.
- If there is an arc ``target`` -> ``source``, it is flipped into an arc ``source`` -> ``target``.
- Else, the graph is left unaffected.

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
)doc";

    std::string undirect_doc =
        "undirect(self: " + signature_type + R"doc(, source: int or str, target: int or str) -> None

Transformation to create the edge ``source`` -- ``target`` when possible.

- If there is not an arc ``target`` -> ``source``, converts the arc ``source`` -> ``target`` into an edge
  ``source`` -- ``target``. If there is not an arc ``source`` -> ``target``, it adds the edge ``source`` -- ``target``.
- Else, the graph is left unaffected

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
)doc";

    std::string has_connection_doc =
        "has_connection(self: " + signature_type + R"doc(, source: int or str, target: int or str) -> bool

Checks whether two nodes ``source`` and ``target`` are connected.

Two nodes ``source`` and ``target`` are connected if there is an edge ``source`` -- ``target``, or an arc
``source`` -> ``target`` or an arc ``target`` -> ``source``.

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
:returns: True if ``source`` and ``target`` are connected, False otherwise.
)doc";

    {
        py::options options;
        options.disable_function_signatures();
        c.def(
             "direct",
             [](CppClass& self, int source, int target) { self.direct(source, target); },
             py::arg("source"),
             py::arg("target"))
            .def(
                "direct",
                [](CppClass& self, const std::string& source, const std::string& target) {
                    self.direct(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                direct_doc.c_str())
            .def(
                "undirect",
                [](CppClass& self, int source, int target) { self.undirect(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "undirect",
                [](CppClass& self, const std::string& source, const std::string& target) {
                    self.undirect(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                undirect_doc.c_str())
            .def(
                "has_connection",
                [](CppClass& self, int source, int target) { return self.has_connection(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "has_connection",
                [](CppClass& self, const std::string& source, const std::string& target) {
                    return self.has_connection(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                has_connection_doc.c_str());
    }
}

template <typename CppClass, typename PyClass>
void add_directed_methods(PyClass& c, std::string class_name) {
    py::options options;
    options.disable_function_signatures();

    std::string signature_type = "pybnesian." + class_name;
    std::string has_path_doc = "has_path(self: " + signature_type + R"doc(, n1: int or str, n2: int or str) -> bool

Checks whether there is a directed path between nodes ``n1`` and ``n2``.

``n1``  and ``n2`` can be the name or the index, **but the type of n1 and n2 must be the same.**

:param n1: A node name or index.
:param n2: A node name or index.
:returns: True if there is an directed path between ``n1`` and ``n2``, False otherwise.
)doc";

    c.def(
         "has_path",
         [](CppClass& self, int source, int target) { return self.has_path(source, target); },
         py::arg("n1"),
         py::arg("n2"))
        .def(
            "has_path",
            [](CppClass& self, const std::string& source, const std::string& target) {
                return self.has_path(source, target);
            },
            py::arg("n1"),
            py::arg("n2"),
            has_path_doc.c_str());
}

template <typename CppClass, typename PyClass>
void add_undirected_methods(PyClass& c, std::string class_name) {
    py::options options;
    options.disable_function_signatures();

    std::string signature_type = "pybnesian." + class_name;
    std::string has_path_doc = "has_path(self: " + signature_type + R"doc(, n1: int or str, n2: int or str) -> bool

Checks whether there is an undirected path between nodes ``n1`` and ``n2``.

``n1``  and ``n2`` can be the name or the index, **but the type of n1  and n2 must be the same.**

:param n1: A node name or index.
:param n2: A node name or index.
:returns: True if there is an undirected path between ``n1`` and ``n2``, False otherwise.
)doc";

    c.def(
         "has_path",
         [](CppClass& self, int source, int target) { return self.has_path(source, target); },
         py::arg("n1"),
         py::arg("n2"))
        .def(
            "has_path",
            [](CppClass& self, const std::string& source, const std::string& target) {
                return self.has_path(source, target);
            },
            py::arg("n1"),
            py::arg("n2"),
            has_path_doc.c_str());
}

template <typename CppClass, typename PyClass>
void add_pickle_methods(PyClass& c) {
    c.def("save", &CppClass::save, py::arg("filename"), R"doc(
Saves the graph in a pickle file with the given name.

:param filename: File name of the saved graph.
)doc")
        .def(py::pickle([](const CppClass& self) { return self.__getstate__(); },
                        [](py::tuple t) { return CppClass::__setstate__(t); }));
}

template <typename CppClass, typename PyClass>
void add_to_conditional_methods(PyClass& c) {
    c.def("conditional_graph", py::overload_cast<>(&CppClass::conditional_graph, py::const_), R"doc(
Transforms the graph to a conditional graph. 

- If ``self`` is not conditional, it returns a conditional version of the graph with the same nodes and without 
  interface nodes.
- If ``self`` is conditional, it returns a copy of ``self``.

:returns: The conditional graph transformation of ``self``.
)doc")
        .def("conditional_graph",
             py::overload_cast<const std::vector<std::string>&, const std::vector<std::string>&>(
                 &CppClass::conditional_graph, py::const_),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             R"doc(
Transforms the graph to a conditional graph. 

- If ``self`` is not conditional, it returns a conditional version of the graph with the given nodes and interface
  nodes.
- If ``self`` is conditional, it returns the same graph type with the given nodes and interface nodes.

:param nodes: The nodes for the new conditional graph.
:param interface_nodes: The interface nodes for the new conditional graph.
:returns: The conditional graph transformation of ``self``.
)doc")
        .def("unconditional_graph", &CppClass::unconditional_graph, R"doc(
Transforms the graph to an unconditional graph. 

- If ``self`` is not conditional, it returns a copy of ``self``.
- If ``self`` is conditional, the interface nodes are included as nodes in the returned graph.

:returns: The unconditional graph transformation of ``self``.
)doc");
}

template <typename DirType, typename DagType, typename UndirType, typename PDAGType>
void pybindings_normal_graph(DirType& dg, DagType& dag, UndirType& ug, PDAGType& pdag) {
    dg.def(py::init<>(), R"doc(
Creates a :class:`DirectedGraph` without nodes or arcs.
)doc")
        .def(py::init<const std::vector<std::string>&>(), py::arg("nodes"), R"doc(
Creates a :class:`DirectedGraph` with the specified nodes and without arcs.

:param nodes: Nodes of the :class:`DirectedGraph`.
)doc")
        .def(py::init<const ArcStringVector&>(), py::arg("arcs"), R"doc(
Creates a :class:`DirectedGraph` with the specified arcs (the nodes are extracted from the arcs).

:param arcs: Arcs of the :class:`DirectedGraph`.
)doc")
        .def(py::init<const std::vector<std::string>&, const ArcStringVector&>(),
             py::arg("nodes"),
             py::arg("arcs"),
             R"doc(
Creates a :class:`DirectedGraph` with the specified nodes and arcs.

:param nodes: Nodes of the :class:`DirectedGraph`.
:param arcs: Arcs of the :class:`DirectedGraph`.
)doc");

    add_graphbase_methods<DirectedGraph>(dg, "DirectedGraph");
    add_arcgraph_methods<DirectedGraph>(dg, "DirectedGraph");
    add_directed_methods<DirectedGraph>(dg, "DirectedGraph");
    add_to_conditional_methods<DirectedGraph>(dg);
    add_pickle_methods<DirectedGraph>(dg);

    dag.def(py::init<>(), R"doc(
Creates a :class:`Dag` without nodes or arcs.
)doc")
        .def(py::init<const std::vector<std::string>&>(), py::arg("nodes"), R"doc(
Creates a :class:`Dag` with the specified nodes and without arcs.

:param nodes: Nodes of the :class:`Dag`.
)doc")
        .def(py::init<const ArcStringVector&>(), py::arg("arcs"), R"doc(
Creates a :class:`Dag` with the specified arcs (the nodes are extracted from the arcs).

:param arcs: Arcs of the :class:`Dag`.
)doc")
        .def(py::init<const std::vector<std::string>&, const ArcStringVector&>(),
             py::arg("nodes"),
             py::arg("arcs"),
             R"doc(
Creates a :class:`Dag` with the specified nodes and arcs.

:param nodes: Nodes of the :class:`Dag`.
:param arcs: Arcs of the :class:`Dag`.
)doc")
        .def("topological_sort", &Dag::topological_sort, R"doc(
Gets the topological sort of the DAG.

:returns: Topological sort as a list of nodes.
)doc")
        .def("to_pdag", &Dag::to_pdag, R"doc(
Gets the :class:`PartiallyDirectedGraph` (PDAG) that represents the equivalence class of this :class:`Dag`.

It implements the DAG-to-PDAG algorithm in [dag2pdag]_. See also [dag2pdag_extra]_.

:returns: A :class:`PartiallyDirectedGraph` that represents the equivalence class of this :class:`Dag`.
)doc");

    {
        py::options options;
        options.disable_function_signatures();

        dag.def(
               "can_add_arc",
               [](Dag& self, int source, int target) { return self.can_add_arc(source, target); },
               py::arg("source"),
               py::arg("target"))
            .def(
                "can_add_arc",
                [](Dag& self, const std::string& source, const std::string& target) {
                    return self.can_add_arc(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                R"doc(
can_add_arc(self: pybnesian.Dag, source: int or str, target: int or str) -> bool

Checks whether an arc between the nodes ``source`` and ``target`` can be added. That is, the arc is valid and do not
generate a cycle.

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
:returns: True if the arc can be added, False otherwise.
)doc")
            .def(
                "can_flip_arc",
                [](Dag& self, int source, int target) { return self.can_flip_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "can_flip_arc",
                [](Dag& self, const std::string& source, const std::string& target) {
                    return self.can_flip_arc(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                R"doc(
can_flip_arc(self: pybnesian.Dag, source: int or str, target: int or str) -> bool

Checks whether an arc between the nodes ``source`` and ``target`` can be flipped. That is, the flipped arc is valid and
do not generate a cycle. If the arc ``source`` -> ``target`` do not exist, it will return :func:`Dag.can_add_arc`.

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
:returns: True if the arc can be flipped, False otherwise.
)doc")
            .def(
                "add_arc",
                [](Dag& self, int source, int target) { self.add_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "add_arc",
                [](Dag& self, const std::string& source, const std::string& target) { self.add_arc(source, target); },
                py::arg("source"),
                py::arg("target"),
                R"doc(
add_arc(self: pybnesian.Dag, source: int or str, target: int or str) -> None

Adds an arc between the nodes ``source`` and ``target``. If the arc already exists, the graph is left unaffected.

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
)doc")
            .def(
                "flip_arc",
                [](Dag& self, int source, int target) { self.flip_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "flip_arc",
                [](Dag& self, const std::string& source, const std::string& target) { self.flip_arc(source, target); },
                py::arg("source"),
                py::arg("target"),
                R"doc(
flip_arc(self: pybnesian.Dag, source: int or str, target: int or str) -> None

Flips (reverses) an arc between the nodes ``source`` and ``target``. If the arc do not exist, the graph is left
unaffected.

``source`` and ``target`` can be the name or the index, but **the type of source and target must be the same**.

:param source: A node name or index.
:param target: A node name or index.
)doc");
    }

    add_to_conditional_methods<Dag>(dag);
    add_pickle_methods<Dag>(dag);

    ug.def(py::init<>(), R"doc(
Creates a :class:`UndirectedGraph` without nodes or edges.
)doc")
        .def(py::init<const std::vector<std::string>&>(), py::arg("nodes"), R"doc(
Creates an :class:`UndirectedGraph` with the specified nodes and without edges.

:param nodes: Nodes of the :class:`UndirectedGraph`.
)doc")
        .def(py::init<const EdgeStringVector&>(), py::arg("edges"), R"doc(
Creates an :class:`UndirectedGraph` with the specified edges (the nodes are extracted from the edges).

:param edges: Edges of the :class:`UndirectedGraph`.
)doc")
        .def(py::init<const std::vector<std::string>&, const EdgeStringVector&>(),
             py::arg("nodes"),
             py::arg("edges"),
             R"doc(
Creates an :class:`UndirectedGraph` with the specified nodes and edges.

:param nodes: Nodes of the :class:`UndirectedGraph`.
:param edges: Edges of the :class:`UndirectedGraph`.
)doc")
        .def_static("Complete", &UndirectedGraph::Complete, py::arg("nodes"), R"doc(
Creates a complete :class:`UndirectedGraph` with the specified nodes.

:param nodes: Nodes of the :class:`UndirectedGraph`.
)doc");

    add_graphbase_methods<UndirectedGraph>(ug, "UndirectedGraph");
    add_edgegraph_methods<UndirectedGraph>(ug, "UndirectedGraph");
    add_undirected_methods<UndirectedGraph>(ug, "UndirectedGraph");
    add_to_conditional_methods<UndirectedGraph>(ug);
    add_pickle_methods<UndirectedGraph>(ug);

    pdag.def(py::init<>(), R"doc(
Creates a :class:`PartiallyDirectedGraph` without nodes, arcs and edges.
)doc")
        .def(py::init<const std::vector<std::string>&>(), py::arg("nodes"), R"doc(
Creates a :class:`PartiallyDirectedGraph` with the specified nodes and without arcs and edges.

:param nodes: Nodes of the :class:`PartiallyDirectedGraph`.
)doc")
        .def(py::init<const ArcStringVector&, const EdgeStringVector&>(), py::arg("arcs"), py::arg("edges"), R"doc(
Creates a :class:`PartiallyDirectedGraph` with the specified arcs and edges (the nodes are extracted from the arcs and
edges).

:param arcs: Arcs of the :class:`PartiallyDirectedGraph`.
:param edges: Edges of the :class:`PartiallyDirectedGraph`.
)doc")
        .def(py::init<const std::vector<std::string>&, const ArcStringVector&, const EdgeStringVector&>(),
             py::arg("nodes"),
             py::arg("arcs"),
             py::arg("edges"),
             R"doc(
Creates a :class:`PartiallyDirectedGraph` with the specified nodes and arcs.

:param nodes: Nodes of the :class:`PartiallyDirectedGraph`.
:param arcs: Arcs of the :class:`PartiallyDirectedGraph`.
:param edges: Edges of the :class:`PartiallyDirectedGraph`.
)doc")
        .def_static("CompleteUndirected", &PartiallyDirectedGraph::CompleteUndirected, py::arg("nodes"), R"doc(
Creates a :class:`PartiallyDirectedGraph` that is a complete undirected graph.

:param nodes: Nodes of the :class:`PartiallyDirectedGraph`.
)doc");

    add_graphbase_methods<PartiallyDirectedGraph>(pdag, "PartiallyDirectedGraph");
    add_arcgraph_methods<PartiallyDirectedGraph>(pdag, "PartiallyDirectedGraph");
    add_edgegraph_methods<PartiallyDirectedGraph>(pdag, "PartiallyDirectedGraph");
    add_pdag_methods<PartiallyDirectedGraph>(pdag, "PartiallyDirectedGraph");
    add_to_conditional_methods<PartiallyDirectedGraph>(pdag);
    add_pickle_methods<PartiallyDirectedGraph>(pdag);
}

template <typename DirType, typename DagType, typename UndirType, typename PDAGType>
void pybindings_conditional_graph(DirType& cdg, DagType& cdag, UndirType& cug, PDAGType& cpdag) {
    cdg.def(py::init<>(), R"doc(
Creates a :class:`ConditionalDirectedGraph` without nodes or arcs.
)doc")
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             R"doc(
Creates a :class:`ConditionalDirectedGraph` with the specified nodes, interface_nodes and without arcs.

:param nodes: Nodes of the :class:`ConditionalDirectedGraph`.
:param interface_nodes: Interface nodes of the :class:`ConditionalDirectedGraph`.
)doc")
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const ArcStringVector&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             R"doc(
Creates a :class:`ConditionalDirectedGraph` with the specified nodes and arcs.

:param nodes: Nodes of the :class:`ConditionalDirectedGraph`.
:param interface_nodes: Interface nodes of the :class:`ConditionalDirectedGraph`.
:param arcs: Arcs of the :class:`ConditionalDirectedGraph`.
)doc");

    add_conditionalgraphbase_methods<ConditionalDirectedGraph>(cdg, "ConditionalDirectedGraph");
    add_arcgraph_methods<ConditionalDirectedGraph>(cdg, "ConditionalDirectedGraph");
    add_directed_methods<ConditionalDirectedGraph>(cdg, "ConditionalDirectedGraph");
    add_to_conditional_methods<ConditionalDirectedGraph>(cdg);
    add_pickle_methods<ConditionalDirectedGraph>(cdg);

    cdag.def(py::init<>(), R"doc(
Creates a :class:`ConditionalDag` without nodes or arcs.
)doc")
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             R"doc(
Creates a :class:`ConditionalDag` with the specified nodes, interface_nodes and without arcs.

:param nodes: Nodes of the :class:`ConditionalDag`.
:param interface_nodes: Interface nodes of the :class:`ConditionalDag`.
)doc")
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const ArcStringVector&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             R"doc(
Creates a :class:`ConditionalDag` with the specified nodes, interface_nodes and arcs.

:param nodes: Nodes of the :class:`ConditionalDag`.
:param interface_nodes: Interface nod"es of the :class:`ConditionalDag`.
:param arcs: Arcs of the :class:`ConditionalDag`.
)doc")
        .def("topological_sort", &ConditionalDag::topological_sort, R"doc(
Gets the topological sort of the conditional DAG. This topological sort does not include the interface nodes, since they
are known to be always roots (they can be included at the very beginning of the topological sort).

:returns: Topological sort as a list of nodes.
)doc")
        .def("to_pdag", &ConditionalDag::to_pdag, R"doc(
Gets the :class:`ConditionalPartiallyDirectedGraph` (PDAG) that represents the equivalence class of this
:class:`ConditionalDag`.

It implements the DAG-to-PDAG algorithm in [dag2pdag]_. See also [dag2pdag_extra]_.

:returns: A :class:`ConditionalPartiallyDirectedGraph` that represents the equivalence class of this
          :class:`ConditionalDag`.
)doc");

    {
        py::options options;
        options.disable_function_signatures();

        cdag.def(
                "can_add_arc",
                [](ConditionalDag& self, int source, int target) { return self.can_add_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "can_add_arc",
                [](ConditionalDag& self, const std::string& source, const std::string& target) {
                    return self.can_add_arc(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                R"doc(
can_add_arc(self: pybnesian.ConditionalDag, source: int or str, target: int or str) -> bool

Checks whether an arc between the nodes ``source`` and ``target`` can be added. That is, the arc is valid and do not
generate a cycle or connects two interface nodes.

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
:returns: True if the arc can be added, False otherwise.
)doc")
            .def(
                "can_flip_arc",
                [](ConditionalDag& self, int source, int target) { return self.can_flip_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "can_flip_arc",
                [](ConditionalDag& self, const std::string& source, const std::string& target) {
                    return self.can_flip_arc(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                R"doc(
can_flip_arc(self: pybnesian.ConditionalDag, source: int or str, target: int or str) -> bool

Checks whether an arc between the nodes ``source`` and ``target`` can be flipped. That is, the flipped arc is valid and
do not generate a cycle. If the arc ``source`` -> ``target`` do not exist, it will return
:func:`ConditionalDag.can_add_arc`.

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
:returns: True if the arc can be flipped, False otherwise.
)doc")
            .def(
                "add_arc",
                [](ConditionalDag& self, int source, int target) { self.add_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "add_arc",
                [](ConditionalDag& self, const std::string& source, const std::string& target) {
                    self.add_arc(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                R"doc(
add_arc(self: pybnesian.ConditionalDag, source: int or str, target: int or str) -> None

Adds an arc between the nodes ``source`` and ``target``. If the arc already exists, the graph is left unaffected.

``source`` and ``target`` can be the name or the index, **but the type of source and target must be the same.**

:param source: A node name or index.
:param target: A node name or index.
)doc")
            .def(
                "flip_arc",
                [](ConditionalDag& self, int source, int target) { self.flip_arc(source, target); },
                py::arg("source"),
                py::arg("target"))
            .def(
                "flip_arc",
                [](ConditionalDag& self, const std::string& source, const std::string& target) {
                    self.flip_arc(source, target);
                },
                py::arg("source"),
                py::arg("target"),
                R"doc(
flip_arc(self: pybnesian.ConditionalDag, source: int or str, target: int or str) -> None

Flips (reverses) an arc between the nodes ``source`` and ``target``. If the arc do not exist, the graph is left
unaffected.

``source`` and ``target`` can be the name or the index, but **the type of source and target must be the same**.

:param source: A node name or index.
:param target: A node name or index.
)doc");
    }

    add_to_conditional_methods<ConditionalDag>(cdag);
    add_pickle_methods<ConditionalDag>(cdag);

    cug.def(py::init<>(), R"doc(
Creates a :class:`ConditionalUndirectedGraph` without nodes or edges.
)doc")
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             R"doc(
Creates a :class:`ConditionalUndirectedGraph` with the specified nodes, interface_nodes and without edges.

:param nodes: Nodes of the :class:`ConditionalUndirectedGraph`.
:param interface_nodes: Interface nodes of the :class:`ConditionalUndirectedGraph`.
)doc")
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const EdgeStringVector&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("edges"),
             R"doc(
Creates a :class:`ConditionalUndirectedGraph` with the specified nodes, interface_nodes and edges.

:param nodes: Nodes of the :class:`ConditionalUndirectedGraph`.
:param interface_nodes: Interface nodes of the :class:`ConditionalUndirectedGraph`.
:param edges: Edges of the :class:`ConditionalUndirectedGraph`.
)doc")
        .def_static(
            "Complete", &ConditionalUndirectedGraph::Complete, py::arg("nodes"), py::arg("interface_nodes"), R"doc(
Creates a complete :class:`ConditionalUndirectedGraph` with the specified nodes. A complete conditional undirected graph
connects every pair of nodes with an edge, except for pairs of interface nodes.

:param nodes: Nodes of the :class:`ConditionalUndirectedGraph`.
:param interface_nodes: Interface nodes of the :class:`ConditionalUndirectedGraph`.
)doc");

    add_conditionalgraphbase_methods<ConditionalUndirectedGraph>(cug, "ConditionalUndirectedGraph");
    add_edgegraph_methods<ConditionalUndirectedGraph>(cug, "ConditionalUndirectedGraph");
    add_undirected_methods<ConditionalUndirectedGraph>(cug, "ConditionalUndirectedGraph");
    add_to_conditional_methods<ConditionalUndirectedGraph>(cug);
    add_pickle_methods<ConditionalUndirectedGraph>(cug);

    cpdag
        .def(py::init<>(), R"doc(
Creates a :class:`ConditionalPartiallyDirectedGraph` without nodes or arcs.
)doc")
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             R"doc(
Creates a :class:`ConditionalPartiallyDirectedGraph` with the specified nodes, interface_nodes and without edges.

:param nodes: Nodes of the :class:`ConditionalPartiallyDirectedGraph`.
:param interface_nodes: Interface nodes of the :class:`ConditionalPartiallyDirectedGraph`.
)doc")
        .def(py::init<const std::vector<std::string>&,
                      const std::vector<std::string>&,
                      const ArcStringVector&,
                      const EdgeStringVector&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             py::arg("edges"),
             R"doc(
Creates a :class:`ConditionalPartiallyDirectedGraph` with the specified nodes and arcs.

:param nodes: Nodes of the :class:`ConditionalPartiallyDirectedGraph`.
:param interface_nodes: Interface nodes of the :class:`ConditionalPartiallyDirectedGraph`.
:param arcs: Arcs of the :class:`ConditionalPartiallyDirectedGraph`.
:param edges: Edges of the :class:`ConditionalPartiallyDirectedGraph`.
)doc")
        .def_static("CompleteUndirected",
                    &ConditionalPartiallyDirectedGraph::CompleteUndirected,
                    py::arg("nodes"),
                    py::arg("interface_nodes"),
                    R"doc(
Creates a :class:`ConditionalPartiallyDirectedGraph` that is a complete undirected graph. A complete conditional
undirected graph connects every pair of nodes with an edge, except for pairs of interface nodes.

:param nodes: Nodes of the :class:`ConditionalPartiallyDirectedGraph`.
:param interface_nodes: Interface nodes of the :class:`ConditionalPartiallyDirectedGraph`.
)doc");

    add_conditionalgraphbase_methods<ConditionalPartiallyDirectedGraph>(cpdag, "ConditionalPartiallyDirectedGraph");
    add_arcgraph_methods<ConditionalPartiallyDirectedGraph>(cpdag, "ConditionalPartiallyDirectedGraph");
    add_edgegraph_methods<ConditionalPartiallyDirectedGraph>(cpdag, "ConditionalPartiallyDirectedGraph");
    add_pdag_methods<ConditionalPartiallyDirectedGraph>(cpdag, "ConditionalPartiallyDirectedGraph");
    add_to_conditional_methods<ConditionalPartiallyDirectedGraph>(cpdag);
    add_pickle_methods<ConditionalPartiallyDirectedGraph>(cpdag);
}

void pybindings_graph(py::module& root) {
    py::class_<DirectedGraph> dg(root, "DirectedGraph", R"doc(
Directed graph that may contain cycles.
)doc");

    py::class_<Dag, DirectedGraph> dag(root,
                                       "Dag",
                                       py::multiple_inheritance(),
                                       R"doc(
Directed acyclic graph.
)doc");

    py::class_<UndirectedGraph> ug(root, "UndirectedGraph", R"doc(
Undirected graph.
)doc");

    py::class_<PartiallyDirectedGraph> pdag(root, "PartiallyDirectedGraph", R"doc(
Partially directed graph. This graph can have edges and arcs.
)doc");

    py::class_<ConditionalDirectedGraph> cdg(root, "ConditionalDirectedGraph", R"doc(
Conditional directed graph.
)doc");

    py::class_<ConditionalDag, ConditionalDirectedGraph> cdag(root, "ConditionalDag", py::multiple_inheritance(), R"doc(
Conditional directed acyclic graph.
)doc");

    py::class_<ConditionalUndirectedGraph> cug(root, "ConditionalUndirectedGraph", R"doc(
Conditional undirected graph.
)doc");

    py::class_<ConditionalPartiallyDirectedGraph> cpdag(root, "ConditionalPartiallyDirectedGraph", R"doc(
Conditional partially directed graph. This graph can have edges and arcs, except between pairs of interface nodes.
)doc");

    pybindings_normal_graph(dg, dag, ug, pdag);
    pybindings_conditional_graph(cdg, cdag, cug, cpdag);
}
