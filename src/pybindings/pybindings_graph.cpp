#include <pybind11/pybind11.h>

#include <graph/dag.hpp>
#include <graph/undirected.hpp>
#include <graph/pdag.hpp>

namespace py = pybind11;

using graph::DirectedGraph, graph::UndirectedGraph, graph::PartiallyDirectedGraph;


void pybindings_graph(py::module& root) {
    auto graph = root.def_submodule("graph","Graph submodule");

    py::class_<DirectedGraph>(graph, "DirectedGraph")
        .def(py::init<>())
        .def(py::init<const std::vector<std::string>&>())
        .def(py::init<const ArcVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcVector&>())
        .def("roots", [](DirectedGraph& self) {
            std::unordered_set<std::string> roots;
            for (const auto& r : self.roots()) {
                roots.insert(self.name(r));
            }
            return roots;
        })
        .def("leaves", [](DirectedGraph& self) {
            std::unordered_set<std::string> leaves;
            for (const auto& lv : self.leaves()) {
                leaves.insert(self.name(lv));
            }
            return leaves;
        })
        .def("num_nodes", &DirectedGraph::num_nodes)
        .def("num_arcs", &DirectedGraph::num_arcs)
        .def("num_parents", py::overload_cast<const std::string&>(&DirectedGraph::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&DirectedGraph::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&DirectedGraph::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&DirectedGraph::num_children, py::const_))
        .def("nodes", &DirectedGraph::nodes)
        .def("index", &DirectedGraph::index)
        .def("name", [](DirectedGraph& self, int idx) -> std::string {
            return self.name(idx);
        })
        .def("contains_node", &DirectedGraph::contains_node)
        .def("arcs", &DirectedGraph::arcs)
        .def("parents", py::overload_cast<const std::string&>(&DirectedGraph::parents, py::const_))
        .def("parents", py::overload_cast<int>(&DirectedGraph::parents, py::const_))
        .def("add_node", &DirectedGraph::add_node)
        .def("remove_node", py::overload_cast<const std::string&>(&DirectedGraph::remove_node))
        .def("remove_node", py::overload_cast<int>(&DirectedGraph::remove_node))
        .def("add_arc", py::overload_cast<const std::string&, const std::string&>(&DirectedGraph::add_arc))
        .def("add_arc", py::overload_cast<int, int>(&DirectedGraph::add_arc))
        .def("has_arc", py::overload_cast<const std::string&, const std::string&>(&DirectedGraph::has_arc, py::const_))
        .def("has_arc", py::overload_cast<int, int>(&DirectedGraph::has_arc, py::const_))
        .def("remove_arc", py::overload_cast<const std::string&, const std::string&>(&DirectedGraph::remove_arc))
        .def("remove_arc", py::overload_cast<int, int>(&DirectedGraph::remove_arc))
        .def("flip_arc", py::overload_cast<const std::string&, const std::string&>(&DirectedGraph::flip_arc))
        .def("flip_arc", py::overload_cast<int, int>(&DirectedGraph::flip_arc))
        .def("has_path", py::overload_cast<const std::string&, const std::string&>(&DirectedGraph::has_path, py::const_))
        .def("has_path", py::overload_cast<int, int>(&DirectedGraph::has_path, py::const_))
        .def("is_valid", &DirectedGraph::is_valid)
        .def("can_add_arc", py::overload_cast<const std::string&, const std::string&>(&DirectedGraph::can_add_arc, py::const_))
        .def("can_add_arc", py::overload_cast<int, int>(&DirectedGraph::can_add_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<const std::string&, const std::string&>(&DirectedGraph::can_flip_arc))
        .def("can_flip_arc", py::overload_cast<int, int>(&DirectedGraph::can_flip_arc));

    py::class_<UndirectedGraph>(graph, "UndirectedGraph")
        .def(py::init<>())
        .def(py::init<const std::vector<std::string>&>())
        .def(py::init<const EdgeVector&>())
        .def(py::init<const std::vector<std::string>&, const EdgeVector&>())
        .def_static("Complete", &UndirectedGraph::Complete)
        .def("num_nodes", &UndirectedGraph::num_nodes)
        .def("num_edges", &UndirectedGraph::num_edges)
        .def("num_neighbors", py::overload_cast<const std::string&>(&UndirectedGraph::num_neighbors, py::const_))
        .def("num_neighbors", py::overload_cast<int>(&UndirectedGraph::num_neighbors, py::const_))
        .def("nodes", &UndirectedGraph::nodes)
        .def("index", &UndirectedGraph::index)
        .def("name", [](UndirectedGraph& self, int idx) -> std::string {
            return self.name(idx);
        })
        .def("contains_node", &UndirectedGraph::contains_node)
        .def("edges", &UndirectedGraph::edges)
        .def("neighbors", py::overload_cast<const std::string&>(&UndirectedGraph::neighbors, py::const_))
        .def("neighbors", py::overload_cast<int>(&UndirectedGraph::neighbors, py::const_))
        .def("add_node", &UndirectedGraph::add_node)
        .def("remove_node", py::overload_cast<const std::string&>(&UndirectedGraph::remove_node))
        .def("remove_node", py::overload_cast<int>(&UndirectedGraph::remove_node))
        .def("add_edge", py::overload_cast<const std::string&, const std::string&>(&UndirectedGraph::add_edge))
        .def("add_edge", py::overload_cast<int, int>(&UndirectedGraph::add_edge))
        .def("has_edge", py::overload_cast<const std::string&, const std::string&>(&UndirectedGraph::has_edge, py::const_))
        .def("has_edge", py::overload_cast<int, int>(&UndirectedGraph::has_edge, py::const_))
        .def("remove_edge", py::overload_cast<const std::string&, const std::string&>(&UndirectedGraph::remove_edge))
        .def("remove_edge", py::overload_cast<int, int>(&UndirectedGraph::remove_edge))
        .def("has_path", py::overload_cast<const std::string&, const std::string&>(&UndirectedGraph::has_path, py::const_))
        .def("has_path", py::overload_cast<int, int>(&UndirectedGraph::has_path, py::const_))
        .def("is_valid", &UndirectedGraph::is_valid);

    py::class_<PartiallyDirectedGraph>(graph, "PartiallyDirectedGraph")
        .def(py::init<>())
        .def(py::init<const std::vector<std::string>&>())
        .def(py::init<const EdgeVector&, const ArcVector&>())
        .def(py::init<const std::vector<std::string>&, const EdgeVector&, const ArcVector&>())
        .def("num_nodes", &PartiallyDirectedGraph::num_nodes)
        .def("num_edges", &PartiallyDirectedGraph::num_edges)
        .def("num_arcs", &PartiallyDirectedGraph::num_arcs)
        .def("num_neighbors", py::overload_cast<const std::string&>(&PartiallyDirectedGraph::num_neighbors, py::const_))
        .def("num_neighbors", py::overload_cast<int>(&PartiallyDirectedGraph::num_neighbors, py::const_))
        .def("num_parents", py::overload_cast<const std::string&>(&PartiallyDirectedGraph::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&PartiallyDirectedGraph::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&PartiallyDirectedGraph::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&PartiallyDirectedGraph::num_children, py::const_))
        .def("nodes", &PartiallyDirectedGraph::nodes)
        .def("index", &PartiallyDirectedGraph::index)
        .def("name", [](PartiallyDirectedGraph& self, int idx) -> std::string {
            return self.name(idx);
        })
        .def("contains_node", &PartiallyDirectedGraph::contains_node)
        .def("edges", &PartiallyDirectedGraph::edges)
        .def("arcs", &PartiallyDirectedGraph::arcs)
        .def("neighbors", py::overload_cast<const std::string&>(&PartiallyDirectedGraph::neighbors, py::const_))
        .def("neighbors", py::overload_cast<int>(&PartiallyDirectedGraph::neighbors, py::const_))
        .def("parents", py::overload_cast<const std::string&>(&PartiallyDirectedGraph::parents, py::const_))
        .def("parents", py::overload_cast<int>(&PartiallyDirectedGraph::parents, py::const_))
        .def("add_node", &PartiallyDirectedGraph::add_node)
        .def("remove_node", py::overload_cast<const std::string&>(&PartiallyDirectedGraph::remove_node))
        .def("remove_node", py::overload_cast<int>(&PartiallyDirectedGraph::remove_node))
        .def("add_edge", py::overload_cast<const std::string&, const std::string&>(&PartiallyDirectedGraph::add_edge))
        .def("add_edge", py::overload_cast<int, int>(&PartiallyDirectedGraph::add_edge))
        .def("add_arc", py::overload_cast<const std::string&, const std::string&>(&PartiallyDirectedGraph::add_arc))
        .def("add_arc", py::overload_cast<int, int>(&PartiallyDirectedGraph::add_arc))
        .def("has_edge", py::overload_cast<const std::string&, const std::string&>(&PartiallyDirectedGraph::has_edge, py::const_))
        .def("has_edge", py::overload_cast<int, int>(&PartiallyDirectedGraph::has_edge, py::const_))
        .def("has_arc", py::overload_cast<const std::string&, const std::string&>(&PartiallyDirectedGraph::has_arc, py::const_))
        .def("has_arc", py::overload_cast<int, int>(&PartiallyDirectedGraph::has_arc, py::const_))
        .def("remove_edge", py::overload_cast<const std::string&, const std::string&>(&PartiallyDirectedGraph::remove_edge))
        .def("remove_edge", py::overload_cast<int, int>(&PartiallyDirectedGraph::remove_edge))
        .def("remove_arc", py::overload_cast<const std::string&, const std::string&>(&PartiallyDirectedGraph::remove_arc))
        .def("remove_arc", py::overload_cast<int, int>(&PartiallyDirectedGraph::remove_arc))
        .def("flip_arc", py::overload_cast<const std::string&, const std::string&>(&PartiallyDirectedGraph::flip_arc))
        .def("flip_arc", py::overload_cast<int, int>(&PartiallyDirectedGraph::flip_arc))
        .def("direct", py::overload_cast<const std::string&, const std::string&>(&PartiallyDirectedGraph::direct))
        .def("direct", py::overload_cast<int, int>(&PartiallyDirectedGraph::direct))
        .def("undirect", py::overload_cast<const std::string&, const std::string&>(&PartiallyDirectedGraph::undirect))
        .def("undirect", py::overload_cast<int, int>(&PartiallyDirectedGraph::undirect))
        .def("is_valid", &PartiallyDirectedGraph::is_valid);
}