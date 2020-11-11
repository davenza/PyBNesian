#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <graph/generic_graph.hpp>

namespace py = pybind11;

using graph::DirectedGraph, graph::Dag, graph::UndirectedGraph, graph::PartiallyDirectedGraph;


void pybindings_graph(py::module& root) {
    auto graph = root.def_submodule("graph","Graph submodule");

    graph.def("load_graph", &graph::load_graph);

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
        .def("num_parents", [](DirectedGraph& self, int n) {
            return self.num_parents(n);
        })
        .def("num_parents", [](DirectedGraph& self, const std::string& n) {
            return self.num_parents(n);
        })
        .def("num_children", [](DirectedGraph& self, int n) {
            return self.num_children(n);
        })
        .def("num_children", [](DirectedGraph& self, const std::string& n) {
            return self.num_children(n);
        })
        .def("nodes", &DirectedGraph::nodes)
        .def("index", &DirectedGraph::index)
        .def("name", &DirectedGraph::name)
        .def("contains_node", &DirectedGraph::contains_node)
        .def("arcs", &DirectedGraph::arcs)
        .def("parents", [](DirectedGraph& self, int n) {
            return self.parents(n);
        })
        .def("parents", [](DirectedGraph& self, const std::string& n) {
            return self.parents(n);
        })
        .def("add_node", &DirectedGraph::add_node)
        .def("remove_node", [](DirectedGraph& self, int n) {
            self.remove_node(n);
        })
        .def("remove_node", [](DirectedGraph& self, const std::string& n) {
            self.remove_node(n);
        })
        .def("add_arc", [](DirectedGraph& self, int source, int target) {
            self.add_arc(source, target);
        })
        .def("add_arc", [](DirectedGraph& self, const std::string& source, const std::string& target) {
            self.add_arc(source, target);
        })
        .def("has_arc", [](DirectedGraph& self, int source, int target) {
            return self.has_arc(source, target);
        })
        .def("has_arc", [](DirectedGraph& self, const std::string& source, const std::string& target) {
            return self.has_arc(source, target);
        })
        .def("remove_arc", [](DirectedGraph& self, int source, int target) {
            self.remove_arc(source, target);
        })
        .def("remove_arc", [](DirectedGraph& self, const std::string& source, const std::string& target) {
            self.remove_arc(source, target);
        })
        .def("flip_arc", [](DirectedGraph& self, int source, int target) {
            self.flip_arc(source, target);
        })
        .def("flip_arc", [](DirectedGraph& self, const std::string& source, const std::string& target) {
            self.flip_arc(source, target);
        })
        .def("has_path", [](DirectedGraph& self, int source, int target) {
            return self.has_path(source, target);
        })
        .def("has_path", [](DirectedGraph& self, const std::string& source, const std::string& target) {
            return self.has_path(source, target);
        })
        .def("is_valid", &DirectedGraph::is_valid)
        .def("save", &DirectedGraph::save)
        .def(py::pickle(
            [](const DirectedGraph& self) {
                return self.__getstate__();
            }, 
            [](py::tuple t) {
                return DirectedGraph::__setstate__(t);
            }
        ));

    py::class_<Dag, DirectedGraph>(graph, "Dag")
        .def(py::init<>())
        .def(py::init<const std::vector<std::string>&>())
        .def(py::init<const ArcVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcVector&>())
        .def("can_add_arc", [](Dag& self, int source, int target) {
            return self.can_add_arc(source, target);
        })
        .def("can_add_arc", [](Dag& self, const std::string& source, const std::string& target) {
            return self.can_add_arc(source, target);
        })
        .def("can_flip_arc", [](Dag& self, int source, int target) {
            return self.can_flip_arc(source, target);
        })
        .def("can_flip_arc", [](Dag& self, const std::string& source, const std::string& target) {
            return self.can_flip_arc(source, target);
        })
        .def("to_pdag", &Dag::to_pdag)
        .def("save", &Dag::save)
        .def(py::pickle(
            [](const Dag& self) {
                return self.__getstate__();
            }, 
            [](py::tuple t) {
                return Dag::__setstate__(t);
            }
        ));

    py::class_<UndirectedGraph>(graph, "UndirectedGraph")
        .def(py::init<>())
        .def(py::init<const std::vector<std::string>&>())
        .def(py::init<const EdgeVector&>())
        .def(py::init<const std::vector<std::string>&, const EdgeVector&>())
        .def_static("Complete", &UndirectedGraph::Complete)
        .def("num_nodes", &UndirectedGraph::num_nodes)
        .def("num_edges", &UndirectedGraph::num_edges)
        .def("num_neighbors", [](UndirectedGraph& self, int n) {
            return self.num_neighbors(n);
        })
        .def("num_neighbors", [](UndirectedGraph& self, const std::string& n) {
            return self.num_neighbors(n);
        })
        .def("nodes", &UndirectedGraph::nodes)
        .def("index", &UndirectedGraph::index)
        .def("name", [](UndirectedGraph& self, int idx) -> std::string {
            return self.name(idx);
        })
        .def("contains_node", &UndirectedGraph::contains_node)
        .def("edges", &UndirectedGraph::edges)
        .def("neighbors", [](UndirectedGraph& self, int n) {
            return self.neighbors(n);
        })
        .def("neighbors", [](UndirectedGraph& self, const std::string& n) {
            return self.neighbors(n);
        })
        .def("add_node", &UndirectedGraph::add_node)
        .def("remove_node", [](UndirectedGraph& self, int n) {
            self.remove_node(n);
        })
        .def("remove_node", [](UndirectedGraph& self, const std::string& n) {
            self.remove_node(n);
        })
        .def("add_edge", [](UndirectedGraph& self, int source, int target) {
            self.add_edge(source, target);
        })
        .def("add_edge", [](UndirectedGraph& self, const std::string& source, const std::string& target) {
            self.add_edge(source, target);
        })
        .def("has_edge", [](UndirectedGraph& self, int source, int target) {
            return self.has_edge(source, target);
        })
        .def("has_edge", [](UndirectedGraph& self, const std::string& source, const std::string& target) {
            return self.has_edge(source, target);
        })
        .def("remove_edge", [](UndirectedGraph& self, int source, int target) {
            self.remove_edge(source, target);
        })
        .def("remove_edge", [](UndirectedGraph& self, const std::string& source, const std::string& target) {
            self.remove_edge(source, target);
        })
        .def("has_path", [](UndirectedGraph& self, int source, int target) {
            return self.has_path(source, target);
        })
        .def("has_path", [](UndirectedGraph& self, const std::string& source, const std::string& target) {
            return self.has_path(source, target);
        })
        .def("is_valid", &UndirectedGraph::is_valid)
        .def("save", &UndirectedGraph::save)
        .def(py::pickle(
            [](const UndirectedGraph& self) {
                return self.__getstate__();
            }, 
            [](py::tuple t) {
                return UndirectedGraph::__setstate__(t);
            }
        ));

    py::class_<PartiallyDirectedGraph>(graph, "PartiallyDirectedGraph")
        .def(py::init<>())
        .def(py::init<const std::vector<std::string>&>())
        .def(py::init<const ArcVector&, const EdgeVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcVector&, const EdgeVector&>())
        .def("num_nodes", &PartiallyDirectedGraph::num_nodes)
        .def("num_edges", &PartiallyDirectedGraph::num_edges)
        .def("num_arcs", &PartiallyDirectedGraph::num_arcs)
        .def("num_neighbors", [](PartiallyDirectedGraph& self, int n) {
            return self.num_neighbors(n);
        })
        .def("num_neighbors", [](PartiallyDirectedGraph& self, const std::string& n) {
            return self.num_neighbors(n);
        })
        .def("num_parents", [](PartiallyDirectedGraph& self, int n) {
            return self.num_parents(n);
        })
        .def("num_parents", [](PartiallyDirectedGraph& self, const std::string& n) {
            return self.num_parents(n);
        })
        .def("num_children", [](PartiallyDirectedGraph& self, int n) {
            return self.num_children(n);
        })
        .def("num_children", [](PartiallyDirectedGraph& self, const std::string& n) {
            return self.num_children(n);
        })
        .def("nodes", &PartiallyDirectedGraph::nodes)
        .def("index", &PartiallyDirectedGraph::index)
        .def("name", [](PartiallyDirectedGraph& self, int idx) -> std::string {
            return self.name(idx);
        })
        .def("contains_node", &PartiallyDirectedGraph::contains_node)
        .def("edges", &PartiallyDirectedGraph::edges)
        .def("arcs", &PartiallyDirectedGraph::arcs)
        .def("neighbors", [](PartiallyDirectedGraph& self, int n) {
            return self.neighbors(n);
        })
        .def("neighbors", [](PartiallyDirectedGraph& self, const std::string& n) {
            return self.neighbors(n);
        })
        .def("parents", [](PartiallyDirectedGraph& self, int n) {
            return self.parents(n);
        })
        .def("parents", [](PartiallyDirectedGraph& self, const std::string& n) {
            return self.parents(n);
        })
        .def("add_node", &PartiallyDirectedGraph::add_node)
        .def("remove_node", [](PartiallyDirectedGraph& self, int n) {
            self.remove_node(n);
        })
        .def("remove_node", [](PartiallyDirectedGraph& self, const std::string& n) {
            self.remove_node(n);
        })
        .def("add_edge", [](PartiallyDirectedGraph& self, int source, int target) {
            self.add_edge(source, target);
        })
        .def("add_edge", [](PartiallyDirectedGraph& self, const std::string& source, const std::string& target) {
            self.add_edge(source, target);
        })
        .def("add_arc", [](PartiallyDirectedGraph& self, int source, int target) {
            self.add_arc(source, target);
        })
        .def("add_arc", [](PartiallyDirectedGraph& self, const std::string& source, const std::string& target) {
            self.add_arc(source, target);
        })
        .def("has_edge", [](PartiallyDirectedGraph& self, int source, int target) {
            return self.has_edge(source, target);
        })
        .def("has_edge", [](PartiallyDirectedGraph& self, const std::string& source, const std::string& target) {
            return self.has_edge(source, target);
        })
        .def("has_arc", [](PartiallyDirectedGraph& self, int source, int target) {
            return self.has_arc(source, target);
        })
        .def("has_arc", [](PartiallyDirectedGraph& self, const std::string& source, const std::string& target) {
            return self.has_arc(source, target);
        })
        .def("remove_edge", [](PartiallyDirectedGraph& self, int source, int target) {
            self.remove_edge(source, target);
        })
        .def("remove_edge", [](PartiallyDirectedGraph& self, const std::string& source, const std::string& target) {
            self.remove_edge(source, target);
        })
        .def("remove_arc", [](PartiallyDirectedGraph& self, int source, int target) {
            self.remove_arc(source, target);
        })
        .def("remove_arc", [](PartiallyDirectedGraph& self, const std::string& source, const std::string& target) {
            self.remove_arc(source, target);
        })
        .def("flip_arc", [](PartiallyDirectedGraph& self, int source, int target) {
            self.flip_arc(source, target);
        })
        .def("flip_arc", [](PartiallyDirectedGraph& self, const std::string& source, const std::string& target) {
            self.flip_arc(source, target);
        })
        .def("direct", [](PartiallyDirectedGraph& self, int source, int target) {
            self.direct(source, target);
        })
        .def("direct", [](PartiallyDirectedGraph& self, const std::string& source, const std::string& target) {
            self.direct(source, target);
        })
        .def("undirect", [](PartiallyDirectedGraph& self, int source, int target) {
            self.undirect(source, target);
        })
        .def("undirect", [](PartiallyDirectedGraph& self, const std::string& source, const std::string& target) {
            self.undirect(source, target);
        })
        .def("is_valid", &PartiallyDirectedGraph::is_valid)
        .def("to_dag", &PartiallyDirectedGraph::to_dag)
        .def("save", &PartiallyDirectedGraph::save)
        .def(py::pickle(
            [](const PartiallyDirectedGraph& self) {
                return self.__getstate__();
            }, 
            [](py::tuple t) {
                return PartiallyDirectedGraph::__setstate__(t);
            }
        ));
}