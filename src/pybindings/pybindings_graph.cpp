#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <graph/generic_graph.hpp>

namespace py = pybind11;

using graph::DirectedGraph, graph::Dag, graph::UndirectedGraph, graph::PartiallyDirectedGraph;
using graph::ConditionalDirectedGraph, graph::ConditionalDag, 
      graph::ConditionalUndirectedGraph, graph::ConditionalPartiallyDirectedGraph;


template<typename CppClass, typename PyClass>
void add_graphbase_methods(PyClass& c) {
    c
    .def("num_nodes", &CppClass::num_nodes)
    .def("nodes", &CppClass::nodes)
    .def("indices", &CppClass::indices)
    .def("collapsed_indices", &CppClass::collapsed_indices)
    .def("contains_node", &CppClass::contains_node)
    .def("add_node", &CppClass::add_node)
    .def("remove_node", [](CppClass& self, int index) {
        self.remove_node(index);
    })
    .def("remove_node", [](CppClass& self, const std::string& name) {
        self.remove_node(name);
    })
    .def("name", &CppClass::name)
    .def("collapsed_name", &CppClass::collapsed_name)
    .def("index", &CppClass::index)
    .def("collapsed_index", &CppClass::collapsed_index)
    .def("index_from_collapsed", &CppClass::index_from_collapsed)
    .def("collapsed_from_index", &CppClass::collapsed_from_index)
    .def("is_valid", &CppClass::is_valid);
}

template<typename CppClass, typename PyClass>
void add_conditionalgraphbase_methods(PyClass& c) {
    c
    .def("num_nodes", &CppClass::num_nodes)
    .def("num_interface_nodes", &CppClass::num_interface_nodes)
    .def("num_total_nodes", &CppClass::num_total_nodes)
    .def("nodes", &CppClass::nodes)
    .def("interface_nodes", &CppClass::interface_nodes)
    .def("all_nodes", &CppClass::all_nodes)
    .def("indices", &CppClass::indices)
    .def("collapsed_indices", &CppClass::collapsed_indices)
    .def("interface_collapsed_indices", &CppClass::interface_collapsed_indices)
    .def("joint_collapsed_indices", &CppClass::joint_collapsed_indices)
    .def("contains_node", &CppClass::contains_node)
    .def("contains_interface_node", &CppClass::contains_interface_node)
    .def("contains_total_node", &CppClass::contains_total_node)
    .def("add_node", &CppClass::add_node)
    .def("add_interface_node", &CppClass::add_interface_node)
    .def("remove_node", [](CppClass& self, int index) {
        self.remove_node(index);
    })
    .def("remove_node", [](CppClass& self, const std::string& name) {
        self.remove_node(name);
    })
    .def("remove_interface_node", [](CppClass& self, int index) {
        self.remove_interface_node(index);
    })
    .def("remove_interface_node", [](CppClass& self, const std::string& name) {
        self.remove_interface_node(name);
    })
    .def("name", &CppClass::name)
    .def("collapsed_name", &CppClass::collapsed_name)
    .def("interface_collapsed_name", &CppClass::interface_collapsed_name)
    .def("joint_collapsed_name", &CppClass::joint_collapsed_name)
    .def("index", &CppClass::index)
    .def("collapsed_index", &CppClass::collapsed_index)
    .def("interface_collapsed_index", &CppClass::interface_collapsed_index)
    .def("joint_collapsed_index", &CppClass::joint_collapsed_index)
    .def("index_from_collapsed", &CppClass::index_from_collapsed)
    .def("index_from_interface_collapsed", &CppClass::index_from_interface_collapsed)
    .def("index_from_joint_collapsed", &CppClass::index_from_joint_collapsed)
    .def("collapsed_from_index", &CppClass::collapsed_from_index)
    .def("interface_collapsed_from_index", &CppClass::interface_collapsed_from_index)
    .def("joint_collapsed_from_index", &CppClass::joint_collapsed_from_index)
    .def("is_interface", [](CppClass& self, int index) {
        return self.is_interface(index);
    })
    .def("is_interface", [](CppClass& self, const std::string& name) {
        return self.is_interface(name);
    })
    .def("set_interface", [](CppClass& self, int index) {
        self.is_interface(index);
    })
    .def("set_interface", [](CppClass& self, const std::string& name) {
        self.set_interface(name);
    })
    .def("set_node", [](CppClass& self, int index) {
        self.set_node(index);
    })
    .def("set_node", [](CppClass& self, const std::string& name) {
        self.set_node(name);
    })
    .def("is_valid", &CppClass::is_valid);
}


template<typename CppClass, typename PyClass>
void add_arcgraph_methods(PyClass& c) {
    c
    .def("num_arcs", &CppClass::num_arcs)
    .def("num_parents", [](CppClass& self, int index) {
        return self.num_parents(index);
    })
    .def("num_parents", [](CppClass& self, const std::string& name) {
        return self.num_parents(name);
    })
    .def("num_children", [](CppClass& self, int index) {
        return self.num_parents(index);
    })
    .def("num_children", [](CppClass& self, const std::string& name) {
        return self.num_parents(name);
    })
    .def("arcs", &CppClass::arcs)
    .def("parents", [](CppClass& self, int index) {
        return self.parents(index);
    })
    .def("parents", [](CppClass& self, const std::string& name) {
        return self.parents(name);
    })
    .def("children", [](CppClass& self, int index) {
        return self.children(index);
    })
    .def("children", [](CppClass& self, const std::string& name) {
        return self.children(name);
    })
    .def("add_arc", [](CppClass& self, int source, int target) {
        self.add_arc(source, target);
    })
    .def("add_arc", [](CppClass& self, const std::string& source, const std::string& target) {
        self.add_arc(source, target);
    })
    .def("has_arc", [](CppClass& self, int source, int target) {
        return self.has_arc(source, target);
    })
    .def("has_arc", [](CppClass& self, const std::string& source, const std::string& target) {
        return self.has_arc(source, target);
    })
    .def("remove_arc", [](CppClass& self, int source, int target) {
        self.remove_arc(source, target);
    })
    .def("remove_arc", [](CppClass& self, const std::string& source, const std::string& target) {
        self.remove_arc(source, target);
    })
    .def("flip_arc", [](CppClass& self, int source, int target) {
        self.flip_arc(source, target);
    })
    .def("flip_arc", [](CppClass& self, const std::string& source, const std::string& target) {
        self.flip_arc(source, target);
    })
    .def("roots", [](CppClass& self) {
        std::unordered_set<std::string> roots;
        for (const auto& r : self.roots()) {
            roots.insert(self.name(r));
        }
        return roots;
    })
    .def("leaves", [](CppClass& self) {
        std::unordered_set<std::string> leaves;
        for (const auto& lv : self.leaves()) {
            leaves.insert(self.name(lv));
        }
        return leaves;
    });

    if constexpr (graph::is_conditional_graph_v<CppClass>) {
        c
        .def("interface_arcs", &CppClass::template interface_arcs<>);
    }
}

template<typename CppClass, typename PyClass>
void add_edgegraph_methods(PyClass& c) {
    c
    .def("num_edges", &CppClass::num_edges)
    .def("num_neighbors", [](CppClass& self, int index) {
        return self.num_neighbors(index);
    })
    .def("num_neighbors", [](CppClass& self, const std::string& name) {
        return self.num_neighbors(name);
    })
    .def("edges", &CppClass::edges)
    .def("neighbors", [](CppClass& self, int index) {
        return self.neighbors(index);
    })
    .def("neighbors", [](CppClass& self, const std::string& name) {
        return self.neighbors(name);
    })
    .def("add_edge", [](CppClass& self, int source, int target) {
        self.add_edge(source, target);
    })
    .def("add_edge", [](CppClass& self, const std::string& source, const std::string& target) {
        self.add_edge(source, target);
    })
    .def("has_edge", [](CppClass& self, int source, int target) {
        return self.has_edge(source, target);
    })
    .def("has_edge", [](CppClass& self, const std::string& source, const std::string& target) {
        return self.has_edge(source, target);
    })
    .def("remove_edge", [](CppClass& self, int source, int target) {
        self.remove_edge(source, target);
    })
    .def("remove_edge", [](CppClass& self, const std::string& source, const std::string& target) {
        self.remove_edge(source, target);
    });

    if constexpr (graph::is_conditional_graph_v<CppClass>) {
        c
        .def("interface_edges", &CppClass::template interface_edges<>);
    }
}

template<typename CppClass, typename PyClass>
void add_pdag_methods(PyClass& c) {
    c
    .def("direct", [](CppClass& self, int source, int target) {
        self.direct(source, target);
    })
    .def("direct", [](CppClass& self, const std::string& source, const std::string& target) {
        self.direct(source, target);
    })
    .def("undirect", [](CppClass& self, int source, int target) {
        self.undirect(source, target);
    })
    .def("undirect", [](CppClass& self, const std::string& source, const std::string& target) {
        self.undirect(source, target);
    })
    .def("has_connection", [](CppClass& self, int source, int target) {
        return self.has_connection(source, target);
    })
    .def("has_connection", [](CppClass& self, const std::string& source, const std::string& target) {
        return self.has_connection(source, target);
    })
    .def("to_dag", [](const CppClass& self) {
        return self.to_dag();
    });
}

template<typename CppClass, typename PyClass>
void add_directed_methods(PyClass& c) {
    c
    .def("has_path", [](CppClass& self, int source, int target) {
        return self.has_path(source, target);
    })
    .def("has_path", [](CppClass& self, const std::string& source, const std::string& target) {
        return self.has_path(source, target);
    });
}

template<typename CppClass, typename PyClass>
void add_undirected_methods(PyClass& c) {
    c
    .def("has_path", [](CppClass& self, int source, int target) {
        return self.has_path(source, target);
    })
    .def("has_path", [](CppClass& self, const std::string& source, const std::string& target) {
        return self.has_path(source, target);
    });
}

template<typename CppClass, typename PyClass>
void add_pickle_methods(PyClass& c) {
    c
    .def("save", &CppClass::save)
    .def(py::pickle(
        [](const CppClass& self) {
            return self.__getstate__();
        }, 
        [](py::tuple t) {
            return CppClass::__setstate__(t);
        }
    ));
}

template<typename CppClass, typename PyClass>
void add_to_conditional_methods(PyClass& c) {
    c
    .def("conditional_graph", py::overload_cast<>(&CppClass::conditional_graph, py::const_))
    .def("conditional_graph", py::overload_cast<const std::vector<std::string>&,
                                                const std::vector<std::string>&>
                                (&CppClass::conditional_graph, py::const_))
    .def("unconditional_graph", &CppClass::unconditional_graph);
}

void pybindings_normal_graph(py::module& graph) {
    py::class_<DirectedGraph> dg(graph, "DirectedGraph");
    dg
    .def(py::init<>())
    .def(py::init<const std::vector<std::string>&>())
    .def(py::init<const ArcStringVector&>())
    .def(py::init<const std::vector<std::string>&, const ArcStringVector&>());

    add_graphbase_methods<DirectedGraph>(dg);
    add_arcgraph_methods<DirectedGraph>(dg);
    add_directed_methods<DirectedGraph>(dg);
    add_to_conditional_methods<DirectedGraph>(dg);
    add_pickle_methods<DirectedGraph>(dg);

    py::class_<Dag, DirectedGraph> dag(graph, "Dag");
    dag
    .def(py::init<>())
    .def(py::init<const std::vector<std::string>&>())
    .def(py::init<const ArcStringVector&>())
    .def(py::init<const std::vector<std::string>&, const ArcStringVector&>())
    .def("topological_sort", &Dag::topological_sort)
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
    .def("add_arc", [](Dag& self, int source, int target) {
        self.add_arc(source, target);
    })
    .def("add_arc", [](Dag& self, const std::string& source, const std::string& target) {
        self.add_arc(source, target);
    })
    .def("flip_arc", [](Dag& self, int source, int target) {
        self.flip_arc(source, target);
    })
    .def("flip_arc", [](Dag& self, const std::string& source, const std::string& target) {
        self.flip_arc(source, target);
    })
    .def("to_pdag", &Dag::to_pdag);

    add_to_conditional_methods<Dag>(dag);
    add_pickle_methods<Dag>(dag);

    py::class_<UndirectedGraph> ug(graph, "UndirectedGraph");
    ug
    .def(py::init<>())
    .def(py::init<const std::vector<std::string>&>())
    .def(py::init<const EdgeStringVector&>())
    .def(py::init<const std::vector<std::string>&, const EdgeStringVector&>())
    .def_static("Complete", &UndirectedGraph::Complete);

    add_graphbase_methods<UndirectedGraph>(ug);
    add_edgegraph_methods<UndirectedGraph>(ug);
    add_undirected_methods<UndirectedGraph>(ug);
    add_to_conditional_methods<UndirectedGraph>(ug);
    add_pickle_methods<UndirectedGraph>(ug);   

    py::class_<PartiallyDirectedGraph> pdag(graph, "PartiallyDirectedGraph");
    pdag
    .def(py::init<>())
    .def(py::init<const std::vector<std::string>&>())
    .def(py::init<const ArcStringVector&, const EdgeStringVector&>())
    .def(py::init<const std::vector<std::string>&, const ArcStringVector&, const EdgeStringVector&>())
    .def_static("CompleteUndirected", &PartiallyDirectedGraph::CompleteUndirected);

    add_graphbase_methods<PartiallyDirectedGraph>(pdag);
    add_arcgraph_methods<PartiallyDirectedGraph>(pdag);
    add_edgegraph_methods<PartiallyDirectedGraph>(pdag);
    add_pdag_methods<PartiallyDirectedGraph>(pdag);
    add_to_conditional_methods<PartiallyDirectedGraph>(pdag);
    add_pickle_methods<PartiallyDirectedGraph>(pdag);
}

void pybindings_conditional_graph(py::module& graph) {
    py::class_<ConditionalDirectedGraph> cdg(graph, "ConditionalDirectedGraph");
    cdg
    .def(py::init<>())
    .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>())
    .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const ArcStringVector&>());

    add_conditionalgraphbase_methods<ConditionalDirectedGraph>(cdg);
    add_arcgraph_methods<ConditionalDirectedGraph>(cdg);
    add_directed_methods<ConditionalDirectedGraph>(cdg);
    add_to_conditional_methods<ConditionalDirectedGraph>(cdg);
    add_pickle_methods<ConditionalDirectedGraph>(cdg);


    py::class_<ConditionalDag, ConditionalDirectedGraph> cdag(graph, "ConditionalDag");
    cdag
    .def(py::init<>())
    .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>())
    .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const ArcStringVector&>())
    .def("topological_sort", &ConditionalDag::topological_sort)
    .def("can_add_arc", [](ConditionalDag& self, int source, int target) {
        return self.can_add_arc(source, target);
    })
    .def("can_add_arc", [](ConditionalDag& self, const std::string& source, const std::string& target) {
        return self.can_add_arc(source, target);
    })
    .def("can_flip_arc", [](ConditionalDag& self, int source, int target) {
        return self.can_flip_arc(source, target);
    })
    .def("can_flip_arc", [](ConditionalDag& self, const std::string& source, const std::string& target) {
        return self.can_flip_arc(source, target);
    })
    .def("add_arc", [](ConditionalDag& self, int source, int target) {
        self.add_arc(source, target);
    })
    .def("add_arc", [](ConditionalDag& self, const std::string& source, const std::string& target) {
        self.add_arc(source, target);
    })
    .def("flip_arc", [](ConditionalDag& self, int source, int target) {
        self.flip_arc(source, target);
    })
    .def("flip_arc", [](ConditionalDag& self, const std::string& source, const std::string& target) {
        self.flip_arc(source, target);
    })
    .def("to_pdag", &ConditionalDag::to_pdag);

    add_to_conditional_methods<ConditionalDag>(cdag);
    add_pickle_methods<ConditionalDag>(cdag);

    py::class_<ConditionalUndirectedGraph> cug(graph, "ConditionalUndirectedGraph");
    cug
    .def(py::init<>())
    .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>())
    .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const EdgeStringVector&>())
    .def_static("Complete", &ConditionalUndirectedGraph::Complete);

    add_conditionalgraphbase_methods<ConditionalUndirectedGraph>(cug);
    add_edgegraph_methods<ConditionalUndirectedGraph>(cug);
    add_undirected_methods<ConditionalUndirectedGraph>(cug);
    add_to_conditional_methods<ConditionalUndirectedGraph>(cug);
    add_pickle_methods<ConditionalUndirectedGraph>(cug);

    py::class_<ConditionalPartiallyDirectedGraph> cpdag(graph, "ConditionalPartiallyDirectedGraph");
    cpdag
    .def(py::init<>())
    .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>())
    .def(py::init<const std::vector<std::string>&,
                  const std::vector<std::string>&,
                  const ArcStringVector&,
                  const EdgeStringVector&>())
    .def_static("CompleteUndirected", &ConditionalPartiallyDirectedGraph::CompleteUndirected);

    add_conditionalgraphbase_methods<ConditionalPartiallyDirectedGraph>(cpdag);
    add_arcgraph_methods<ConditionalPartiallyDirectedGraph>(cpdag);
    add_edgegraph_methods<ConditionalPartiallyDirectedGraph>(cpdag);
    add_pdag_methods<ConditionalPartiallyDirectedGraph>(cpdag);
    add_to_conditional_methods<ConditionalPartiallyDirectedGraph>(cpdag);
    add_pickle_methods<ConditionalPartiallyDirectedGraph>(cpdag);
}



void pybindings_graph(py::module& root) {
    auto graph = root.def_submodule("graph","Graph submodule");

    pybindings_normal_graph(graph);
    pybindings_conditional_graph(graph);

}