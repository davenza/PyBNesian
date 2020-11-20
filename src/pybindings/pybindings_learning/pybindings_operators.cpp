#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <learning/operators/operators.hpp>

namespace py = pybind11;

using learning::operators::OperatorType, learning::operators::Operator, 
      learning::operators::ArcOperator, learning::operators::AddArc, 
      learning::operators::RemoveArc, learning::operators::FlipArc, 
      learning::operators::ChangeNodeType, learning::operators::OperatorTabuSet, 
      learning::operators::OperatorSetType, learning::operators::OperatorSet, 
      learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet, 
      learning::operators::OperatorPool;

void register_ArcOperators(py::module& m) {
    py::class_<Operator, std::shared_ptr<Operator>>(m, "Operator")
        .def_property_readonly("delta", &Operator::delta)
        .def_property_readonly("type", &Operator::type)
        .def("apply", &Operator::apply)
        .def("opposite", &Operator::opposite, py::return_value_policy::take_ownership)
        .def("__eq__", [](const Operator& self, const Operator& other) {
            return self == other;
        }, py::is_operator())
        .def("__ne__", [](const Operator& self, const Operator& other) {
            return self != other;
        }, py::is_operator());

    py::class_<ArcOperator, Operator, std::shared_ptr<ArcOperator>>(m, "ArcOperator")
        .def_property_readonly("source", &ArcOperator::source)
        .def_property_readonly("target", &ArcOperator::target);

    py::class_<AddArc, ArcOperator, std::shared_ptr<AddArc>>(m, "AddArc")
        .def(py::init<std::string, std::string, double>())
        .def("apply", &AddArc::apply)
        .def("opposite", &AddArc::opposite, py::return_value_policy::take_ownership);

    py::class_<RemoveArc, ArcOperator, std::shared_ptr<RemoveArc>>(m, "RemoveArc")
        .def(py::init<std::string, std::string, double>())
        .def("apply", &RemoveArc::apply)
        .def("opposite", &RemoveArc::opposite, py::return_value_policy::take_ownership);

    py::class_<FlipArc, ArcOperator, std::shared_ptr<FlipArc>>(m, "FlipArc")
        .def(py::init<std::string, std::string, double>())
        .def("apply", &FlipArc::apply)
        .def("opposite", &FlipArc::opposite, py::return_value_policy::take_ownership);
}

void register_OperatorTabuSet(py::module& m) {

    // TODO: Implement copy operation.
    py::class_<OperatorTabuSet>(m, "OperatorTabuSet")
        .def(py::init<>())
        .def(py::init<const OperatorTabuSet&>())
        .def("insert", py::overload_cast<std::shared_ptr<Operator>>(&OperatorTabuSet::insert))
        .def("contains", py::overload_cast<std::shared_ptr<Operator>&>(&OperatorTabuSet::contains, py::const_))
        .def("clear", &OperatorTabuSet::clear)
        .def("empty", &OperatorTabuSet::empty);
}

template<typename Model, typename... Models>
py::class_<OperatorSet, std::shared_ptr<OperatorSet>> register_OperatorSet(py::module& m) {

    auto op_set = [&m]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<OperatorSet, std::shared_ptr<OperatorSet>> op_set(m, "OperatorSet");
            // op_set.def(py::init<>());
            return op_set;
        } else {
            return register_OperatorSet<Models...>(m);
        }
    }();

    op_set.def("cache_scores", [](OperatorSet& self, Model& model) {
        self.cache_scores(model);
    });
    op_set.def("find_max", [](OperatorSet& self, Model& model) {
        return self.find_max(model);
    });
    op_set.def("find_max", [](OperatorSet& self, Model& model, OperatorTabuSet& tabu) {
        return self.find_max(model, tabu);
    });
    op_set.def("update_scores", [](OperatorSet& self, Model& model, Operator& op) {
        self.update_scores(model, op);
    });

    op_set.def("set_arc_blacklist", &OperatorSet::set_arc_blacklist);
    op_set.def("set_arc_whitelist", &OperatorSet::set_arc_whitelist);
    op_set.def("set_max_indegree", &OperatorSet::set_max_indegree);
    op_set.def("set_type_whitelist", &OperatorSet::set_type_whitelist);

    return op_set;
}

template<typename DerivedOpSet, typename Model, typename... Models>
py::class_<DerivedOpSet, OperatorSet, std::shared_ptr<DerivedOpSet>> register_DerivedOperatorSet(py::module& m, const char* name) {
    auto op_set = [&m, name]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<DerivedOpSet, OperatorSet, std::shared_ptr<DerivedOpSet>> op_set(m, name);
            return op_set;
        } else {
            return register_DerivedOperatorSet<DerivedOpSet, Models...>(m, name);
        }
    }();

    op_set.def("cache_scores", [](DerivedOpSet& self, Model& model) {
        self.cache_scores(model);
    })
    .def("find_max", [](DerivedOpSet& self, Model& model) {
        return self.find_max(model);
    })
    .def("find_max", [](DerivedOpSet& self, Model& model, OperatorTabuSet& tabu) {
        return self.find_max(model, tabu);
    })
    .def("update_scores", [](DerivedOpSet& self, Model& model, Operator& op) {
        self.update_scores(model, op);
    });

    return op_set;
}


template<typename Model, typename... Models>
py::class_<OperatorPool> register_OperatorPool(py::module& m) {
    auto pool = [&m]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<OperatorPool> pool(m, "OperatorPool");
            return pool;
        } else {
            return register_OperatorPool<Models...>(m);
        }
    }();

    pool.def(py::init<Model&, std::shared_ptr<Score>&, std::vector<std::shared_ptr<OperatorSet>>>());
    pool.def("cache_scores", [](OperatorPool& self, Model& model) {
        self.cache_scores(model);
    });
    pool.def("find_max", [](OperatorPool& self, Model& model) {
        return self.find_max(model);
    });
    pool.def("find_max", [](OperatorPool& self, Model& model, OperatorTabuSet& tabu) {
        return self.find_max(model, tabu);
    });
    pool.def("update_scores", [](OperatorPool& self, Model& model, Operator& op) {
        self.update_scores(model, op);
    });

    return pool;
}


void pybindings_operators(py::module& root) {
    auto operators = root.def_submodule("operators", "Learning operators submodule");

    py::class_<OperatorType>(operators, "OperatorType")
        .def_property_readonly_static("ADD_ARC", [](const py::object&) { 
            return OperatorType(OperatorType::ADD_ARC);
        })
        .def_property_readonly_static("REMOVE_ARC", [](const py::object&) { 
            return OperatorType(OperatorType::REMOVE_ARC);
        })
        .def_property_readonly_static("FLIP_ARC", [](const py::object&) { 
            return OperatorType(OperatorType::FLIP_ARC);
        })
        .def_property_readonly_static("CHANGE_NODE_TYPE", [](const py::object&) { 
            return OperatorType(OperatorType::CHANGE_NODE_TYPE);
        })
        .def(py::self == py::self)
        .def(py::self != py::self);

    register_ArcOperators(operators);

    py::class_<ChangeNodeType, Operator, std::shared_ptr<ChangeNodeType>>(operators, "ChangeNodeType")
        .def(py::init<std::string, FactorType, double>())
        .def_property_readonly("node", &ChangeNodeType::node)
        .def_property_readonly("node_type", &ChangeNodeType::node_type)
        .def("apply", &ChangeNodeType::apply)
        .def("opposite", &ChangeNodeType::opposite);

    register_OperatorTabuSet(operators);

    register_OperatorSet<GaussianNetwork, SemiparametricBN>(operators);
    auto arc_set = register_DerivedOperatorSet<ArcOperatorSet,
                                                GaussianNetwork,
                                                SemiparametricBN>(operators, "ArcOperatorSet");

    arc_set.def(py::init<std::shared_ptr<Score>&, ArcStringVector, ArcStringVector, int>(),
                py::arg("score"),
                py::arg("blacklist") = ArcStringVector(),
                py::arg("whitelist") = ArcStringVector(),
                py::arg("max_indegree") = 0);


    auto nodetype = register_DerivedOperatorSet<ChangeNodeTypeSet,
                                                    SemiparametricBN>(operators, "ChangeNodeTypeSet");
    nodetype.def(py::init<std::shared_ptr<Score>&, FactorStringTypeVector>(), 
                 py::arg("score"),
                 py::arg("type_whitelist") = FactorStringTypeVector());
    
    register_OperatorPool<GaussianNetwork,
                          SemiparametricBN>(operators);

    py::class_<OperatorSetType>(operators, "OperatorSetType")
        .def_property_readonly_static("ARCS", [](const py::object&) { 
            return OperatorSetType(OperatorSetType::ARCS);
        })
        .def_property_readonly_static("NODE_TYPE", [](const py::object&) { 
            return OperatorSetType(OperatorSetType::NODE_TYPE);
        })
        .def(py::self == py::self)
        .def(py::self != py::self);
}
