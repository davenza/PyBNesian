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
        .def("insert", py::overload_cast<const std::shared_ptr<Operator>&>(&OperatorTabuSet::insert))
        .def("contains", py::overload_cast<const std::shared_ptr<Operator>&>(&OperatorTabuSet::contains, py::const_))
        .def("clear", &OperatorTabuSet::clear)
        .def("empty", &OperatorTabuSet::empty);
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
        .def_static("from_string", &OperatorType::from_string)
        .def(py::self == py::self)
        .def(py::self != py::self);

    register_ArcOperators(operators);

    py::class_<ChangeNodeType, Operator, std::shared_ptr<ChangeNodeType>>(operators, "ChangeNodeType")
        .def(py::init<std::string, NodeType, double>())
        .def_property_readonly("node", &ChangeNodeType::node)
        .def_property_readonly("node_type", &ChangeNodeType::node_type)
        .def("apply", &ChangeNodeType::apply)
        .def("opposite", &ChangeNodeType::opposite);

    register_OperatorTabuSet(operators);

    py::class_<OperatorSet, std::shared_ptr<OperatorSet>>(operators, "OperatorSet")
        .def("cache_scores", [](OperatorSet& self, BayesianNetworkBase& model, Score& score) {
            self.cache_scores(model, score);
        })
        .def("find_max", [](OperatorSet& self, BayesianNetworkBase& model) {
            return self.find_max(model);
        })
        .def("find_max", [](OperatorSet& self, BayesianNetworkBase& model, OperatorTabuSet& tabu) {
            return self.find_max(model, tabu);
        })
        .def("update_scores", [](OperatorSet& self, BayesianNetworkBase& model, Score& score, Operator& op) {
            self.update_scores(model, score, op);
        })
        .def("set_arc_blacklist", py::overload_cast<const ArcStringVector&>(&OperatorSet::set_arc_blacklist))
        .def("set_arc_whitelist", py::overload_cast<const ArcStringVector&>(&OperatorSet::set_arc_whitelist))
        .def("set_max_indegree", &OperatorSet::set_max_indegree)
        .def("set_type_whitelist", &OperatorSet::set_type_whitelist);

    py::class_<ArcOperatorSet, OperatorSet, std::shared_ptr<ArcOperatorSet>>(operators, "ArcOperatorSet")
        .def(py::init<ArcStringVector, ArcStringVector, int>(),
                py::arg("blacklist") = ArcStringVector(),
                py::arg("whitelist") = ArcStringVector(),
                py::arg("max_indegree") = 0);

    py::class_<ChangeNodeTypeSet, OperatorSet, std::shared_ptr<ChangeNodeTypeSet>>(operators, "ChangeNodeTypeSet")
        .def(py::init<FactorStringTypeVector>(), 
                 py::arg("type_whitelist") = FactorStringTypeVector());

    py::class_<OperatorSetType>(operators, "OperatorSetType")
        .def_property_readonly_static("ARCS", [](const py::object&) { 
            return OperatorSetType(OperatorSetType::ARCS);
        })
        .def_property_readonly_static("NODE_TYPE", [](const py::object&) { 
            return OperatorSetType(OperatorSetType::NODE_TYPE);
        })
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<OperatorPool, OperatorSet, std::shared_ptr<OperatorPool>>(operators, "OperatorPool")
        .def(py::init<std::vector<std::shared_ptr<OperatorSet>>>());
}
