#include <pybind11/operators.h>
#include <models/BayesianNetwork.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <models/DiscreteBN.hpp>


using models::BayesianNetworkBase, models::BayesianNetwork, models::BayesianNetworkType, 
      models::GaussianNetwork, models::SemiparametricBN, models::DiscreteBN;

template<typename DerivedBN>
py::class_<DerivedBN, BayesianNetwork<DerivedBN>> register_BayesianNetwork(py::module& m, const char* derivedbn_name) {
    using BaseClass = BayesianNetwork<DerivedBN>;
    std::string base_name = std::string("BayesianNetwork<") + derivedbn_name + ">";
    // TODO: Implement copy operation.
    py::class_<BaseClass, BayesianNetworkBase>(m, base_name.c_str())
        .def("num_nodes", &BaseClass::num_nodes)
        .def("num_arcs", &BaseClass::num_arcs)
        .def("nodes", &BaseClass::nodes, py::return_value_policy::take_ownership)
        .def("arcs", &BaseClass::arcs, py::return_value_policy::take_ownership)
        .def("indices", &BaseClass::indices, py::return_value_policy::reference_internal)
        .def("index", py::overload_cast<const std::string&>(&BaseClass::index, py::const_))
        .def("contains_node", &BaseClass::contains_node)
        .def("name", py::overload_cast<int>(&BaseClass::name, py::const_), py::return_value_policy::reference_internal)
        .def("num_parents", py::overload_cast<const std::string&>(&BaseClass::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&BaseClass::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&BaseClass::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&BaseClass::num_children, py::const_))
        .def("parents", py::overload_cast<const std::string&>(&BaseClass::parents, py::const_), py::return_value_policy::take_ownership)
        .def("parents", py::overload_cast<int>(&BaseClass::parents, py::const_), py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<const std::string&>(&BaseClass::parent_indices, py::const_), py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<int>(&BaseClass::parent_indices, py::const_), py::return_value_policy::take_ownership)
        .def("has_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::has_arc, py::const_))
        .def("has_arc", py::overload_cast<int, int>(&BaseClass::has_arc, py::const_))
        .def("has_path", py::overload_cast<const std::string&, const std::string&>(&BaseClass::has_path, py::const_))
        .def("has_path", py::overload_cast<int, int>(&BaseClass::has_path, py::const_))
        .def("add_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::add_arc))
        .def("add_arc", py::overload_cast<int, int>(&BaseClass::add_arc))
        .def("can_add_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::can_add_arc, py::const_))
        .def("can_add_arc", py::overload_cast<int, int>(&BaseClass::can_add_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::can_flip_arc))
        .def("can_flip_arc", py::overload_cast<int, int>(&BaseClass::can_flip_arc))
        .def("remove_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::remove_arc))
        .def("remove_arc", py::overload_cast<int, int>(&BaseClass::remove_arc))
        .def("fit", &BaseClass::fit)
        .def("add_cpds", &BaseClass::add_cpds)
        .def("cpd", py::overload_cast<const std::string&>(&BaseClass::cpd), py::return_value_policy::reference_internal)
        .def("cpd", py::overload_cast<int>(&BaseClass::cpd), py::return_value_policy::reference_internal)
        .def("logl", &BaseClass::logl, py::return_value_policy::take_ownership)
        .def("slogl", &BaseClass::slogl);

    return py::class_<DerivedBN, BaseClass>(m, derivedbn_name)
            .def(py::init<const std::vector<std::string>&>())
            .def(py::init<const ArcVector&>())
            .def(py::init<const std::vector<std::string>&, const ArcVector&>());
}

void pybindings_models(py::module& root) {
    auto models = root.def_submodule("models", "Models submodule.");

    py::class_<BayesianNetworkType>(models, "BayesianNetworkType")
        .def_property_readonly_static("GBN", [](const py::object&) { 
            return BayesianNetworkType(BayesianNetworkType::GBN);
        })
        .def_property_readonly_static("DISCRETEBN", [](const py::object&) { 
            return BayesianNetworkType(BayesianNetworkType::DISCRETEBN);
        })
        .def_property_readonly_static("SPBN", [](const py::object&) { 
            return BayesianNetworkType(BayesianNetworkType::SPBN);
        })
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<BayesianNetworkBase>(models, "BayesianNetworkBase")
        .def("num_nodes", &BayesianNetworkBase::num_nodes)
        .def("num_arcs", &BayesianNetworkBase::num_arcs)
        .def("nodes", &BayesianNetworkBase::nodes, py::return_value_policy::take_ownership)
        .def("arcs", &BayesianNetworkBase::arcs, py::return_value_policy::take_ownership)
        .def("indices", &BayesianNetworkBase::indices, py::return_value_policy::reference_internal)
        .def("index", py::overload_cast<const std::string&>(&BayesianNetworkBase::index, py::const_))
        .def("contains_node", &BayesianNetworkBase::contains_node)
        .def("name", py::overload_cast<int>(&BayesianNetworkBase::name, py::const_), py::return_value_policy::reference_internal)
        .def("num_parents", py::overload_cast<const std::string&>(&BayesianNetworkBase::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&BayesianNetworkBase::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&BayesianNetworkBase::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&BayesianNetworkBase::num_children, py::const_))
        .def("parents", py::overload_cast<const std::string&>(&BayesianNetworkBase::parents, py::const_), py::return_value_policy::take_ownership)
        .def("parents", py::overload_cast<int>(&BayesianNetworkBase::parents, py::const_), py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<const std::string&>(&BayesianNetworkBase::parent_indices, py::const_), py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<int>(&BayesianNetworkBase::parent_indices, py::const_), py::return_value_policy::take_ownership)
        .def("has_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::has_arc, py::const_))
        .def("has_arc", py::overload_cast<int, int>(&BayesianNetworkBase::has_arc, py::const_))
        .def("has_path", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::has_path, py::const_))
        .def("has_path", py::overload_cast<int, int>(&BayesianNetworkBase::has_path, py::const_))
        .def("add_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::add_arc))
        .def("add_arc", py::overload_cast<int, int>(&BayesianNetworkBase::add_arc))
        .def("can_add_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::can_add_arc, py::const_))
        .def("can_add_arc", py::overload_cast<int, int>(&BayesianNetworkBase::can_add_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::can_flip_arc))
        .def("can_flip_arc", py::overload_cast<int, int>(&BayesianNetworkBase::can_flip_arc))
        .def("remove_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::remove_arc))
        .def("remove_arc", py::overload_cast<int, int>(&BayesianNetworkBase::remove_arc))
        .def("fit", &BayesianNetworkBase::fit)
        .def("logl", &BayesianNetworkBase::logl, py::return_value_policy::take_ownership)
        .def("slogl", &BayesianNetworkBase::slogl);

    register_BayesianNetwork<GaussianNetwork>(models, "GaussianNetwork");
    
    auto spbn = register_BayesianNetwork<SemiparametricBN>(models, "SemiparametricBN");

    spbn.def(py::init<const std::vector<std::string>&, FactorTypeVector&>())
        .def(py::init<const ArcVector&, FactorTypeVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcVector&, FactorTypeVector&>())
        .def("node_type", py::overload_cast<const std::string&>(&SemiparametricBN::node_type, py::const_))
        .def("node_type", py::overload_cast<int>(&SemiparametricBN::node_type, py::const_))
        .def("set_node_type", py::overload_cast<const std::string&, FactorType>(&SemiparametricBN::set_node_type))
        .def("set_node_type", py::overload_cast<int, FactorType>(&SemiparametricBN::set_node_type));

    register_BayesianNetwork<DiscreteBN>(models, "DiscreteBN");
}