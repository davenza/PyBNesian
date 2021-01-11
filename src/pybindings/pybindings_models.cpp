#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <models/BayesianNetwork.hpp>
#include <models/ConditionalBayesianNetwork.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <models/DiscreteBN.hpp>

using models::BayesianNetworkBase, models::BayesianNetwork, models::BayesianNetworkType, 
      models::GaussianNetwork, models::SemiparametricBN, models::DiscreteBN;

using models::ConditionalBayesianNetworkBase, models::ConditionalBayesianNetwork,
      models::ConditionalGaussianNetwork, models::ConditionalSemiparametricBN, 
      models::ConditionalDiscreteBN;

template<typename DerivedBN>
py::class_<DerivedBN, BayesianNetwork<DerivedBN>> register_BayesianNetwork(py::module& m, const char* derivedbn_name) {
    using BaseClass = BayesianNetwork<DerivedBN>;
    std::string base_name = std::string("BayesianNetwork<") + derivedbn_name + ">";
    // TODO: Implement copy operation.
    py::class_<BaseClass, BayesianNetworkBase>(m, base_name.c_str())
        .def("num_nodes", &BaseClass::num_nodes)
        .def("num_arcs", &BaseClass::num_arcs)
        .def("nodes", &BaseClass::nodes, py::return_value_policy::reference_internal)
        .def("arcs", &BaseClass::arcs, py::return_value_policy::take_ownership)
        .def("indices", &BaseClass::indices, py::return_value_policy::reference_internal)
        .def("index", py::overload_cast<const std::string&>(&BaseClass::index, py::const_))
        .def("is_valid", &BaseClass::is_valid)
        .def("contains_node", &BaseClass::contains_node)
        .def("add_node", &BaseClass::add_node)
        .def("remove_node", py::overload_cast<int>(&BaseClass::remove_node))
        .def("remove_node", py::overload_cast<const std::string&>(&BaseClass::remove_node))
        .def("name", py::overload_cast<int>(&BaseClass::name, py::const_), 
                                                    py::return_value_policy::reference_internal)
        .def("num_parents", py::overload_cast<const std::string&>(&BaseClass::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&BaseClass::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&BaseClass::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&BaseClass::num_children, py::const_))
        .def("parents", py::overload_cast<const std::string&>(&BaseClass::parents, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("parents", py::overload_cast<int>(&BaseClass::parents, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<const std::string&>(&BaseClass::parent_indices, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<int>(&BaseClass::parent_indices, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("children", py::overload_cast<const std::string&>(&BaseClass::children, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children", py::overload_cast<int>(&BaseClass::children, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children_indices", py::overload_cast<const std::string&>(&BaseClass::children_indices, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children_indices", py::overload_cast<int>(&BaseClass::children_indices, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("has_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::has_arc, py::const_))
        .def("has_arc", py::overload_cast<int, int>(&BaseClass::has_arc, py::const_))
        .def("has_path", py::overload_cast<const std::string&, const std::string&>(&BaseClass::has_path, py::const_))
        .def("has_path", py::overload_cast<int, int>(&BaseClass::has_path, py::const_))
        .def("add_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::add_arc))
        .def("add_arc", py::overload_cast<int, int>(&BaseClass::add_arc))
        .def("remove_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::remove_arc))
        .def("remove_arc", py::overload_cast<int, int>(&BaseClass::remove_arc))
        .def("flip_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::flip_arc))
        .def("flip_arc", py::overload_cast<int, int>(&BaseClass::flip_arc))
        .def("can_add_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::can_add_arc, py::const_))
        .def("can_add_arc", py::overload_cast<int, int>(&BaseClass::can_add_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::can_flip_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<int, int>(&BaseClass::can_flip_arc, py::const_))
        .def("check_blacklist", py::overload_cast<const ArcStringVector&>(&BayesianNetworkBase::check_blacklist, py::const_))
        .def("force_whitelist", py::overload_cast<const ArcStringVector&>(&BayesianNetworkBase::force_whitelist))
        .def("fitted", &BaseClass::fitted)
        .def("add_cpds", &BaseClass::add_cpds)
        .def("fit", &BaseClass::fit)
        .def("cpd", py::overload_cast<const std::string&>(&BaseClass::cpd), py::return_value_policy::reference_internal)
        .def("cpd", py::overload_cast<int>(&BaseClass::cpd), py::return_value_policy::reference_internal)
        .def("logl", &BaseClass::logl, py::return_value_policy::take_ownership)
        .def("slogl", &BaseClass::slogl)
        .def("sample", &BaseClass::sample, py::return_value_policy::move)
        .def("save", &BaseClass::save, py::arg("name"), py::arg("include_cpd") = false);


    return py::class_<DerivedBN, BaseClass>(m, derivedbn_name)
        .def(py::init<const std::vector<std::string>&>())
        .def(py::init<const ArcStringVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcStringVector&>())
        .def(py::init<const Dag&>())
        .def(py::pickle(
            [](const DerivedBN& self) {
                return self.__getstate__();
            },
            [](py::tuple t) {
                return DerivedBN::__setstate__(t);
            }
        ));
}

template<typename DerivedBN>
py::class_<DerivedBN, ConditionalBayesianNetwork<DerivedBN>> register_ConditionalBayesianNetwork(py::module& m, const char* derivedbn_name) {
    using BaseClass = ConditionalBayesianNetwork<DerivedBN>;
    std::string base_name = std::string("ConditionalBayesianNetwork<") + derivedbn_name + ">";
    // TODO: Implement copy operation.
    py::class_<BaseClass, ConditionalBayesianNetworkBase>(m, base_name.c_str())
        .def("num_nodes", &BaseClass::num_nodes)
        .def("num_interface_nodes", &BaseClass::num_interface_nodes)
        .def("num_total_nodes", &BaseClass::num_total_nodes)
        .def("num_arcs", &BaseClass::num_arcs)
        .def("nodes", &BaseClass::nodes, py::return_value_policy::reference_internal)
        .def("interface_nodes", &BaseClass::interface_nodes, py::return_value_policy::reference_internal)
        .def("all_nodes", &BaseClass::all_nodes, py::return_value_policy::reference_internal)
        .def("arcs", &BaseClass::arcs, py::return_value_policy::take_ownership)
        .def("indices", &BaseClass::indices, py::return_value_policy::reference_internal)
        .def("interface_indices", &BaseClass::interface_indices, py::return_value_policy::reference_internal)
        .def("index", py::overload_cast<const std::string&>(&BaseClass::index, py::const_))
        .def("is_valid", &BaseClass::is_valid)
        .def("is_interface", py::overload_cast<int>(&BaseClass::is_interface, py::const_))
        .def("is_interface", py::overload_cast<const std::string&>(&BaseClass::is_interface, py::const_))
        .def("contains_node", &BaseClass::contains_node)
        .def("contains_interface_node", &BaseClass::contains_interface_node)
        .def("contains_all_node", &BaseClass::contains_all_node)
        .def("add_node", &BaseClass::add_node)
        .def("add_interface_node", &BaseClass::add_interface_node)
        .def("remove_node", py::overload_cast<int>(&BaseClass::remove_node))
        .def("remove_node", py::overload_cast<const std::string&>(&BaseClass::remove_node))
        .def("remove_interface_node", py::overload_cast<int>(&BaseClass::remove_interface_node))
        .def("remove_interface_node", py::overload_cast<const std::string&>(&BaseClass::remove_interface_node))
        .def("name", py::overload_cast<int>(&BaseClass::name, py::const_), 
                                                    py::return_value_policy::reference_internal)
        .def("num_parents", py::overload_cast<const std::string&>(&BaseClass::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&BaseClass::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&BaseClass::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&BaseClass::num_children, py::const_))
        .def("parents", py::overload_cast<const std::string&>(&BaseClass::parents, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("parents", py::overload_cast<int>(&BaseClass::parents, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<const std::string&>(&BaseClass::parent_indices, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<int>(&BaseClass::parent_indices, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("children", py::overload_cast<const std::string&>(&BaseClass::children, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children", py::overload_cast<int>(&BaseClass::children, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children_indices", py::overload_cast<const std::string&>(&BaseClass::children_indices, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children_indices", py::overload_cast<int>(&BaseClass::children_indices, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("has_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::has_arc, py::const_))
        .def("has_arc", py::overload_cast<int, int>(&BaseClass::has_arc, py::const_))
        .def("has_path", py::overload_cast<const std::string&, const std::string&>(&BaseClass::has_path, py::const_))
        .def("has_path", py::overload_cast<int, int>(&BaseClass::has_path, py::const_))
        .def("add_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::add_arc))
        .def("add_arc", py::overload_cast<int, int>(&BaseClass::add_arc))
        .def("remove_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::remove_arc))
        .def("remove_arc", py::overload_cast<int, int>(&BaseClass::remove_arc))
        .def("flip_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::flip_arc))
        .def("flip_arc", py::overload_cast<int, int>(&BaseClass::flip_arc))
        .def("can_add_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::can_add_arc, py::const_))
        .def("can_add_arc", py::overload_cast<int, int>(&BaseClass::can_add_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<const std::string&, const std::string&>(&BaseClass::can_flip_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<int, int>(&BaseClass::can_flip_arc, py::const_))
        .def("check_blacklist", py::overload_cast<const ArcStringVector&>(&BayesianNetworkBase::check_blacklist, py::const_))
        .def("force_whitelist", py::overload_cast<const ArcStringVector&>(&BayesianNetworkBase::force_whitelist))
        .def("fitted", &BaseClass::fitted)
        .def("add_cpds", &BaseClass::add_cpds)
        .def("fit", &BaseClass::fit)
        .def("cpd", py::overload_cast<const std::string&>(&BaseClass::cpd), py::return_value_policy::reference_internal)
        .def("cpd", py::overload_cast<int>(&BaseClass::cpd), py::return_value_policy::reference_internal)
        .def("logl", &BaseClass::logl, py::return_value_policy::take_ownership)
        .def("slogl", &BaseClass::slogl)
        .def("sample", [](const BaseClass& self,
                          const DataFrame& evidence,
                          bool concat_evidence,
                          bool ordered) {
                return self.sample(evidence, std::random_device{}(), concat_evidence, ordered);
        }, py::return_value_policy::move, 
           py::arg("evidence"),
           py::arg("concat_evidence") = false,
           py::arg("ordered") = false)
        .def("sample", py::overload_cast<const DataFrame&,
                                         unsigned int,
                                         bool,
                                         bool>(&BaseClass::sample, py::const_),
                        py::return_value_policy::move, 
                        py::arg("evidence"),
                        py::arg("seed"),
                        py::arg("concat_evidence") = false,
                        py::arg("ordered") = false)
        .def("save", &BaseClass::save, py::arg("name"), py::arg("include_cpd") = false);


    return py::class_<DerivedBN, BaseClass>(m, derivedbn_name)
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>())
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const ArcStringVector&>())
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const Dag&>())
        .def(py::pickle(
            [](const DerivedBN& self) {
                return self.__getstate__();
            },
            [](py::tuple t) {
                return DerivedBN::__setstate__(t);
            }
        ));
}


void pybindings_models(py::module& root) {
    auto models = root.def_submodule("models", "Models submodule.");

    models.def("load_model", &models::load_model);

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
        .def_property_readonly("type", &BayesianNetworkBase::type)
        .def("num_nodes", &BayesianNetworkBase::num_nodes)
        .def("num_arcs", &BayesianNetworkBase::num_arcs)
        .def("nodes", &BayesianNetworkBase::nodes, py::return_value_policy::reference_internal)
        .def("arcs", &BayesianNetworkBase::arcs, py::return_value_policy::take_ownership)
        .def("indices", &BayesianNetworkBase::indices, py::return_value_policy::reference_internal)
        .def("index", py::overload_cast<const std::string&>(&BayesianNetworkBase::index, py::const_))
        .def("is_valid", &BayesianNetworkBase::is_valid)
        .def("contains_node", &BayesianNetworkBase::contains_node)
        .def("add_node", &BayesianNetworkBase::add_node)
        .def("remove_node", py::overload_cast<int>(&BayesianNetworkBase::remove_node))
        .def("remove_node", py::overload_cast<const std::string&>(&BayesianNetworkBase::remove_node))
        .def("name", py::overload_cast<int>(&BayesianNetworkBase::name, py::const_), 
                                                            py::return_value_policy::reference_internal)
        .def("num_parents", py::overload_cast<const std::string&>(&BayesianNetworkBase::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&BayesianNetworkBase::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&BayesianNetworkBase::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&BayesianNetworkBase::num_children, py::const_))
        .def("parents", py::overload_cast<const std::string&>(&BayesianNetworkBase::parents, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("parents", py::overload_cast<int>(&BayesianNetworkBase::parents, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<const std::string&>(&BayesianNetworkBase::parent_indices, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<int>(&BayesianNetworkBase::parent_indices, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children", py::overload_cast<const std::string&>(&BayesianNetworkBase::children, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children", py::overload_cast<int>(&BayesianNetworkBase::children, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children_indices", py::overload_cast<const std::string&>(&BayesianNetworkBase::children_indices, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children_indices", py::overload_cast<int>(&BayesianNetworkBase::children_indices, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("has_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::has_arc, py::const_))
        .def("has_arc", py::overload_cast<int, int>(&BayesianNetworkBase::has_arc, py::const_))
        .def("has_path", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::has_path, py::const_))
        .def("has_path", py::overload_cast<int, int>(&BayesianNetworkBase::has_path, py::const_))
        .def("add_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::add_arc))
        .def("add_arc", py::overload_cast<int, int>(&BayesianNetworkBase::add_arc))
        .def("remove_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::remove_arc))
        .def("remove_arc", py::overload_cast<int, int>(&BayesianNetworkBase::remove_arc))
        .def("flip_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::flip_arc))
        .def("flip_arc", py::overload_cast<int, int>(&BayesianNetworkBase::flip_arc))
        .def("can_add_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::can_add_arc, py::const_))
        .def("can_add_arc", py::overload_cast<int, int>(&BayesianNetworkBase::can_add_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::can_flip_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<int, int>(&BayesianNetworkBase::can_flip_arc, py::const_))
        .def("fit", &BayesianNetworkBase::fit)
        .def("logl", &BayesianNetworkBase::logl, py::return_value_policy::take_ownership)
        .def("slogl", &BayesianNetworkBase::slogl)
        .def("sample", [](const BayesianNetworkBase& self, int n, bool ordered) {
                return self.sample(n, std::random_device{}(), ordered);
        }, py::return_value_policy::move, py::arg("n"), py::arg("ordered") = false)
        .def("sample", &BayesianNetworkBase::sample, py::return_value_policy::move, 
                py::arg("n"), py::arg("seed"), py::arg("ordered") = false)
        .def("save", &BayesianNetworkBase::save)
        .def("clone", &BayesianNetworkBase::clone);

    py::class_<ConditionalBayesianNetworkBase, BayesianNetworkBase>(models, "ConditionalBayesianNetworkBase")
        .def("num_interface_nodes", &ConditionalBayesianNetworkBase::num_interface_nodes)
        .def("num_total_nodes", &ConditionalBayesianNetworkBase::num_total_nodes)
        .def("interface_nodes", &ConditionalBayesianNetworkBase::interface_nodes, py::return_value_policy::reference_internal)
        .def("all_nodes", &ConditionalBayesianNetworkBase::all_nodes, py::return_value_policy::reference_internal)
        .def("interface_indices", &ConditionalBayesianNetworkBase::interface_indices, py::return_value_policy::reference_internal)
        .def("contains_interface_node", &ConditionalBayesianNetworkBase::contains_interface_node)
        .def("contains_all_node", &ConditionalBayesianNetworkBase::contains_all_node)
        .def("add_interface_node", &ConditionalBayesianNetworkBase::add_interface_node)
        .def("remove_interface_node", py::overload_cast<int>(&ConditionalBayesianNetworkBase::remove_interface_node))
        .def("remove_interface_node", py::overload_cast<const std::string&>(&ConditionalBayesianNetworkBase::remove_interface_node))
        .def("is_interface", py::overload_cast<int>(&ConditionalBayesianNetworkBase::is_interface, py::const_))
        .def("is_interface", py::overload_cast<const std::string&>(&ConditionalBayesianNetworkBase::is_interface, py::const_))
        .def("sample", [](const ConditionalBayesianNetworkBase& self,
                          const DataFrame& evidence,
                          bool concat_evidence,
                          bool ordered) {
                return self.sample(evidence, std::random_device{}(), concat_evidence, ordered);
        }, py::return_value_policy::move, 
           py::arg("evidence"),
           py::arg("concat_evidence") = false,
           py::arg("ordered") = false)
        .def("sample", py::overload_cast<const DataFrame&,
                                         unsigned int,
                                         bool,
                                         bool>(&ConditionalBayesianNetworkBase::sample, py::const_),
                        py::return_value_policy::move, 
                        py::arg("evidence"),
                        py::arg("seed"),
                        py::arg("concat_evidence") = false,
                        py::arg("ordered") = false);

    register_BayesianNetwork<GaussianNetwork>(models, "GaussianNetwork");
    
    auto spbn = register_BayesianNetwork<SemiparametricBN>(models, "SemiparametricBN");

    spbn.def(py::init<const std::vector<std::string>&, FactorStringTypeVector&>())
        .def(py::init<const ArcStringVector&, FactorStringTypeVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcStringVector&, FactorStringTypeVector&>())
        .def(py::init<const Dag&, FactorStringTypeVector&>())
        .def("node_type", py::overload_cast<const std::string&>(&SemiparametricBN::node_type, py::const_))
        .def("node_type", py::overload_cast<int>(&SemiparametricBN::node_type, py::const_))
        .def("set_node_type", py::overload_cast<const std::string&, FactorType>(&SemiparametricBN::set_node_type))
        .def("set_node_type", py::overload_cast<int, FactorType>(&SemiparametricBN::set_node_type))
        .def("node_types", &SemiparametricBN::node_types);

    register_BayesianNetwork<DiscreteBN>(models, "DiscreteBN");


    register_ConditionalBayesianNetwork<ConditionalGaussianNetwork>(models, "ConditionalGaussianNetwork");

    auto conditional_spbn = register_ConditionalBayesianNetwork<ConditionalSemiparametricBN>(models, "ConditionalSemiparametricBN");

    conditional_spbn.def(py::init<const std::vector<std::string>&,
                                  const std::vector<std::string>&,
                                  FactorStringTypeVector&>())
                    .def(py::init<const std::vector<std::string>&,
                                  const std::vector<std::string>&,
                                  const ArcStringVector&,
                                  FactorStringTypeVector&>())
                    .def(py::init<const std::vector<std::string>&,
                                  const std::vector<std::string>&,
                                  const Dag&,
                                  FactorStringTypeVector&>())
                    .def("node_type", py::overload_cast<const std::string&>(&ConditionalSemiparametricBN::node_type, py::const_))
                    .def("node_type", py::overload_cast<int>(&ConditionalSemiparametricBN::node_type, py::const_))
                    .def("set_node_type", py::overload_cast<const std::string&, FactorType>(&ConditionalSemiparametricBN::set_node_type))
                    .def("set_node_type", py::overload_cast<int, FactorType>(&ConditionalSemiparametricBN::set_node_type))
                    .def("node_types", &ConditionalSemiparametricBN::node_types);
    
    register_ConditionalBayesianNetwork<ConditionalDiscreteBN>(models, "ConditionalDiscreteBN");
}