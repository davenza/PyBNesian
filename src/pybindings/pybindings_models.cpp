#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <models/DiscreteBN.hpp>

using models::BayesianNetworkType, models::BayesianNetworkBase, models::BayesianNetworkImpl,
      models::BayesianNetwork, models::GaussianNetwork, models::SemiparametricBN, models::DiscreteBN;

using models::ConditionalBayesianNetworkBase, models::ConditionalBayesianNetworkImpl,
      models::ConditionalBayesianNetwork, models::ConditionalGaussianNetwork, 
      models::ConditionalSemiparametricBN, models::ConditionalDiscreteBN;

using models::DynamicBayesianNetworkBase, models::DynamicBayesianNetworkImpl,
      models::DynamicGaussianNetwork, models::DynamicSemiparametricBN, models::DynamicDiscreteBN;


template<typename Derived, typename BaseClass>
py::class_<BayesianNetworkImpl<Derived>, BaseClass>
register_BayesianNetworkImpl(py::module& m, const char* derivedbn_name) {
    std::string impl_name = std::string("BayesianNetworkImpl<") + derivedbn_name + ">";
    
    using ImplClass = BayesianNetworkImpl<Derived>;

    return py::class_<ImplClass, BaseClass>(m, impl_name.c_str())
        .def_property_readonly("type", &ImplClass::type)
        .def("num_nodes", &ImplClass::num_nodes)
        .def("num_arcs", &ImplClass::num_arcs)
        .def("nodes", &ImplClass::nodes, py::return_value_policy::reference_internal)
        .def("arcs", &ImplClass::arcs, py::return_value_policy::take_ownership)
        .def("indices", &ImplClass::indices, py::return_value_policy::reference_internal)
        .def("index", &ImplClass::index)
        .def("collapsed_index", &ImplClass::collapsed_index)
        .def("index_from_collapsed", &ImplClass::index_from_collapsed)
        .def("collapsed_from_index", &ImplClass::collapsed_from_index)
        .def("collapsed_indices", &ImplClass::collapsed_indices, py::return_value_policy::reference_internal)
        .def("is_valid", &ImplClass::is_valid)
        .def("contains_node", &ImplClass::contains_node)
        .def("add_node", &ImplClass::add_node)
        .def("remove_node", py::overload_cast<int>(&ImplClass::remove_node))
        .def("remove_node", py::overload_cast<const std::string&>(&ImplClass::remove_node))
        .def("name", &ImplClass::name)
        .def("collapsed_name", &ImplClass::collapsed_name)
        .def("num_parents", py::overload_cast<const std::string&>(&ImplClass::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&ImplClass::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&ImplClass::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&ImplClass::num_children, py::const_))
        .def("parents", py::overload_cast<const std::string&>(&ImplClass::parents, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("parents", py::overload_cast<int>(&ImplClass::parents, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<const std::string&>(&ImplClass::parent_indices, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<int>(&ImplClass::parent_indices, py::const_), 
                                                    py::return_value_policy::take_ownership)
        .def("children", py::overload_cast<const std::string&>(&ImplClass::children, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children", py::overload_cast<int>(&ImplClass::children, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children_indices", py::overload_cast<const std::string&>(&ImplClass::children_indices, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("children_indices", py::overload_cast<int>(&ImplClass::children_indices, py::const_), 
                                                            py::return_value_policy::take_ownership)
        .def("has_arc", py::overload_cast<const std::string&, const std::string&>(&ImplClass::has_arc, py::const_))
        .def("has_arc", py::overload_cast<int, int>(&ImplClass::has_arc, py::const_))
        .def("has_path", py::overload_cast<const std::string&, const std::string&>(&ImplClass::has_path, py::const_))
        .def("has_path", py::overload_cast<int, int>(&ImplClass::has_path, py::const_))
        .def("add_arc", py::overload_cast<const std::string&, const std::string&>(&ImplClass::add_arc))
        .def("add_arc", py::overload_cast<int, int>(&ImplClass::add_arc))
        .def("remove_arc", py::overload_cast<const std::string&, const std::string&>(&ImplClass::remove_arc))
        .def("remove_arc", py::overload_cast<int, int>(&ImplClass::remove_arc))
        .def("flip_arc", py::overload_cast<const std::string&, const std::string&>(&ImplClass::flip_arc))
        .def("flip_arc", py::overload_cast<int, int>(&ImplClass::flip_arc))
        .def("can_add_arc", py::overload_cast<const std::string&, const std::string&>(&ImplClass::can_add_arc, py::const_))
        .def("can_add_arc", py::overload_cast<int, int>(&ImplClass::can_add_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<const std::string&, const std::string&>(&ImplClass::can_flip_arc, py::const_))
        .def("can_flip_arc", py::overload_cast<int, int>(&ImplClass::can_flip_arc, py::const_))
        .def("force_whitelist", py::overload_cast<const ArcStringVector&>(&BayesianNetworkBase::force_whitelist))
        .def("fitted", &ImplClass::fitted)
        .def("add_cpds", &ImplClass::add_cpds)
        .def("fit", &ImplClass::fit)
        .def("cpd", py::overload_cast<const std::string&>(&ImplClass::cpd), py::return_value_policy::reference_internal)
        .def("cpd", py::overload_cast<int>(&ImplClass::cpd), py::return_value_policy::reference_internal)
        .def("logl", &ImplClass::logl, py::return_value_policy::take_ownership)
        .def("slogl", &ImplClass::slogl)
        .def("sample", [](const ImplClass& self, int n, bool ordered) {
                return self.sample(n, std::random_device{}(), ordered);
        }, py::return_value_policy::move, py::arg("n"), py::arg("ordered") = false)
        .def("sample", py::overload_cast<int, unsigned int, bool>(&ImplClass::sample, py::const_), 
                       py::return_value_policy::move, 
                       py::arg("n"),
                       py::arg("seed"),
                       py::arg("ordered") = false)
        .def("conditional_bn", py::overload_cast<const std::vector<std::string>&, 
                                                 const std::vector<std::string>&>(&ImplClass::conditional_bn, py::const_))
        .def("conditional_bn", py::overload_cast<>(&ImplClass::conditional_bn, py::const_))
        .def("unconditional_bn", &ImplClass::unconditional_bn)
        .def("save", &ImplClass::save, py::arg("name"), py::arg("include_cpd") = false);
}

template<typename Derived>
py::class_<ConditionalBayesianNetworkImpl<Derived>, BayesianNetworkImpl<Derived>>
register_ConditionalBayesianNetworkImpl(py::module& m, const char* derivedbn_name) {
    std::string impl_name = std::string("ConditionalBayesianNetworkImpl<") + derivedbn_name + ">";
    
    using BaseImplClass = BayesianNetworkImpl<Derived>;
    using ImplClass = ConditionalBayesianNetworkImpl<Derived>;

    return py::class_<ImplClass, BaseImplClass>(m, impl_name.c_str())
        .def("num_interface_nodes", &ImplClass::num_interface_nodes)
        .def("num_total_nodes", &ImplClass::num_total_nodes)
        .def("interface_nodes", &ImplClass::interface_nodes, py::return_value_policy::reference_internal)
        .def("all_nodes", &ImplClass::all_nodes, py::return_value_policy::reference_internal)
        .def("interface_collapsed_index", &ImplClass::interface_collapsed_index)
        .def("joint_collapsed_index", &ImplClass::joint_collapsed_index)
        .def("interface_collapsed_indices", &ImplClass::interface_collapsed_indices, py::return_value_policy::reference_internal)
        .def("joint_collapsed_indices", &ImplClass::joint_collapsed_indices, py::return_value_policy::reference_internal)
        .def("index_from_interface_collapsed", &ImplClass::index_from_interface_collapsed)
        .def("index_from_joint_collapsed", &ImplClass::index_from_joint_collapsed)
        .def("interface_collapsed_from_index", &ImplClass::interface_collapsed_from_index)
        .def("joint_collapsed_from_index", &ImplClass::joint_collapsed_from_index)
        .def("interface_collapsed_name", &ImplClass::interface_collapsed_name)
        .def("joint_collapsed_name", &ImplClass::joint_collapsed_name)
        .def("contains_interface_node", &ImplClass::contains_interface_node)
        .def("contains_total_node", &ImplClass::contains_total_node)
        .def("add_interface_node", &ImplClass::add_interface_node)
        .def("remove_interface_node", py::overload_cast<int>(&ImplClass::remove_interface_node))
        .def("remove_interface_node", py::overload_cast<const std::string&>(&ImplClass::remove_interface_node))
        .def("is_interface", py::overload_cast<int>(&ImplClass::is_interface, py::const_))
        .def("is_interface", py::overload_cast<const std::string&>(&ImplClass::is_interface, py::const_))
        .def("set_interface", py::overload_cast<int>(&ImplClass::set_interface))
        .def("set_interface", py::overload_cast<const std::string&>(&ImplClass::set_interface))
        .def("set_node", py::overload_cast<int>(&ImplClass::set_node))
        .def("set_node", py::overload_cast<const std::string&>(&ImplClass::set_node))
        .def("sample", [](const ImplClass& self,
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
                                         bool>(&ImplClass::sample, py::const_),
                        py::return_value_policy::move, 
                        py::arg("evidence"),
                        py::arg("seed"),
                        py::arg("concat_evidence") = false,
                        py::arg("ordered") = false);
}

template<typename Derived>
py::class_<DynamicBayesianNetworkImpl<Derived>, DynamicBayesianNetworkBase>
register_DynamicBayesianNetworkImpl(py::module& m, const char* derivedbn_name) {
    std::string impl_name = std::string("DynamicBayesianNetworkImpl<") + derivedbn_name + ">";
    using ImplClass = DynamicBayesianNetworkImpl<Derived>;

    return py::class_<ImplClass, DynamicBayesianNetworkBase>(m, impl_name.c_str())
        .def_property_readonly("type", &ImplClass::type)
        .def("static_bn", &ImplClass::static_bn, py::return_value_policy::reference_internal)
        .def("transition_bn", &ImplClass::transition_bn, py::return_value_policy::reference_internal)
        .def("markovian_order", &ImplClass::markovian_order)
        .def("num_variables", &ImplClass::num_variables)
        .def("variables", &ImplClass::variables)
        .def("contains_variable", &ImplClass::contains_variable)
        .def("add_variable", &ImplClass::add_variable)
        .def("remove_variable", &ImplClass::remove_variable)
        .def("fitted", &ImplClass::fitted)
        .def("fit", &ImplClass::fit)
        .def("logl", &ImplClass::logl)
        .def("slogl", &ImplClass::slogl)
        .def("sample", &ImplClass::sample)
        .def("save", &ImplClass::save);
}

template<typename DerivedBN>
py::class_<DerivedBN, BayesianNetworkImpl<DerivedBN>> 
register_BayesianNetwork(py::module& m, const char* derivedbn_name) {
    register_BayesianNetworkImpl<DerivedBN, BayesianNetworkBase>(m, derivedbn_name);

    using BaseImpl = BayesianNetworkImpl<DerivedBN>;
    return py::class_<DerivedBN, BaseImpl>(m, derivedbn_name)
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
py::class_<DerivedBN, ConditionalBayesianNetworkImpl<DerivedBN>> 
register_ConditionalBayesianNetwork(py::module& m, const char* derivedbn_name) {
    register_BayesianNetworkImpl<DerivedBN, ConditionalBayesianNetworkBase>(m, derivedbn_name);
    register_ConditionalBayesianNetworkImpl<DerivedBN>(m, derivedbn_name);

    using BaseImpl = ConditionalBayesianNetworkImpl<DerivedBN>;
    return py::class_<DerivedBN, BaseImpl>(m, derivedbn_name)
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>())
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const ArcStringVector&>())
        .def(py::init<const ConditionalDag&>())
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
py::class_<DerivedBN, DynamicBayesianNetworkImpl<DerivedBN>> 
register_DynamicBayesianNetwork(py::module& m, const char* derivedbn_name) {

    register_DynamicBayesianNetworkImpl<DerivedBN>(m, derivedbn_name);

    using BaseImpl = DynamicBayesianNetworkImpl<DerivedBN>;
    using NormalBN = BayesianNetwork<DerivedBN::TYPE>;
    using ConditionalBN = ConditionalBayesianNetwork<DerivedBN::TYPE>;

    return py::class_<DerivedBN, BaseImpl>(m, derivedbn_name)
        .def(py::init<const std::vector<std::string>&, int>())
        .def(py::init<const std::vector<std::string>&, int, NormalBN, ConditionalBN>());
}


void pybindings_models(py::module& root) {
    auto models = root.def_submodule("models", "Models submodule.");

    models.def("load_model", &models::load_model);

    py::class_<BayesianNetworkType>(models, "BayesianNetworkType")
        .def_property_readonly_static("Gaussian", [](const py::object&) { 
            return BayesianNetworkType(BayesianNetworkType::Gaussian);
        })
        .def_property_readonly_static("Semiparametric", [](const py::object&) { 
            return BayesianNetworkType(BayesianNetworkType::Semiparametric);
        })
        .def_property_readonly_static("Discrete", [](const py::object&) { 
            return BayesianNetworkType(BayesianNetworkType::Discrete);
        })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__", &BayesianNetworkType::ToString)
        .def("__str__", &BayesianNetworkType::ToString);

    py::class_<BayesianNetworkBase>(models, "BayesianNetworkBase")
        .def_property_readonly("type", &BayesianNetworkBase::type)
        .def("num_nodes", &BayesianNetworkBase::num_nodes)
        .def("num_arcs", &BayesianNetworkBase::num_arcs)
        .def("nodes", &BayesianNetworkBase::nodes, py::return_value_policy::reference_internal)
        .def("arcs", &BayesianNetworkBase::arcs, py::return_value_policy::take_ownership)
        .def("indices", &BayesianNetworkBase::indices, py::return_value_policy::reference_internal)
        .def("index", &BayesianNetworkBase::index)
        .def("collapsed_index", &BayesianNetworkBase::collapsed_index)
        .def("index_from_collapsed", &BayesianNetworkBase::index_from_collapsed)
        .def("collapsed_from_index", &BayesianNetworkBase::collapsed_from_index)
        .def("collapsed_indices", &BayesianNetworkBase::collapsed_indices, py::return_value_policy::reference_internal)
        .def("is_valid", &BayesianNetworkBase::is_valid)
        .def("contains_node", &BayesianNetworkBase::contains_node)
        .def("add_node", &BayesianNetworkBase::add_node)
        .def("remove_node", py::overload_cast<int>(&BayesianNetworkBase::remove_node))
        .def("remove_node", py::overload_cast<const std::string&>(&BayesianNetworkBase::remove_node))
        .def("name", &BayesianNetworkBase::name)
        .def("collapsed_name", &BayesianNetworkBase::collapsed_name)
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
        .def("check_blacklist", py::overload_cast<const ArcStringVector&>(&BayesianNetworkBase::check_blacklist, py::const_))
        .def("force_whitelist", py::overload_cast<const ArcStringVector&>(&BayesianNetworkBase::force_whitelist))
        .def("fitted", &BayesianNetworkBase::fitted)
        .def("fit", &BayesianNetworkBase::fit)
        .def("logl", &BayesianNetworkBase::logl, py::return_value_policy::take_ownership)
        .def("slogl", &BayesianNetworkBase::slogl)
        .def("sample", [](const BayesianNetworkBase& self, int n, bool ordered) {
                return self.sample(n, std::random_device{}(), ordered);
        }, py::return_value_policy::move, py::arg("n"), py::arg("ordered") = false)
        .def("sample", &BayesianNetworkBase::sample, py::return_value_policy::move, 
                py::arg("n"), py::arg("seed"), py::arg("ordered") = false)
        .def("conditional_bn", py::overload_cast<const std::vector<std::string>&, 
                                                 const std::vector<std::string>&>(&BayesianNetworkBase::conditional_bn, py::const_))
        .def("conditional_bn", py::overload_cast<>(&BayesianNetworkBase::conditional_bn, py::const_))
        .def("unconditional_bn", &BayesianNetworkBase::unconditional_bn)
        .def("save", &BayesianNetworkBase::save)
        .def("clone", &BayesianNetworkBase::clone);

    py::class_<ConditionalBayesianNetworkBase, BayesianNetworkBase>(models, "ConditionalBayesianNetworkBase")
        .def("num_interface_nodes", &ConditionalBayesianNetworkBase::num_interface_nodes)
        .def("num_total_nodes", &ConditionalBayesianNetworkBase::num_total_nodes)
        .def("interface_nodes", &ConditionalBayesianNetworkBase::interface_nodes, py::return_value_policy::reference_internal)
        .def("all_nodes", &ConditionalBayesianNetworkBase::all_nodes, py::return_value_policy::reference_internal)
        .def("interface_collapsed_index", &ConditionalBayesianNetworkBase::interface_collapsed_index)
        .def("joint_collapsed_index", &ConditionalBayesianNetworkBase::joint_collapsed_index)
        .def("interface_collapsed_indices", &ConditionalBayesianNetworkBase::interface_collapsed_indices, py::return_value_policy::reference_internal)
        .def("joint_collapsed_indices", &ConditionalBayesianNetworkBase::joint_collapsed_indices, py::return_value_policy::reference_internal)
        .def("index_from_interface_collapsed", &ConditionalBayesianNetworkBase::index_from_interface_collapsed)
        .def("index_from_joint_collapsed", &ConditionalBayesianNetworkBase::index_from_joint_collapsed)
        .def("interface_collapsed_from_index", &ConditionalBayesianNetworkBase::interface_collapsed_from_index)
        .def("joint_collapsed_from_index", &ConditionalBayesianNetworkBase::joint_collapsed_from_index)
        .def("interface_collapsed_name", &ConditionalBayesianNetworkBase::interface_collapsed_name)
        .def("joint_collapsed_name", &ConditionalBayesianNetworkBase::joint_collapsed_name)
        .def("contains_interface_node", &ConditionalBayesianNetworkBase::contains_interface_node)
        .def("contains_total_node", &ConditionalBayesianNetworkBase::contains_total_node)
        .def("add_interface_node", &ConditionalBayesianNetworkBase::add_interface_node)
        .def("remove_interface_node", py::overload_cast<int>(&ConditionalBayesianNetworkBase::remove_interface_node))
        .def("remove_interface_node", py::overload_cast<const std::string&>(&ConditionalBayesianNetworkBase::remove_interface_node))
        .def("is_interface", py::overload_cast<int>(&ConditionalBayesianNetworkBase::is_interface, py::const_))
        .def("is_interface", py::overload_cast<const std::string&>(&ConditionalBayesianNetworkBase::is_interface, py::const_))
        .def("set_interface", py::overload_cast<int>(&ConditionalBayesianNetworkBase::set_interface))
        .def("set_interface", py::overload_cast<const std::string&>(&ConditionalBayesianNetworkBase::set_interface))
        .def("set_node", py::overload_cast<int>(&ConditionalBayesianNetworkBase::set_node))
        .def("set_node", py::overload_cast<const std::string&>(&ConditionalBayesianNetworkBase::set_node))
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
                        py::arg("ordered") = false)
        .def("clone", &ConditionalBayesianNetworkBase::clone);

    py::class_<DynamicBayesianNetworkBase>(models, "DynamicBayesianNetworkBase")
        .def_property_readonly("type", &DynamicBayesianNetworkBase::type)
        .def("static_bn", &DynamicBayesianNetworkBase::static_bn, py::return_value_policy::reference_internal)
        .def("transition_bn", &DynamicBayesianNetworkBase::transition_bn, py::return_value_policy::reference_internal)
        .def("markovian_order", &DynamicBayesianNetworkBase::markovian_order)
        .def("num_variables", &DynamicBayesianNetworkBase::num_variables)
        .def("variables", &DynamicBayesianNetworkBase::variables)
        .def("contains_variable", &DynamicBayesianNetworkBase::contains_variable)
        .def("add_variable", &DynamicBayesianNetworkBase::add_variable)
        .def("remove_variable", &DynamicBayesianNetworkBase::remove_variable)
        .def("fitted", &DynamicBayesianNetworkBase::fitted)
        .def("fit", &DynamicBayesianNetworkBase::fit)
        .def("logl", &DynamicBayesianNetworkBase::logl)
        .def("slogl", &DynamicBayesianNetworkBase::slogl)
        .def("sample", &DynamicBayesianNetworkBase::sample)
        .def("save", &DynamicBayesianNetworkBase::save);

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
                    .def(py::init<const ConditionalDag&,
                                  FactorStringTypeVector&>())
                    .def("node_type", py::overload_cast<const std::string&>(&ConditionalSemiparametricBN::node_type, py::const_))
                    .def("node_type", py::overload_cast<int>(&ConditionalSemiparametricBN::node_type, py::const_))
                    .def("set_node_type", py::overload_cast<const std::string&, FactorType>(&ConditionalSemiparametricBN::set_node_type))
                    .def("set_node_type", py::overload_cast<int, FactorType>(&ConditionalSemiparametricBN::set_node_type))
                    .def("node_types", &ConditionalSemiparametricBN::node_types);
    register_ConditionalBayesianNetwork<ConditionalDiscreteBN>(models, "ConditionalDiscreteBN");

    register_DynamicBayesianNetwork<DynamicGaussianNetwork>(models, "DynamicGaussianNetwork");
    register_DynamicBayesianNetwork<DynamicSemiparametricBN>(models, "DynamicSemiparametricBN");
    register_DynamicBayesianNetwork<DynamicDiscreteBN>(models, "DynamicDiscreteBN");
}
