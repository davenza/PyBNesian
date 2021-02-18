#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <factors/continuous/SemiparametricCPD.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <factors/factors.hpp>

namespace py = pybind11;

using factors::continuous::LinearGaussianCPD, factors::continuous::KDE, 
      factors::continuous::CKDE, factors::continuous::SemiparametricCPD;
using factors::discrete::DiscreteFactor;
using factors::NodeType;

void pybindings_factors(py::module& root) {
    auto factors = root.def_submodule("factors", "Factors submodule.");

    factors.def("load_factor", &factors::load_factor);

    py::class_<NodeType>(factors, "NodeType")
        .def_property_readonly_static("LinearGaussianCPD", [](const py::object&) { 
            return NodeType(NodeType::LinearGaussianCPD);
        })
        .def_property_readonly_static("CKDE", [](const py::object&) { 
            return NodeType(NodeType::CKDE);
        })
        .def_property_readonly_static("SemiparametricCPD", [](const py::object&) { 
            return NodeType(NodeType::SemiparametricCPD);
        })
        .def_property_readonly_static("DiscreteFactor", [](const py::object&) { 
            return NodeType(NodeType::DiscreteFactor);
        })
        .def_static("from_string", &NodeType::from_string)
        .def("opposite_semiparametric", &NodeType::opposite_semiparametric)
        .def("__str__", &NodeType::ToString)
        .def("__repr__", &NodeType::ToString)
        .def(py::self == py::self)
        .def(py::self != py::self);

    auto continuous = factors.def_submodule("continuous", "Continuous factors submodule.");

    py::class_<LinearGaussianCPD>(continuous, "LinearGaussianCPD")
        .def(py::init<const std::string, const std::vector<std::string>>())
        .def(py::init<const std::string, const std::vector<std::string>, const std::vector<double>, double>())
        .def_property_readonly("variable", &LinearGaussianCPD::variable)
        .def_property_readonly("evidence", &LinearGaussianCPD::evidence)
        .def_property_readonly("node_type", &LinearGaussianCPD::node_type)
        .def_property("beta", [](const LinearGaussianCPD& self) {
            if (self.fitted())
                return self.beta();
            else
                throw py::value_error("LinearGaussianCPD not fitted.");
        }, &LinearGaussianCPD::set_beta)
        .def_property("variance", [](const LinearGaussianCPD& self) {
            if (self.fitted())
                return self.variance();
            else
                throw py::value_error("LinearGaussianCPD not fitted.");
        }, &LinearGaussianCPD::set_variance)
        .def_property_readonly("fitted", &LinearGaussianCPD::fitted)
        .def("fit", &LinearGaussianCPD::fit)
        .def("logl", &LinearGaussianCPD::logl, py::return_value_policy::take_ownership)
        .def("slogl", &LinearGaussianCPD::slogl)
        .def("cdf", &LinearGaussianCPD::cdf, py::return_value_policy::take_ownership)
        .def("sample", [](const LinearGaussianCPD& self, int n, std::optional<const DataFrame> evidence_values) {
            if (evidence_values)
                return self.sample(n, *evidence_values, std::random_device{}());
            else
                return self.sample(n, DataFrame(), std::random_device{}());
        }, py::arg("n"), py::arg("evidence_values"))
        .def("sample", [](const LinearGaussianCPD& self, int n, std::optional<const DataFrame> evidence_values, unsigned int seed) {
            if (evidence_values)
                return self.sample(n, *evidence_values, seed);
            else
                return self.sample(n, DataFrame(), seed);
        }, py::arg("n"), py::arg("evidence_values"), py::arg("seed"))
        .def("__str__", &LinearGaussianCPD::ToString)
        .def("__repr__", &LinearGaussianCPD::ToString)
        .def("save", &LinearGaussianCPD::save)
        .def(py::pickle(
            [](const LinearGaussianCPD& self) {
                return self.__getstate__();
            }, 
            [](py::tuple t) {
                return LinearGaussianCPD::__setstate__(t);
            }
        ));

    py::class_<KDE>(continuous, "KDE")
        .def(py::init<std::vector<std::string>>())
        .def_property_readonly("variables", &KDE::variables)
        .def_property_readonly("N", [](const KDE& self) {
            if (self.fitted())
                return self.num_instances();
            else
                throw py::value_error("KDE not fitted.");
        })
        .def_property_readonly("d", [](const KDE& self) {
            if (self.fitted())
                return self.num_variables();
            else
                throw py::value_error("KDE not fitted.");
        })
        .def_property("bandwidth", [](const KDE& self) {
            if (self.fitted())
                return self.bandwidth();
            else
                throw py::value_error("KDE not fitted.");
        }, &KDE::setBandwidth)
        .def_property_readonly("dataset", [](const KDE& self) {
            if (self.fitted())
                return self.training_data();
            else
                throw py::value_error("KDE not fitted.");
        })
        .def_property_readonly("fitted", &KDE::fitted)
        .def("fit", (void (KDE::*)(const DataFrame&))&KDE::fit)
        .def("logl", &KDE::logl, py::return_value_policy::take_ownership)
        .def("slogl", &KDE::slogl)
        .def("save", &KDE::save)
        .def(py::pickle(
            [](const KDE& self) {
                return self.__getstate__();
            }, 
            [](py::tuple t) {
                return KDE::__setstate__(t);
            }
        ));

    py::class_<CKDE>(continuous, "CKDE")
        .def(py::init<const std::string, const std::vector<std::string>>())
        .def_property_readonly("variable", &CKDE::variable)
        .def_property_readonly("evidence", &CKDE::evidence)
        .def_property_readonly("node_type", &CKDE::node_type)
        .def_property_readonly("N", [](const CKDE& self) {
            if (self.fitted())
                return self.num_instances();
            else
                throw py::value_error("CKDE not fitted.");
        })
        .def_property_readonly("kde_joint", [](CKDE& self) -> KDE& {
            if (self.fitted())
                return self.kde_joint();
            else
                throw py::value_error("CKDE not fitted.");
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("kde_marg", [](CKDE& self) -> KDE& {
            if (self.fitted())
                return self.kde_marg();
            else
                throw py::value_error("CKDE not fitted.");
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("fitted", &CKDE::fitted)
        .def("fit", &CKDE::fit)
        .def("logl", &CKDE::logl, py::return_value_policy::take_ownership)
        .def("slogl", &CKDE::slogl)
        .def("cdf", &CKDE::cdf, py::return_value_policy::take_ownership)
        .def("sample", [](const CKDE& self, int n, std::optional<const DataFrame> evidence_values) {
            if (evidence_values) {
                return self.sample(n, *evidence_values, std::random_device{}());
            }
            else {
                return self.sample(n, DataFrame(), std::random_device{}());
            }
        }, py::arg("n"), py::arg("evidence_values"))
        .def("sample", [](const CKDE& self, int n, std::optional<const DataFrame> evidence_values, unsigned int seed) {
            if (evidence_values) {
                return self.sample(n, *evidence_values, seed);
            }
            else {
                return self.sample(n, DataFrame(), seed);
            }
        }, py::arg("n"), py::arg("evidence_values"), py::arg("seed"))
        .def("save", &CKDE::save)
        .def("__str__", &CKDE::ToString)
        .def("__repr__", &CKDE::ToString)
        .def(py::pickle(
            [](const CKDE& self) {
                return self.__getstate__();
            }, 
            [](py::tuple t) {
                return CKDE::__setstate__(t);
            }
        ));

    py::class_<SemiparametricCPD>(continuous, "SemiparametricCPD")
        .def(py::init<LinearGaussianCPD>())
        .def(py::init<CKDE>())
        .def_property_readonly("variable", &SemiparametricCPD::variable)
        .def_property_readonly("evidence", &SemiparametricCPD::evidence)
        .def_property_readonly("node_type", &SemiparametricCPD::node_type)
        .def_property_readonly("underlying_node_type", &SemiparametricCPD::underlying_node_type)
        .def_property_readonly("fitted", &SemiparametricCPD::fitted)
        .def("as_lg", py::overload_cast<>(&SemiparametricCPD::as_lg), py::return_value_policy::reference_internal)
        .def("as_ckde", py::overload_cast<>(&SemiparametricCPD::as_ckde), py::return_value_policy::reference_internal)
        .def("fit", &SemiparametricCPD::fit)
        .def("logl", &SemiparametricCPD::logl, py::return_value_policy::take_ownership)
        .def("slogl", &SemiparametricCPD::slogl)
        .def("cdf", &SemiparametricCPD::cdf, py::return_value_policy::take_ownership)
        .def("sample", [](const SemiparametricCPD& self, int n, std::optional<const DataFrame> evidence_values) {
            if (evidence_values) {
                return self.sample(n, *evidence_values, std::random_device{}());
            }
            else {
                return self.sample(n, DataFrame(), std::random_device{}());
            }
        }, py::arg("n"), py::arg("evidence_values"))
        .def("sample", [](const SemiparametricCPD& self, 
                          int n, 
                          std::optional<const DataFrame> evidence_values, 
                          unsigned int seed) 
        {
            if (evidence_values) {
                return self.sample(n, *evidence_values, seed);
            }
            else {
                return self.sample(n, DataFrame(), seed);
            }
        }, py::arg("n"), py::arg("evidence_values"), py::arg("seed"))
        .def("save", &SemiparametricCPD::save)
        .def(py::pickle(
            [](const SemiparametricCPD& self) {
                return self.__getstate__();
            }, 
            [](py::tuple t) {
                return SemiparametricCPD::__setstate__(t);
            }
        ));
    
    py::implicitly_convertible<LinearGaussianCPD, SemiparametricCPD>();
    py::implicitly_convertible<CKDE, SemiparametricCPD>();

    auto discrete = factors.def_submodule("discrete", "Discrete factors submodule.");

    py::class_<DiscreteFactor>(discrete, "DiscreteFactor")
        .def(py::init<std::string, std::vector<std::string>>())
        .def_property_readonly("variable", &DiscreteFactor::variable)
        .def_property_readonly("evidence", &DiscreteFactor::evidence)
        .def_property_readonly("node_type", &DiscreteFactor::node_type)
        .def_property_readonly("fitted", &DiscreteFactor::fitted)
        .def("fit", &DiscreteFactor::fit)
        .def("logl", &DiscreteFactor::logl, py::arg("df"), py::arg("check_domain") = true)
        .def("slogl", &DiscreteFactor::slogl, py::arg("df"), py::arg("check_domain") = true)
        .def("ToString", &DiscreteFactor::ToString)
        .def("save", &DiscreteFactor::save)
        .def(py::pickle(
            [](const DiscreteFactor& self) {
                return self.__getstate__();
            }, 
            [](py::tuple t) {
                return DiscreteFactor::__setstate__(t);
            }
        ));

}