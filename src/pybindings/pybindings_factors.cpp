#include <pybind11/pybind11.h>
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
using factors::FactorType;

void pybindings_factors(py::module& root) {
    auto factors = root.def_submodule("factors", "Factors submodule.");

    py::class_<FactorType>(factors, "FactorType")
        .def_property_readonly_static("LinearGaussianCPD", [](const py::object&) { 
            return FactorType(FactorType::LinearGaussianCPD);
        })
        .def_property_readonly_static("CKDE", [](const py::object&) { 
            return FactorType(FactorType::CKDE);
        })
        .def("opposite", &FactorType::opposite)
        .def(py::self == py::self)
        .def(py::self != py::self);

    auto continuous = factors.def_submodule("continuous", "Continuous factors submodule.");

    py::class_<LinearGaussianCPD>(continuous, "LinearGaussianCPD")
        .def(py::init<const std::string, const std::vector<std::string>>())
        .def(py::init<const std::string, const std::vector<std::string>, const std::vector<double>, double>())
        .def_property_readonly("variable", &LinearGaussianCPD::variable)
        .def_property_readonly("evidence", &LinearGaussianCPD::evidence)
        .def_property("beta", &LinearGaussianCPD::beta, &LinearGaussianCPD::setBeta)
        .def_property("variance", &LinearGaussianCPD::variance, &LinearGaussianCPD::setVariance)
        .def_property_readonly("fitted", &LinearGaussianCPD::fitted)
        .def("fit", &LinearGaussianCPD::fit)
        .def("logl", &LinearGaussianCPD::logl, py::return_value_policy::take_ownership)
        .def("slogl", &LinearGaussianCPD::slogl);

    py::class_<KDE>(continuous, "KDE")
        .def(py::init<std::vector<std::string>>())
        .def_property_readonly("variables", &KDE::variables)
        .def_property_readonly("n", &KDE::num_instances)
        .def_property_readonly("d", &KDE::num_variables)
        .def_property("bandwidth", &KDE::bandwidth, &KDE::setBandwidth)
        .def_property_readonly("fitted", &KDE::fitted)
        .def("fit", (void (KDE::*)(const DataFrame&))&KDE::fit)
        .def("logl", &KDE::logl, py::return_value_policy::take_ownership)
        .def("slogl", &KDE::slogl);

    py::class_<CKDE>(continuous, "CKDE")
        .def(py::init<const std::string, const std::vector<std::string>>())
        .def_property_readonly("variable", &CKDE::variable)
        .def_property_readonly("evidence", &CKDE::evidence)
        .def_property_readonly("n", &CKDE::num_instances)
        .def_property_readonly("kde_joint", &CKDE::kde_joint)
        .def_property_readonly("kde_marg", &CKDE::kde_marg)
        .def_property_readonly("fitted", &CKDE::fitted)
        .def("fit", &CKDE::fit)
        .def("logl", &CKDE::logl, py::return_value_policy::take_ownership)
        .def("slogl", &CKDE::slogl);

    py::class_<SemiparametricCPD>(continuous, "SemiparametricCPD")
        .def(py::init<LinearGaussianCPD>())
        .def(py::init<CKDE>())
        .def_property_readonly("variable", &SemiparametricCPD::variable)
        .def_property_readonly("evidence", &SemiparametricCPD::evidence)
        .def_property_readonly("node_type", &SemiparametricCPD::node_type)
        .def_property_readonly("fitted", &SemiparametricCPD::fitted)
        .def("as_lg", &SemiparametricCPD::as_lg, py::return_value_policy::reference_internal)
        .def("as_ckde", &SemiparametricCPD::as_ckde, py::return_value_policy::reference_internal)
        .def("fit", &SemiparametricCPD::fit)
        .def("logl", &SemiparametricCPD::logl, py::return_value_policy::take_ownership)
        .def("slogl", &SemiparametricCPD::slogl);
    
    py::implicitly_convertible<LinearGaussianCPD, SemiparametricCPD>();
    py::implicitly_convertible<CKDE, SemiparametricCPD>();

    auto discrete = factors.def_submodule("discrete", "Discrete factors submodule.");

    py::class_<DiscreteFactor>(discrete, "DiscreteFactor")
        .def(py::init<std::string, std::vector<std::string>>())
        .def_property_readonly("variable", &DiscreteFactor::variable)
        .def_property_readonly("evidence", &DiscreteFactor::evidence)
        .def_property_readonly("fitted", &DiscreteFactor::fitted)
        .def("fit", &DiscreteFactor::fit)
        .def("logl", &DiscreteFactor::logl, py::arg("df"), py::arg("check_domain") = true)
        .def("slogl", &DiscreteFactor::slogl, py::arg("df"), py::arg("check_domain") = true)
        .def("ToString", &DiscreteFactor::ToString);

}