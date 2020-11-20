#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybindings/pybindings_learning/pybindings_mle.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <learning/parameters/mle_base.hpp>
#include <learning/parameters/mle_LinearGaussianCPD.hpp>

namespace py = pybind11;

using factors::continuous::LinearGaussianCPD;
using learning::parameters::MLE;

void pybindings_parameters(py::module& root) {
    auto parameters = root.def_submodule("parameters", "Learning parameters submodule.");

    parameters.def("MLE", &pybindings::learning::parameters::mle_python_wrapper, py::return_value_policy::take_ownership);

    // TODO Fit LinearGaussianCPD with ParamsClass.
    py::class_<LinearGaussianCPD::ParamsClass>(parameters, "MLELinearGaussianParams")
        .def_readwrite("beta", &LinearGaussianCPD::ParamsClass::beta)
        .def_readwrite("variance", &LinearGaussianCPD::ParamsClass::variance);

    py::class_<MLE<LinearGaussianCPD>>(parameters, "MLE<LinearGaussianCPD>")
        .def("estimate", [](MLE<LinearGaussianCPD> self, const DataFrame& df, std::string var, std::vector<std::string> evidence) {
            return self.estimate(df, var, evidence.begin(), evidence.end());
        }, py::return_value_policy::take_ownership)
        .def("estimate", [](MLE<LinearGaussianCPD> self, const DataFrame& df, int idx, std::vector<int> evidence_idx) {
            return self.estimate(df, idx, evidence_idx.begin(), evidence_idx.end());
        }, py::return_value_policy::take_ownership);
}