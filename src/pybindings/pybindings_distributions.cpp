#include <pybind11/pybind11.h>
#include <distributions/normal.hpp>

namespace py = pybind11;

using distributions::Normal;

void pybindings_distributions(py::module& root) {
    auto distributions = root.def_submodule("distributions", "Distributions functionality.");

    py::class_<Normal>(distributions, "Normal")
        .def(py::init<const std::string&, double, double>())
        .def("logl", &Normal::logl)
        .def("natural_logl", &Normal::natural_logl);


}