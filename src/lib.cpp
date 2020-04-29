#include <pybind11/pybind11.h>
#include <iostream>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <arrow/python/pyarrow.h>

namespace py = pybind11;

namespace pyarrow = arrow::py;

using namespace factors::continuous;


PYBIND11_MODULE(pgm_dataset, m) {
//    TODO: Check error
    pyarrow::import_pyarrow();

    m.doc() = "pybind11 data plugin"; // optional module docstring

    py::class_<LinearGaussianCPD>(m, "LinearGaussianCPD")
            .def(py::init<const std::string, const std::vector<std::string>>())
            .def(py::init<const std::string, const std::vector<std::string>, const std::vector<double>>())
            .def("fit", &LinearGaussianCPD::fit);

}