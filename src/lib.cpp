#include <iostream>
#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>

#include <graph/dag.hpp>

#include <learning/scores/bic.hpp>
// #include <learning/scores/bic.hpp>
#include <learning/parameter/mle.hpp>
#include <learning/algorithms/hillclimbing.hpp>

namespace py = pybind11;

namespace pyarrow = arrow::py;

// using namespace factors::continuous;
using namespace ::graph;

using factors::continuous::LinearGaussianCPD;
using factors::continuous::KDE;
using factors::continuous::CKDE;


PYBIND11_MODULE(pgm_dataset, m) {
//    TODO: Check error
    pyarrow::import_pyarrow();

    m.doc() = "pybind11 data plugin"; // optional module docstring

    auto factors = m.def_submodule("factors", "Factors submodule.");
    auto continuous = factors.def_submodule("continuous", "Continuous factors submodule.");

    py::class_<LinearGaussianCPD>(continuous, "LinearGaussianCPD")
            .def(py::init<const std::string, const std::vector<std::string>>())
            .def(py::init<const std::string, const std::vector<std::string>, const std::vector<double>, double>())
            .def("fit", py::overload_cast<py::handle>(&LinearGaussianCPD::fit));

    py::class_<KDE>(continuous, "KDE")
             .def(py::init<std::vector<std::string>>())
            //  .def("fit", py::overload_cast<py::handle>(&KDE::fit))
             .def("fit", (void (KDE::*)(py::handle)) &KDE::fit)
             .def("logpdf", py::overload_cast<py::handle>(&KDE::logpdf, py::const_));

//     py::class_<CKDE>(continuous, "CKDE")
//             .def(py::init<const std::string, const std::vector<std::string>>())
            // .def(py::init<const std::string, const std::vector<std::string>, const std::vector<double>, double>())
        //     .def("fit", py::overload_cast<py::handle>(&CKDE::fit));

    auto learning = m.def_submodule("learning", "Learning submodule");
    auto algorithms = learning.def_submodule("algorithms", "Learning algorithms");

    algorithms.def("hc", &learning::algorithms::hc, "Hill climbing estimate");
}