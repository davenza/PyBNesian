#include <iostream>
#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>

// #include <factors/continuous/LinearGaussianCPD.hpp>

#include <graph/dag.hpp>

#include <learning/scores/scores.hpp>
// #include <learning/scores/bic.hpp>
#include <learning/parameter/mle.hpp>
#include <learning/algorithms/hillclimbing.hpp>

namespace py = pybind11;

namespace pyarrow = arrow::py;

// using namespace factors::continuous;
using namespace ::graph;


PYBIND11_MODULE(pgm_dataset, m) {
//    TODO: Check error
    pyarrow::import_pyarrow();

    m.doc() = "pybind11 data plugin"; // optional module docstring

    py::class_<LinearGaussianCPD>(m, "LinearGaussianCPD")
            .def(py::init<const std::string, const std::vector<std::string>>())
            .def(py::init<const std::string, const std::vector<std::string>, const std::vector<double>, double>())
            .def("fit", &LinearGaussianCPD::fit);


    m.def("estimate", &learning::algorithms::estimate, "Hill climbing");

    m.def("benchmark_sort_vec", &learning::algorithms::benchmark_sort_vec, "Hill climbing");
    m.def("benchmark_sort_set", &learning::algorithms::benchmark_sort_set, "Hill climbing");
    m.def("benchmark_sort_priority", &learning::algorithms::benchmark_sort_priority, "Hill climbing");
    m.def("benchmark_sort_heap", &learning::algorithms::benchmark_sort_heap, "Hill climbing");
}