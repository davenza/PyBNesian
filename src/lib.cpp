#include <iostream>
#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>

#include <dataset/crossvalidation_adaptator.hpp>
#include <dataset/holdout_adaptator.hpp>

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

using dataset::DataFrame, dataset::CrossValidation, dataset::HoldOut;

using namespace ::graph;

using factors::continuous::LinearGaussianCPD;
using factors::continuous::KDE;
using factors::continuous::CKDE;



PYBIND11_MODULE(pgm_dataset, m) {
//    TODO: Check error
    pyarrow::import_pyarrow();

    m.doc() = "pybind11 data plugin"; // optional module docstring

    auto dataset = m.def_submodule("dataset", "Dataset functionality.");

    py::class_<CrossValidation>(dataset, "CrossValidation")
            .def(py::init<DataFrame, int, bool>(), 
                        py::arg("df"), 
                        py::arg("k") = 10, 
                        py::arg("include_null") = false)
            .def(py::init<DataFrame, int, int, bool>(), 
                        py::arg("df"), 
                        py::arg("k") = 10, 
                        py::arg("seed"), 
                        py::arg("include_null") = false)
            .def("__iter__", [](CrossValidation& self) { 
                        return py::make_iterator(self.begin(), self.end()); }, 
                py::keep_alive<0, 1>())
            .def("fold", &CrossValidation::fold)
            .def("loc", [](CrossValidation& self, std::string name) { return self.loc(name); })
            .def("loc", [](CrossValidation& self, int idx) { return self.loc(idx); })
            .def("loc", [](CrossValidation& self, std::vector<std::string> v) { return self.loc(v); })
            .def("loc", [](CrossValidation& self, std::vector<int> v) { return self.loc(v); })
            .def("indices", [](CrossValidation& self) { 
                        return py::make_iterator(self.begin_indices(), self.end_indices()); }
                        );

    py::class_<HoldOut>(dataset, "HoldOut")
            .def(py::init<const DataFrame&, double, bool>(), 
                        py::arg("df"), 
                        py::arg("test_ratio") = 0.2, 
                        py::arg("include_null") = false)
            .def(py::init<const DataFrame&, double, int, bool>(), 
                        py::arg("df"), 
                        py::arg("test_ratio") = 0.2, 
                        py::arg("seed"), 
                        py::arg("include_null") = false)
            .def("training_data", &HoldOut::training_data)
            .def("test_data", &HoldOut::test_data);

    auto factors = m.def_submodule("factors", "Factors submodule.");
    auto continuous = factors.def_submodule("continuous", "Continuous factors submodule.");

    py::class_<LinearGaussianCPD>(continuous, "LinearGaussianCPD")
            .def(py::init<const std::string, const std::vector<std::string>>())
            .def(py::init<const std::string, const std::vector<std::string>, const std::vector<double>, double>())
            .def("fit", &LinearGaussianCPD::fit)
            .def("logpdf", &LinearGaussianCPD::logpdf)
            .def("slogpdf", &LinearGaussianCPD::slogpdf);

    py::class_<KDE>(continuous, "KDE")
             .def(py::init<std::vector<std::string>>())
        //      .def("fit", &KDE::fit) // errors for some reason
             .def("fit", (void (KDE::*)(const DataFrame&))&KDE::fit)
             .def("logpdf", &KDE::logpdf)
             .def("slogpdf", &KDE::slogpdf);

    py::class_<CKDE>(continuous, "CKDE")
             .def(py::init<const std::string, const std::vector<std::string>>())
             .def("fit", &CKDE::fit)
             .def("logpdf", &CKDE::logpdf)
             .def("slogpdf", &CKDE::slogpdf);

    auto learning = m.def_submodule("learning", "Learning submodule");
    auto algorithms = learning.def_submodule("algorithms", "Learning algorithms");

    algorithms.def("hc", &learning::algorithms::hc, "Hill climbing estimate");
    // py::class_<GreedyHillClimbing>(algorithms, "GreedyHillClimbing")
    //         .def(py::init<>())
    //         .def("estimate")
}