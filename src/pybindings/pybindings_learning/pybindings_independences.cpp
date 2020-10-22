#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <learning/independences/independence.hpp>
#include <learning/independences/continuous/linearcorrelation.hpp>
#include <learning/independences/continuous/kdtree.hpp>
#include <learning/independences/continuous/mutual_information.hpp>

namespace py = pybind11;

using learning::independences::IndependenceTest, learning::independences::continuous::LinearCorrelation;

using learning::independences::KDTree;
using learning::independences::KMutualInformation;

void pybindings_independence_tests(py::module& root) {
    auto independence_tests = root.def_submodule("independences", "Independence Hypothesis tests.");

    py::class_<IndependenceTest, std::shared_ptr<IndependenceTest>>(independence_tests, "IndependenceTest")
        .def("pvalue", [](IndependenceTest& self, int v1, int v2) {
            return self.pvalue(v1, v2);
        })
        .def("pvalue", [](IndependenceTest& self, const std::string& v1, const std::string& v2) {
            return self.pvalue(v1, v2);
        })
        .def("pvalue", [](IndependenceTest& self, int v1, int v2, int cond) {
            return self.pvalue(v1, v2, cond);
        })
        .def("pvalue", [](IndependenceTest& self, const std::string& v1, const std::string& v2, const std::string& cond) {
            return self.pvalue(v1, v2, cond);
        })
        .def("pvalue", [](IndependenceTest& self, int v1, int v2, std::vector<int>& cond) {
            return self.pvalue(v1, v2, cond.begin(), cond.end());
        })
        .def("pvalue", [](IndependenceTest& self, const std::string& v1, const std::string& v2, std::vector<std::string>& cond) {
            return self.pvalue(v1, v2, cond.begin(), cond.end());
        });

    py::class_<LinearCorrelation, IndependenceTest, std::shared_ptr<LinearCorrelation>>(independence_tests, "LinearCorrelation")
        .def(py::init<const DataFrame>())
        .def("pvalue", [](LinearCorrelation& self, int v1, int v2) {
            return self.pvalue(v1, v2);
        })
        .def("pvalue", [](LinearCorrelation& self, const std::string& v1, const std::string& v2) {
            return self.pvalue(v1, v2);
        })
        .def("pvalue", [](LinearCorrelation& self, int v1, int v2, int cond) {
            return self.pvalue(v1, v2, cond);
        })
        .def("pvalue", [](LinearCorrelation& self, const std::string& v1, const std::string& v2, const std::string& cond) {
            return self.pvalue(v1, v2, cond);
        })
        .def("pvalue", [](LinearCorrelation& self, int v1, int v2, std::vector<int>& cond) {
            return self.pvalue(v1, v2, cond.begin(), cond.end());
        })
        .def("pvalue", [](LinearCorrelation& self, const std::string& v1, const std::string& v2, std::vector<std::string>& cond) {
            return self.pvalue(v1, v2, cond.begin(), cond.end());
        });

    py::class_<KDTree>(independence_tests, "KDTree")
        .def(py::init<DataFrame, int>())
        .def("query", &KDTree::query, py::arg("test_df"), py::arg("k") = 1, py::arg("p") = 2.)
        .def("count_subspace_eps", &KDTree::count_subspace_eps, 
                                    py::arg("test_df"), 
                                    py::arg("subspace_index"), 
                                    py::arg("eps"));

    py::class_<KMutualInformation>(independence_tests, "KMutualInformation")
        .def(py::init<DataFrame, int>(),
            py::arg("df"), py::arg("k"))
        .def("mi", [](KMutualInformation& self, int v1, int v2) {
            return self.mi(v1, v2);
        })
        .def("mi", [](KMutualInformation& self, const std::string& v1, const std::string& v2) {
            return self.mi(v1, v2);
        })
        .def("mi", [](KMutualInformation& self, int v1, int v2, int cond) {
            return self.mi(v1, v2, cond);
        })
        .def("mi", [](KMutualInformation& self, const std::string& v1, const std::string& v2, const std::string& cond) {
            return self.mi(v1, v2, cond);
        })
        .def("mi", [](KMutualInformation& self, int v1, int v2, std::vector<int>& cond) {
            return self.mi(v1, v2, cond.begin(), cond.end());
        })
        .def("mi", [](KMutualInformation& self, const std::string& v1, const std::string& v2, std::vector<std::string>& cond) {
            return self.mi(v1, v2, cond.begin(), cond.end());
        })
        .def("pvalue", [](KMutualInformation& self, int v1, int v2, int samples) {
            return self.pvalue(v1, v2, samples);
        }, py::arg("v1"), py::arg("v2"), py::arg("samples") = 1000)
        .def("pvalue", [](KMutualInformation& self, const std::string& v1, const std::string& v2, int samples) {
            return self.pvalue(v1, v2, samples);
        }, py::arg("v1"), py::arg("v2"), py::arg("samples") = 1000);
}