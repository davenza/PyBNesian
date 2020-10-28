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
        .def("query", &KDTree::query, py::arg("test_df"), py::arg("k") = 1, py::arg("p") = 2.);

    py::class_<KMutualInformation>(independence_tests, "KMutualInformation")
        .def(py::init<DataFrame, int, long unsigned int, int, int>(),
            py::arg("df"), py::arg("k"), py::arg("seed"), py::arg("shuffle_neighbors") = 5, py::arg("samples") = 1000)
        .def(py::init<DataFrame, int,  int, int>(),
            py::arg("df"), py::arg("k"), py::arg("shuffle_neighbors") = 5, py::arg("samples") = 1000)
        .def_property("samples", &KMutualInformation::samples, &KMutualInformation::set_samples)
        .def_property("seed", &KMutualInformation::seed, &KMutualInformation::set_seed)
        .def("mi", [](KMutualInformation& self, int x, int y) {
            return self.mi(x, y);
        })
        .def("mi", [](KMutualInformation& self, const std::string& x, const std::string& y) {
            return self.mi(x, y);
        })
        .def("mi", [](KMutualInformation& self, int x, int y, int z) {
            return self.mi(x, y, z);
        })
        .def("mi", [](KMutualInformation& self, const std::string& x, const std::string& y, const std::string& z) {
            return self.mi(x, y, z);
        })
        .def("mi", [](KMutualInformation& self, int x, int y, const std::vector<int>& z) {
            return self.mi(x, y, z.begin(), z.end());
        })
        .def("mi", [](KMutualInformation& self, const std::string& x, const std::string& y, const std::vector<std::string>& z) {
            return self.mi(x, y, z.begin(), z.end());
        })
        .def("pvalue", [](KMutualInformation& self, int x, int y) {
            return self.pvalue(x, y);
        }, py::arg("x"), py::arg("y"))
        .def("pvalue", [](KMutualInformation& self, const std::string& x, const std::string& y) {
            return self.pvalue(x, y);
        }, py::arg("x"), py::arg("y"))
        .def("pvalue", [](KMutualInformation& self, int x, int y, int z) {
            return self.pvalue(x, y, z);
        }, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("pvalue", [](KMutualInformation& self, 
                            const std::string& x, 
                            const std::string& y, 
                            const std::string& z) {
            return self.pvalue(x, y, z);
        }, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("pvalue", [](KMutualInformation& self, int x, int y, const std::vector<int>& z) {
            return self.pvalue(x, y, z.begin(), z.end());
        }, py::arg("x"), py::arg("y"), py::arg("z"))
        .def("pvalue", [](KMutualInformation& self, const std::string& x, const std::string& y, const std::vector<std::string>& z) {
            return self.pvalue(x, y, z.begin(), z.end());
        }, py::arg("x"), py::arg("y"), py::arg("z"));
}