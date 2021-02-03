#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <learning/independences/independence.hpp>
#include <learning/independences/continuous/linearcorrelation.hpp>
#include <learning/independences/continuous/mutual_information.hpp>

namespace py = pybind11;

using learning::independences::IndependenceTest, learning::independences::continuous::LinearCorrelation;
using learning::independences::DynamicIndependenceTest, learning::independences::continuous::DynamicLinearCorrelation;

// using kdtree::KDTree;
using learning::independences::continuous::KMutualInformation,
      learning::independences::continuous::DynamicKMutualInformation;

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
        })
        .def("num_variables", &IndependenceTest::num_variables)
        .def("variable_names", &IndependenceTest::variable_names)
        .def("name", &IndependenceTest::name)
        .def("has_variables", py::overload_cast<const std::string&>(&IndependenceTest::has_variables, py::const_))
        .def("has_variables", py::overload_cast<const std::vector<std::string>&>(&IndependenceTest::has_variables, py::const_));

    py::class_<LinearCorrelation, IndependenceTest, std::shared_ptr<LinearCorrelation>>(independence_tests, "LinearCorrelation")
        .def(py::init<const DataFrame>());

    py::class_<KMutualInformation, IndependenceTest, std::shared_ptr<KMutualInformation>>(independence_tests, "KMutualInformation")
        .def(py::init<DataFrame, int, unsigned int, int, int>(),
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
        });

    py::class_<DynamicIndependenceTest,
               std::shared_ptr<DynamicIndependenceTest>>
                        (independence_tests, "DynamicIndependenceTest")
        .def("static_tests", &DynamicIndependenceTest::static_tests)
        .def("transition_tests", &DynamicIndependenceTest::transition_tests)
        .def("variable_names", &DynamicIndependenceTest::variable_names)
        .def("name", &DynamicIndependenceTest::name)
        .def("has_variables", py::overload_cast<const std::string&>(&DynamicIndependenceTest::has_variables, py::const_))
        .def("has_variables", py::overload_cast<const std::vector<std::string>&>(&DynamicIndependenceTest::has_variables, py::const_))
        .def("num_variables", &DynamicIndependenceTest::num_variables)
        .def("markovian_order", &DynamicIndependenceTest::markovian_order);

    py::class_<DynamicLinearCorrelation,
               DynamicIndependenceTest,
               std::shared_ptr<DynamicLinearCorrelation>>
                        (independence_tests, "DynamicLinearCorrelation")
        .def(py::init<const DynamicDataFrame&>());

    py::class_<DynamicKMutualInformation,
               DynamicIndependenceTest,
               std::shared_ptr<DynamicKMutualInformation>>
                        (independence_tests, "DynamicKMutualInformation")
        .def(py::init<const DynamicDataFrame&, int, unsigned int, int, int>(),
            py::arg("df"), py::arg("k"), py::arg("seed"), py::arg("shuffle_neighbors") = 5, py::arg("samples") = 1000)
        .def(py::init<const DynamicDataFrame&, int,  int, int>(),
            py::arg("df"), py::arg("k"), py::arg("shuffle_neighbors") = 5, py::arg("samples") = 1000);
}