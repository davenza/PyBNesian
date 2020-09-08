#include <pybind11/pybind11.h>
#include <learning/independences/independence.hpp>
#include <learning/independences/continuous/linearcorrelation.hpp>

namespace py = pybind11;

using learning::independences::IndependenceTest, learning::independences::continuous::LinearCorrelation;

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
}