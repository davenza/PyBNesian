#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <learning/independences/independence.hpp>
#include <learning/independences/continuous/linearcorrelation.hpp>
#include <learning/independences/continuous/mutual_information.hpp>
#include <learning/independences/continuous/RCoT.hpp>
#include <util/util_types.hpp>

namespace py = pybind11;

using learning::independences::IndependenceTest, learning::independences::continuous::LinearCorrelation,
    learning::independences::continuous::KMutualInformation, learning::independences::continuous::RCoT;

using learning::independences::DynamicIndependenceTest, learning::independences::continuous::DynamicLinearCorrelation,
    learning::independences::continuous::DynamicKMutualInformation;

using util::random_seed_arg;

class PyIndependenceTest : public IndependenceTest {
public:
    using IndependenceTest::IndependenceTest;

    double pvalue(const std::string& v1, const std::string& v2) const override {
        PYBIND11_OVERRIDE_PURE(double,           /* Return type */
                               IndependenceTest, /* Parent class */
                               pvalue,           /* Name of function in C++ (must match Python name) */
                               v1,
                               v2,
                               std::nullopt /* Argument(s) */
        );
    }

    double pvalue(const std::string& v1, const std::string& v2, const std::string& ev) const override {
        PYBIND11_OVERRIDE_PURE(double,           /* Return type */
                               IndependenceTest, /* Parent class */
                               pvalue,           /* Name of function in C++ (must match Python name) */
                               v1,
                               v2,
                               ev /* Argument(s) */
        );
    }

    double pvalue(const std::string& v1, const std::string& v2, const std::vector<std::string>& ev) const override {
        PYBIND11_OVERRIDE_PURE(double,           /* Return type */
                               IndependenceTest, /* Parent class */
                               pvalue,           /* Name of function in C++ (must match Python name) */
                               v1,
                               v2,
                               ev /* Argument(s) */
        );
    }

    int num_variables() const override {
        PYBIND11_OVERRIDE_PURE(int,              /* Return type */
                               IndependenceTest, /* Parent class */
                               num_variables     /* Name of function in C++ (must match Python name) */
        );
    }

    std::vector<std::string> variable_names() const override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::string>, /* Return type */
                               IndependenceTest,         /* Parent class */
                               variable_names            /* Name of function in C++ (must match Python name) */
        );
    }

    const std::string& name(int i) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, /* Return type */
                               IndependenceTest,   /* Parent class */
                               name,               /* Name of function in C++ (must match Python name) */
                               i                   /* Argument(s) */
        );
    }

    bool has_variables(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(bool,             /* Return type */
                               IndependenceTest, /* Parent class */
                               has_variables,    /* Name of function in C++ (must match Python name) */
                               name              /* Argument(s) */
        );
    }

    bool has_variables(const std::vector<std::string>& cols) const override {
        PYBIND11_OVERRIDE_PURE(bool,             /* Return type */
                               IndependenceTest, /* Parent class */
                               has_variables,    /* Name of function in C++ (must match Python name) */
                               cols              /* Argument(s) */
        );
    }
};

void pybindings_independence_tests(py::module& root) {
    auto independence_tests = root.def_submodule("independences", "Independence Hypothesis tests.");

    py::class_<IndependenceTest, PyIndependenceTest, std::shared_ptr<IndependenceTest>>(independence_tests,
                                                                                        "IndependenceTest")
        .def(py::init<>())
        .def("pvalue",
             [](IndependenceTest& self, const std::string& v1, const std::string& v2) { return self.pvalue(v1, v2); })
        .def("pvalue",
             [](IndependenceTest& self, const std::string& v1, const std::string& v2, const std::string& cond) {
                 return self.pvalue(v1, v2, cond);
             })
        .def("pvalue",
             [](IndependenceTest& self, const std::string& v1, const std::string& v2, std::vector<std::string>& cond) {
                 return self.pvalue(v1, v2, cond);
             })
        .def("num_variables", &IndependenceTest::num_variables)
        .def("variable_names", &IndependenceTest::variable_names)
        .def("name", &IndependenceTest::name)
        .def("has_variables", py::overload_cast<const std::string&>(&IndependenceTest::has_variables, py::const_))
        .def("has_variables",
             py::overload_cast<const std::vector<std::string>&>(&IndependenceTest::has_variables, py::const_));

    py::class_<LinearCorrelation, IndependenceTest, std::shared_ptr<LinearCorrelation>>(independence_tests,
                                                                                        "LinearCorrelation")
        .def(py::init<const DataFrame&>());

    py::class_<KMutualInformation, IndependenceTest, std::shared_ptr<KMutualInformation>>(independence_tests,
                                                                                          "KMutualInformation")
        .def(py::init([](DataFrame df, int k, std::optional<unsigned int> seed, int shuffle_neighbors, int samples) {
                 return KMutualInformation(df, k, random_seed_arg(seed), shuffle_neighbors, samples);
             }),
             py::arg("df"),
             py::arg("k"),
             py::arg("seed") = std::nullopt,
             py::arg("shuffle_neighbors") = 5,
             py::arg("samples") = 1000)
        .def_property("samples", &KMutualInformation::samples, &KMutualInformation::set_samples)
        .def_property("seed", &KMutualInformation::seed, &KMutualInformation::set_seed)
        .def("mi", [](KMutualInformation& self, const std::string& x, const std::string& y) { return self.mi(x, y); })
        .def("mi",
             [](KMutualInformation& self, const std::string& x, const std::string& y, const std::string& z) {
                 return self.mi(x, y, z);
             })
        .def("mi",
             [](KMutualInformation& self,
                const std::string& x,
                const std::string& y,
                const std::vector<std::string>& z) { return self.mi(x, y, z); });

    py::class_<RCoT, IndependenceTest, std::shared_ptr<RCoT>>(independence_tests, "RCoT")
        .def(py::init<const DataFrame&, int, int>(),
             py::arg("df"),
             py::arg("random_fourier_xy") = 5,
             py::arg("random_fourier_z") = 100);

    py::class_<DynamicIndependenceTest, std::shared_ptr<DynamicIndependenceTest>>(independence_tests,
                                                                                  "DynamicIndependenceTest")
        .def("static_tests", &DynamicIndependenceTest::static_tests, py::return_value_policy::reference_internal)
        .def(
            "transition_tests", &DynamicIndependenceTest::transition_tests, py::return_value_policy::reference_internal)
        .def("variable_names", &DynamicIndependenceTest::variable_names)
        .def("name", &DynamicIndependenceTest::name)
        .def("has_variables",
             py::overload_cast<const std::string&>(&DynamicIndependenceTest::has_variables, py::const_))
        .def("has_variables",
             py::overload_cast<const std::vector<std::string>&>(&DynamicIndependenceTest::has_variables, py::const_))
        .def("num_variables", &DynamicIndependenceTest::num_variables)
        .def("markovian_order", &DynamicIndependenceTest::markovian_order);

    py::class_<DynamicLinearCorrelation, DynamicIndependenceTest, std::shared_ptr<DynamicLinearCorrelation>>(
        independence_tests, "DynamicLinearCorrelation", py::multiple_inheritance())
        .def(py::init<const DynamicDataFrame&>());

    py::class_<DynamicKMutualInformation, DynamicIndependenceTest, std::shared_ptr<DynamicKMutualInformation>>(
        independence_tests, "DynamicKMutualInformation", py::multiple_inheritance())
        .def(py::init([](const DynamicDataFrame& df,
                         int k,
                         std::optional<unsigned int> seed,
                         int shuffle_neighbors,
                         int samples) {
                 return DynamicKMutualInformation(
                     df, k, static_cast<unsigned int>(random_seed_arg(seed)), shuffle_neighbors, samples);
             }),
             py::arg("df"),
             py::arg("k"),
             py::arg("seed") = std::nullopt,
             py::arg("shuffle_neighbors") = 5,
             py::arg("samples") = 1000);
}
