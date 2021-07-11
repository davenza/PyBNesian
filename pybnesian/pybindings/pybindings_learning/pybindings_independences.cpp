#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <learning/independences/independence.hpp>
#include <learning/independences/continuous/linearcorrelation.hpp>
#include <learning/independences/continuous/mutual_information.hpp>
#include <learning/independences/continuous/RCoT.hpp>
#include <learning/independences/discrete/chi_square.hpp>
#include <learning/independences/hybrid/mutual_information.hpp>
#include <util/util_types.hpp>

namespace py = pybind11;

using learning::independences::IndependenceTest, learning::independences::continuous::LinearCorrelation,
    learning::independences::continuous::KMutualInformation, learning::independences::continuous::RCoT,
    learning::independences::discrete::ChiSquare, learning::independences::hybrid::MutualInformation;

using learning::independences::DynamicIndependenceTest, learning::independences::continuous::DynamicLinearCorrelation,
    learning::independences::continuous::DynamicKMutualInformation, learning::independences::continuous::DynamicRCoT,
    learning::independences::discrete::DynamicChiSquare, learning::independences::hybrid::DynamicMutualInformation;

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
    py::class_<IndependenceTest, PyIndependenceTest, std::shared_ptr<IndependenceTest>> indep_test(
        root, "IndependenceTest", R"doc(
The :class:`IndependenceTest` is an abstract class defining an interface for a conditional test of independence.

An :class:`IndependenceTest` is defined over a set of variables and can calculate the p-value of any conditional test on
these variables.
)doc");
    indep_test
        .def(py::init<>(), R"doc(
Initializes an :class:`IndependenceTest`.
)doc")
        .def(
            "pvalue",
            [](IndependenceTest& self, const std::string& v1, const std::string& v2) { return self.pvalue(v1, v2); },
            py::arg("x"),
            py::arg("y"),
            R"doc(
Calculates the p-value of the unconditional test of independence :math:`x \perp y`.

:param x: A variable name.
:param y: A variable name.
:returns: The p-value of the unconditional test of independence :math:`x \perp y`.
)doc")
        .def(
            "pvalue",
            [](IndependenceTest& self, const std::string& v1, const std::string& v2, const std::string& cond) {
                return self.pvalue(v1, v2, cond);
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            R"doc(
Calculates the p-value of an univariate conditional test of independence :math:`x \perp y \mid z`.

:param x: A variable name.
:param y: A variable name.
:param z: A variable name.
:returns: The p-value of an univariate conditional test of independence :math:`x \perp y \mid z`.
)doc")
        .def(
            "pvalue",
            [](IndependenceTest& self, const std::string& v1, const std::string& v2, std::vector<std::string>& cond) {
                return self.pvalue(v1, v2, cond);
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            R"doc(
Calculates the p-value of a multivariate conditional test of independence :math:`x \perp y \mid \mathbf{z}`.

:param x: A variable name.
:param y: A variable name.
:param z: A list of variable names.
:returns: The p-value of a multivariate conditional test of independence :math:`x \perp y \mid \mathbf{z}`.
)doc")
        .def("num_variables", &IndependenceTest::num_variables, R"doc(
Gets the number of variables of the :class:`IndependenceTest`.

:returns: Number of variables of the :class:`IndependenceTest`.
)doc")
        .def("variable_names", &IndependenceTest::variable_names, R"doc(
Gets the list of variable names of the :class:`IndependenceTest`.

:returns: List of variable names of the :class:`IndependenceTest`.
)doc")
        .def("name", &IndependenceTest::name, py::arg("index"), R"doc(
Gets the variable name of the index-th variable.

:param index: Index of the variable.
:returns: Variable name at the ``index`` position.
)doc");

    {
        py::options options;
        options.disable_function_signatures();

        indep_test
            .def("has_variables",
                 py::overload_cast<const std::string&>(&IndependenceTest::has_variables, py::const_),
                 py::arg("variables"))
            .def("has_variables",
                 py::overload_cast<const std::vector<std::string>&>(&IndependenceTest::has_variables, py::const_),
                 py::arg("variables"),
                 R"doc(
has_variables(self: pybnesian.IndependenceTest, variables: str or List[str]) -> bool

Checks whether this :class:`IndependenceTest` has the given ``variables``.

:param variables: Name or list of variables.
:returns: True if the :class:`IndependenceTest` is defined over the set of ``variables``, False otherwise.
)doc");
    }

    py::class_<LinearCorrelation, IndependenceTest, std::shared_ptr<LinearCorrelation>>(
        root, "LinearCorrelation", R"doc(
This class implements a partial linear correlation independence test. This independence is only valid for continuous
data.
)doc")
        .def(py::init<const DataFrame&>(), py::arg("df"), R"doc(
Initializes a :class:`LinearCorrelation` for the continuous variables in the DataFrame ``df``.

:param df: DataFrame on which to calculate the independence tests.
)doc");

    py::class_<MutualInformation, IndependenceTest, std::shared_ptr<MutualInformation>>(root,
                                                                                        "MutualInformation",
                                                                                        R"doc(
This class implements a hypothesis test based on mutual information. This independence is implemented for a mix of
categorical and continuous data. The estimation of the mutual information assumes that the continuous data has a
Gaussian probability distribution. To compute the p-value, we use the relation between the
`Likelihood-ratio test <https://en.wikipedia.org/wiki/Likelihood-ratio_test>`_ and the mutual information, so it is known
that the null distribution has a chi-square distribution.

The theory behind this implementation is described with more detail in the following
:download:`document <../../mutual_information_pdf/mutual_information.pdf>`.
)doc")
        .def(py::init<const DataFrame&, bool>(),
             py::arg("df"),
             py::arg("asymptotic_df") = true,
             R"doc(
Initializes a :class:`MutualInformation` for data ``df``. The degrees of freedom for the chi-square null distribution
can be calculated with the with the asymptotic (if ``asymptotic_df`` is true) or empirical (if ``asymptotic_df`` is
false) expressions.

:param df: DataFrame on which to calculate the independence tests.
:param asymptotic_df: Whether to calculate the degrees of freedom with the asympototic or empirical expression. See the
    :download:`theory document <../../mutual_information_pdf/mutual_information.pdf>`.
)doc")
        .def(
            "mi",
            [](MutualInformation& self, const std::string& x, const std::string& y) { return self.mi(x, y); },
            py::arg("x"),
            py::arg("y"),
            R"doc(
Estimates the unconditional mutual information :math:`\text{MI}(x, y)`.

:param x: A variable name.
:param y: A variable name.
:returns: The unconditional mutual information :math:`\text{MI}(x, y)`.
)doc")
        .def(
            "mi",
            [](MutualInformation& self, const std::string& x, const std::string& y, const std::string& z) {
                return self.mi(x, y, z);
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            R"doc(
Estimates the univariate conditional mutual information :math:`\text{MI}(x, y \mid z)`.

:param x: A variable name.
:param y: A variable name.
:param z: A variable name.
:returns: The univariate conditional mutual information :math:`\text{MI}(x, y \mid z)`.
)doc")
        .def(
            "mi",
            [](MutualInformation& self, const std::string& x, const std::string& y, const std::vector<std::string>& z) {
                return self.mi(x, y, z);
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            R"doc(
Estimates the multivariate conditional mutual information :math:`\text{MI}(x, y \mid \mathbf{z})`.

:param x: A variable name.
:param y: A variable name.
:param z: A list of variable names.
:returns: The multivariate conditional mutual information :math:`\text{MI}(x, y \mid \mathbf{z})`.
)doc");

    py::class_<KMutualInformation, IndependenceTest, std::shared_ptr<KMutualInformation>>(
        root, "KMutualInformation", R"doc(
This class implements a non-parametric independence test that is based on the estimation of the mutual information
using k-nearest neighbors. This independence is only implemented for continuous data.

This independence test is based on [CMIknn]_.
)doc")
        .def(py::init([](DataFrame df, int k, std::optional<unsigned int> seed, int shuffle_neighbors, int samples) {
                 return KMutualInformation(df, k, random_seed_arg(seed), shuffle_neighbors, samples);
             }),
             py::arg("df"),
             py::arg("k"),
             py::arg("seed") = std::nullopt,
             py::arg("shuffle_neighbors") = 5,
             py::arg("samples") = 1000,
             R"doc(
Initializes a :class:`KMutualInformation` for data ``df``. ``k`` is the number of neighbors in the k-nn model used to
estimate the mutual information.

This is a permutation independence test, so ``samples`` defines the number of permutations. ``shuffle neighbors``
(:math:`k_{perm}` in the original paper [CMIknn]_) defines how many neighbors are used to perform the conditional
permutations.

:param df: DataFrame on which to calculate the independence tests.
:param k: number of neighbors in the k-nn model used to estimate the mutual information.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
:param shuffle_neighbors: Number of neighbors used to perform the conditional permutation.
:param samples: Number of permutations for the :class:`KMutualInformation`.
)doc")
        .def(
            "mi",
            [](KMutualInformation& self, const std::string& x, const std::string& y) { return self.mi(x, y); },
            py::arg("x"),
            py::arg("y"),
            R"doc(
Estimates the unconditional mutual information :math:`\text{MI}(x, y)`.

:param x: A variable name.
:param y: A variable name.
:returns: The unconditional mutual information :math:`\text{MI}(x, y)`.
)doc")
        .def(
            "mi",
            [](KMutualInformation& self, const std::string& x, const std::string& y, const std::string& z) {
                return self.mi(x, y, z);
            },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            R"doc(
Estimates the univariate conditional mutual information :math:`\text{MI}(x, y \mid z)`.

:param x: A variable name.
:param y: A variable name.
:param z: A variable name.
:returns: The univariate conditional mutual information :math:`\text{MI}(x, y \mid z)`.
)doc")
        .def(
            "mi",
            [](KMutualInformation& self,
               const std::string& x,
               const std::string& y,
               const std::vector<std::string>& z) { return self.mi(x, y, z); },
            py::arg("x"),
            py::arg("y"),
            py::arg("z"),
            R"doc(
Estimates the multivariate conditional mutual information :math:`\text{MI}(x, y \mid \mathbf{z})`.

:param x: A variable name.
:param y: A variable name.
:param z: A list of variable names.
:returns: The multivariate conditional mutual information :math:`\text{MI}(x, y \mid \mathbf{z})`.
)doc");

    py::class_<RCoT, IndependenceTest, std::shared_ptr<RCoT>>(root, "RCoT", R"doc(
This class implements a non-parametric independence test called Randomized Conditional Correlation Test (RCoT). This
method is described in [RCoT]_. This independence is only implemented for continuous data.

This method uses random fourier features and is designed to be a fast non-parametric independence test.
)doc")
        .def(py::init<const DataFrame&, int, int>(),
             py::arg("df"),
             py::arg("random_fourier_xy") = 5,
             py::arg("random_fourier_z") = 100,
             R"doc(
Initializes a :class:`RCoT` for data ``df``. The number of random fourier features used for the ``x`` and ``y`` variables
in :class:`IndependenceTest.pvalue` is ``random_fourier_xy``. The number of random features used for ``z`` is equal
to ``random_fourier_z``.

:param df: DataFrame on which to calculate the independence tests.
:param random_fourier_xy: Number of random fourier features for the variables of the independence test.
:param randoum_fourier_z: Number of random fourier features for the conditioning variables of the independence test.
)doc");

    py::class_<ChiSquare, IndependenceTest, std::shared_ptr<ChiSquare>>(root, "ChiSquare", R"doc(
Initializes a :class:`ChiSquare` for data ``df``. This independence test is only valid for categorical data.

It implements the Pearson's X^2 test.

:param df: DataFrame on which to calculate the independence tests.
)doc")
        .def(py::init<const DataFrame&>(), py::arg("df"));

    py::class_<DynamicIndependenceTest, std::shared_ptr<DynamicIndependenceTest>> dynamic_indep_test(
        root, "DynamicIndependenceTest", R"doc(
A :class:`DynamicIndependenceTest` adapts the static :class:`IndependenceTest` to learn dynamic Bayesian networks.
It generates a static and a transition independence test to learn the static and transition components of the dynamic
Bayesian network.

The dynamic independence tests are usually implemented using a
:class:`DynamicDataFrame <pybnesian.DynamicDataFrame>` with the methods
:func:`DynamicDataFrame.static_df <pybnesian.DynamicDataFrame.static_df>` and
:func:`DynamicDataFrame.transition_df <pybnesian.DynamicDataFrame.transition_df>`.
)doc");
    dynamic_indep_test
        .def("static_tests", &DynamicIndependenceTest::static_tests, py::return_value_policy::reference_internal, R"doc(
It returns the static independence test component of the :class:`DynamicIndependenceTest`.

:returns: The static independence test component.
)doc")
        .def("transition_tests",
             &DynamicIndependenceTest::transition_tests,
             py::return_value_policy::reference_internal,
             R"doc(
It returns the transition independence test component of the :class:`DynamicIndependenceTest`.

:returns: The transition independence test component.
)doc")
        .def("variable_names", &DynamicIndependenceTest::variable_names, R"doc(
Gets the list of variable names of the :class:`DynamicIndependenceTest`.

:returns: List of variable names of the :class:`DynamicIndependenceTest`.
)doc")
        .def("name", &DynamicIndependenceTest::name, py::arg("index"), R"doc(
Gets the variable name of the index-th variable.

:param index: Index of the variable.
:returns: Variable name at the ``index`` position.
)doc")
        .def("num_variables", &DynamicIndependenceTest::num_variables, R"doc(
Gets the number of variables of the :class:`DynamicIndependenceTest`.

:returns: Number of variables of the :class:`DynamicIndependenceTest`.
)doc")
        .def("markovian_order", &DynamicIndependenceTest::markovian_order, R"doc(
Gets the markovian order used in this :class:`DynamicIndependenceTest`.

:returns: Markovian order of the :class:`DynamicIndependenceTest`.
)doc");

    {
        py::options options;
        options.disable_function_signatures();

        dynamic_indep_test
            .def("has_variables",
                 py::overload_cast<const std::string&>(&DynamicIndependenceTest::has_variables, py::const_),
                 py::arg("variables"))
            .def(
                "has_variables",
                py::overload_cast<const std::vector<std::string>&>(&DynamicIndependenceTest::has_variables, py::const_),
                py::arg("variables"),
                R"doc(
has_variables(self: pybnesian.DynamicScore, variables: str or List[str]) -> bool

Checks whether this :class:`DynamicScore` has the given ``variables``.

:param variables: Name or list of variables.
:returns: True if the :class:`DynamicScore` is defined over the set of ``variables``, False otherwise.
)doc");
    }

    py::class_<DynamicLinearCorrelation, DynamicIndependenceTest, std::shared_ptr<DynamicLinearCorrelation>>(
        root, "DynamicLinearCorrelation", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`LinearCorrelation` independence test.
)doc")
        .def(py::init<const DynamicDataFrame&>(), py::arg("ddf"), R"doc(
Initializes a :class:`DynamicLinearCorrelation` with the given :class:`DynamicDataFrame` ``ddf``.

:param ddf: :class:`DynamicDataFrame` to create the :class:`DynamicLinearCorrelation`.
)doc");

    py::class_<DynamicMutualInformation, DynamicIndependenceTest, std::shared_ptr<DynamicMutualInformation>>(
        root, "DynamicMutualInformation", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`MutualInformation` independence test.
)doc")
        .def(py::init<const DynamicDataFrame&, bool>(),
             py::arg("ddf"),
             py::arg("asymptotic_df") = true,
             R"doc(
Initializes a :class:`DynamicMutualInformation` with the given :class:`DynamicDataFrame` ``df``. The ``asymptotic_df``
parameter is passed to the static and transition components of :class:`MutualInformation`.

:param ddf: :class:`DynamicDataFrame` to create the :class:`DynamicMutualInformation`.
:param asymptotic_df: Whether to calculate the asymptotic or empirical degrees of freedom of the chi-square null
    distribution.
)doc");

    py::class_<DynamicKMutualInformation, DynamicIndependenceTest, std::shared_ptr<DynamicKMutualInformation>>(
        root, "DynamicKMutualInformation", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`KMutualInformation` independence test.
)doc")
        .def(py::init([](const DynamicDataFrame& df,
                         int k,
                         std::optional<unsigned int> seed,
                         int shuffle_neighbors,
                         int samples) {
                 return DynamicKMutualInformation(
                     df, k, static_cast<unsigned int>(random_seed_arg(seed)), shuffle_neighbors, samples);
             }),
             py::arg("ddf"),
             py::arg("k"),
             py::arg("seed") = std::nullopt,
             py::arg("shuffle_neighbors") = 5,
             py::arg("samples") = 1000,
             R"doc(
Initializes a :class:`DynamicKMutualInformation` with the given :class:`DynamicDataFrame` ``df``. The ``k``, ``seed``,
``shuffle_neighbors`` and ``samples`` parameters are passed to the static and transition components of
:class:`KMutualInformation`.

:param ddf: :class:`DynamicDataFrame` to create the :class:`DynamicKMutualInformation`.
:param k: number of neighbors in the k-nn model used to estimate the mutual information.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
:param shuffle_neighbors: Number of neighbors used to perform the conditional permutation.
:param samples: Number of permutations for the :class:`KMutualInformation`.
)doc");

    py::class_<DynamicRCoT, DynamicIndependenceTest, std::shared_ptr<DynamicRCoT>>(
        root, "DynamicRCoT", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`RCoT` independence test.
)doc")
        .def(py::init<const DynamicDataFrame&, int, int>(),
             py::arg("ddf"),
             py::arg("random_fourier_xy") = 5,
             py::arg("random_fourier_z") = 100,
             R"doc(
Initializes a :class:`DynamicRCoT` with the given :class:`DynamicDataFrame` ``df``. The ``random_fourier_xy`` and
``random_fourier_z`` parameters are passed to the static and transition components of
:class:`RCoT`.

:param ddf: :class:`DynamicDataFrame` to create the :class:`DynamicRCoT`.
:param random_fourier_xy: Number of random fourier features for the variables of the independence test.
:param randoum_fourier_z: Number of random fourier features for the conditioning variables of the independence test.
)doc");

    py::class_<DynamicChiSquare, DynamicIndependenceTest, std::shared_ptr<DynamicChiSquare>>(
        root, "DynamicChiSquare", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`ChiSquare` independence test.
)doc")
        .def(py::init<const DynamicDataFrame&>(),
             py::arg("ddf"),
             R"doc(
Initializes a :class:`DynamicChiSquare` with the given :class:`DynamicDataFrame` ``df``.

:param ddf: :class:`DynamicDataFrame` to create the :class:`DynamicChiSquare`.
)doc");
}
