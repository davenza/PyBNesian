#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybindings/pybindings_learning/pybindings_mle.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <learning/parameters/mle_base.hpp>
#include <learning/parameters/mle_LinearGaussianCPD.hpp>

namespace py = pybind11;

using factors::continuous::LinearGaussianCPD;
using factors::discrete::DiscreteFactor;
using learning::parameters::MLE;

DiscreteFactor::ParamsClass numpy_to_discrete_params(
    py::array_t<double, py::array::f_style | py::array::forcecast> logprob) {
    auto size = logprob.size();
    auto info = logprob.request();
    double* ptr = static_cast<double*>(info.ptr);

    DiscreteFactor::ParamsClass params{VectorXd(logprob.size()), VectorXi(logprob.ndim())};

    std::copy(ptr, ptr + size, params.logprob.data());
    std::copy(logprob.shape(), logprob.shape() + logprob.ndim(), params.cardinality.data());

    return params;
}

void pybindings_parameters(py::module& root) {
    root.def("MLE",
             &pybindings::learning::parameters::mle_python_wrapper,
             py::return_value_policy::take_ownership,
             py::arg("factor_type"),
             R"doc(
Generates an MLE estimator for the given ``factor_type``.

:param factor_type: A :class:`FactorType <pybnesian.FactorType>`.
:returns: An MLE estimator.
)doc");

    // TODO Fit LinearGaussianCPD with ParamsClass.
    py::class_<LinearGaussianCPD::ParamsClass>(root, "LinearGaussianParams")
        .def(py::init([](VectorXd b, double v) {
                 return LinearGaussianCPD::ParamsClass{/*beta = */ b,
                                                       /*variance = */ v};
             }),
             py::arg("beta"),
             py::arg("variance"),
             R"doc(
Initializes :class:`MLELinearGaussianParams` with the given ``beta`` and ``variance``.
)doc")
        .def_readwrite("beta", &LinearGaussianCPD::ParamsClass::beta, R"doc(
The beta vector of parameters. The beta vector is a :class:`numpy.ndarray` vector of type :class:`numpy.float64` with
size ``len(evidence) + 1``.

``beta[0]`` is always the intercept coefficient and ``beta[i]`` is the corresponding coefficient for the variable
``evidence[i-1]`` for ``i > 0``.
)doc")
        .def_readwrite("variance", &LinearGaussianCPD::ParamsClass::variance, R"doc(
The variance of the linear Gaussian CPD. This is a :class:`float` value.
)doc");

    py::class_<MLE<LinearGaussianCPD>>(root, "MLELinearGaussianCPD", R"doc(
Maximum Likelihood Estimator (MLE) for :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>`.

This class is created using the function :func:`MLE`.

.. doctest::

    >>> from pybnesian import LinearGaussianCPDType, MLE
    >>> mle = MLE(LinearGaussianCPDType())

)doc")
        .def(
            "estimate",
            [](MLE<LinearGaussianCPD> self, const DataFrame& df, std::string var, std::vector<std::string> evidence) {
                return self.estimate(df, var, evidence);
            },
            py::return_value_policy::take_ownership,
            py::arg("df"),
            py::arg("variable"),
            py::arg("evidence"),
            R"doc(
Estimate the parameters of a :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` with the
given ``variable`` and ``evidence``. The parameters are estimated with maximum likelihood estimation on the data ``df``.

:param df: DataFrame to estimate the parameters.
:param variable: Variable of the :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>`.
:param evidence: Evidence of the :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>`.
)doc");

    py::class_<DiscreteFactor::ParamsClass>(root, "DiscreteFactorParams")
        .def(py::init(&numpy_to_discrete_params), py::arg("logprob"), R"doc(
Initializes :class:`DiscreteFactorParams` with a given ``logprob`` (see :attr:`DiscreteFactorParams.logprob`).
)doc")
        .def_property(
            "logprob",
            [](DiscreteFactor::ParamsClass& self) {
                if (self.logprob.rows() > 0) {
                    std::vector<size_t> shape, strides;
                    shape.reserve(self.cardinality.rows());
                    strides.reserve(self.cardinality.rows());

                    shape.push_back(self.cardinality(0));
                    strides.push_back(sizeof(double));
                    for (int i = 1; i < self.cardinality.rows(); ++i) {
                        shape.push_back(self.cardinality(i));
                        strides.push_back(strides[i - 1] * self.cardinality(i - 1));
                    }

                    // https://github.com/pybind/pybind11/issues/1429
                    return py::array_t<double>(py::buffer_info(self.logprob.data(),
                                                               sizeof(double),
                                                               py::format_descriptor<double>::format(),
                                                               self.cardinality.rows(),
                                                               shape,
                                                               strides));
                } else {
                    return py::array_t<double>{};
                }
            },
            [](DiscreteFactor::ParamsClass& self,
               py::array_t<double, py::array::f_style | py::array::forcecast> logprob) {
                auto new_logprob = numpy_to_discrete_params(logprob);
                self.logprob = std::move(new_logprob.logprob);
                self.cardinality = std::move(new_logprob.cardinality);
            },
            R"doc(
A conditional probability table (in log domain). This is a :class:`numpy.ndarray` with ``(len(evidence) + 1)``
dimensions. The first dimension corresponds to the variable being modelled, while the rest corresponds to the evidence
variables.

Each dimension have a shape equal to the cardinality of the corresponding variable and each value is equal
to the log-probability of the assignments for all the variables.

For example, if we are modelling the parameters for the
:class:`DiscreteFactor <pybnesian.DiscreteFactor>` of a variable with two evidence variables:

.. math::

    \text{logprob}[i, j, k] = \log P(\text{variable} = i \mid \text{evidence}_{1} = j, \text{evidence}_{2} = k)

.. testsetup::

    import numpy as np
    import pandas as pd

As logprob defines a conditional probability table, the sum of conditional probabilities must sum 1.

.. doctest::


    >>> from pybnesian import DiscreteFactorType, MLE
    >>> variable = np.random.choice(["a1", "a2", "a3"], size=50, p=[0.5, 0.3, 0.2])
    >>> evidence = np.random.choice(["b1", "b2"], size=50, p=[0.5, 0.5])
    >>> df = pd.DataFrame({'variable': variable, 'evidence': evidence}, dtype="category")
    >>> mle = MLE(DiscreteFactorType())
    >>> params = mle.estimate(df, "variable", ["evidence"])
    >>> assert params.logprob.ndim == 2
    >>> assert params.logprob.shape == (3, 2)
    >>> ss = np.exp(params.logprob).sum(axis=0)
    >>> assert np.all(np.isclose(ss, np.ones(2)))
)doc");

    py::class_<MLE<DiscreteFactor>>(root, "MLEDiscreteFactor", R"doc(
Maximum Likelihood Estimator (MLE) for :class:`DiscreteFactor <pybnesian.DiscreteFactor>`.

This class is created using the function :func:`MLE`.

.. doctest::

    >>> from pybnesian import DiscreteFactorType, MLE
    >>> mle = MLE(DiscreteFactorType())

)doc")
        .def(
            "estimate",
            [](MLE<DiscreteFactor> self, const DataFrame& df, std::string var, std::vector<std::string> evidence) {
                return self.estimate(df, var, evidence);
            },
            py::return_value_policy::take_ownership,
            py::arg("df"),
            py::arg("variable"),
            py::arg("evidence"),
            R"doc(
Estimate the parameters of a :class:`DiscreteFactor <pybnesian.DiscreteFactor>` with the
given ``variable`` and ``evidence``. The parameters are estimated with maximum likelihood estimation on the data ``df``.

:param df: DataFrame to estimate the parameters.
:param variable: Variable of the :class:`DiscreteFactor <pybnesian.DiscreteFactor>`.
:param evidence: Evidence of the :class:`DiscreteFactor <pybnesian.DiscreteFactor>`.
)doc");
}
