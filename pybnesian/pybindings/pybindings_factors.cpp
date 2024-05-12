#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <factors/factors.hpp>
#include <factors/arguments.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <factors/assignment.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <factors/factors.hpp>
#include <models/BayesianNetwork.hpp>
#include <util/util_types.hpp>

namespace py = pybind11;

using factors::Arguments, factors::Args, factors::Kwargs;
using factors::Factor, factors::continuous::LinearGaussianCPD, factors::continuous::CLinearGaussianCPD,
    factors::continuous::HCKDE, factors::continuous::CKDE, factors::discrete::DiscreteFactor;
using factors::FactorType, factors::continuous::LinearGaussianCPDType, factors::continuous::CKDEType,
    factors::discrete::DiscreteFactorType;

using factors::Assignment, factors::AssignmentValue, factors::AssignmentHash;
using models::BayesianNetworkBase, models::ConditionalBayesianNetworkBase;
using util::random_seed_arg;

class PyFactorType : public FactorType {
public:
    using FactorType::FactorType;
    PyFactorType(const PyFactorType&) = delete;
    void operator=(const PyFactorType&) = delete;

    PyFactorType() { m_hash = reinterpret_cast<std::uintptr_t>(nullptr); }

    bool is_python_derived() const override { return true; }

    std::shared_ptr<Factor> new_factor(const BayesianNetworkBase& model,
                                       const std::string& variable,
                                       const std::vector<std::string>& parents,
                                       py::args args = py::args{},
                                       py::kwargs kwargs = py::kwargs{}) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const FactorType*>(this), "new_factor");

        if (override) {
            auto o = override(model.shared_from_this(), variable, parents, *args, **kwargs);

            if (o.is(py::none())) {
                throw std::invalid_argument("FactorType::new_factor cannot return None.");
            }

            try {
                auto f = o.cast<std::shared_ptr<Factor>>();
                Factor::keep_python_alive(f);
                return f;
            } catch (py::cast_error& e) {
                throw std::runtime_error("The returned object of FactorType::new_factor is not a Factor.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"FactorType::new_factor\"");
    }

    std::shared_ptr<Factor> new_factor(const ConditionalBayesianNetworkBase& model,
                                       const std::string& variable,
                                       const std::vector<std::string>& parents,
                                       py::args args = py::args{},
                                       py::kwargs kwargs = py::kwargs{}) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const FactorType*>(this), "new_factor");

        if (override) {
            auto o = override(model.shared_from_this(), variable, parents, *args, **kwargs);

            if (o.is(py::none())) {
                throw std::invalid_argument("FactorType::new_factor cannot return None.");
            }

            try {
                auto f = o.cast<std::shared_ptr<Factor>>();
                Factor::keep_python_alive(f);
                return f;
            } catch (py::cast_error& e) {
                throw std::runtime_error("The returned object of FactorType::new_factor is not a Factor.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"FactorType::new_factor\"");
    }

    std::string ToString() const override {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, FactorType, "__str__", ToString, );
    }

    size_t hash() const override {
        if (m_hash == reinterpret_cast<std::uintptr_t>(nullptr)) {
            py::object o = py::cast(this);
            py::handle ttype = o.get_type();
            // Get the pointer of the Python derived type class.
            // !!!! We have to do this here because in the constructor,
            // "this" is just a FactorType instead of the derived Python class !!!!!!!!!!!!!!!
            m_hash = reinterpret_cast<std::uintptr_t>(ttype.ptr());
        }

        return m_hash;
    }

    py::tuple __getstate__() const override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(static_cast<const FactorType*>(this), "__getstate_extra__");
        if (override) {
            return py::make_tuple(true, override());
        } else {
            return py::make_tuple(false, py::make_tuple());
        }
    }

    static void __setstate__(py::object& self, py::tuple& t) {
        // Call trampoline constructor
        py::gil_scoped_acquire gil;
        auto pyfactortype = py::type::of<FactorType>();
        pyfactortype.attr("__init__")(self);

        auto ptr = self.cast<const FactorType*>();

        auto extra_info = t[0].cast<bool>();
        if (extra_info) {
            py::function override = py::get_override(ptr, "__setstate_extra__");
            if (override) {
                override(t[1]);
            } else {
                py::pybind11_fail("Tried to call function \"FactorType::__setstate_extra__\"");
            }
        }
    }
};

bool is_factortype_subclass(py::handle& factor_type_class) {
    auto factor_type = py::type::of<FactorType>();
    int subclass = PyObject_IsSubclass(factor_type_class.ptr(), factor_type.ptr());
    return subclass == 1;
}

class PyFactor : public Factor {
public:
    using Factor::Factor;

    bool is_python_derived() const override { return true; }

    std::shared_ptr<FactorType> type() const override {
        py::gil_scoped_acquire gil;

        pybind11::function override = pybind11::get_override(static_cast<const Factor*>(this), "type");
        if (override) {
            auto o = override();

            if (o.is(py::none())) {
                throw std::invalid_argument("Factor::type cannot return None.");
            }

            try {
                m_type = o.cast<std::shared_ptr<FactorType>>();
                // Keep the type in the class member, so type_ref() can return a valid reference.
                FactorType::keep_python_alive(m_type);
                return m_type;
            } catch (py::cast_error& e) {
                throw std::runtime_error("The returned object of Factor::type is not a FactorType.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"Factor::type\"");
    }

    FactorType& type_ref() const override {
        auto t = type();
        return *t;
    }

    std::shared_ptr<arrow::DataType> data_type() const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<arrow::DataType>, Factor, data_type, );
    }

    bool fitted() const override { PYBIND11_OVERRIDE_PURE(bool, Factor, fitted, ); }

    void fit(const DataFrame& df) override { PYBIND11_OVERRIDE_PURE(void, Factor, fit, df); }

    VectorXd logl(const DataFrame& df) const override { PYBIND11_OVERRIDE_PURE(VectorXd, Factor, logl, df); }

    double slogl(const DataFrame& df) const override { PYBIND11_OVERRIDE_PURE(double, Factor, slogl, df); }

    std::string ToString() const override { PYBIND11_OVERRIDE_PURE_NAME(std::string, Factor, "__str__", ToString, ); }

    Array_ptr sample(int n,
                     const DataFrame& evidence_values,
                     unsigned int seed = std::random_device{}()) const override {
        PYBIND11_OVERRIDE_PURE(Array_ptr, Factor, sample, n, evidence_values, seed);
    }

    py::tuple __getstate__() const override {
        py::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const Factor*>(this), "__getstate_extra__");
        if (override) {
            auto o = override();
            return py::make_tuple(variable(), evidence(), true, py::make_tuple(o));
        } else {
            return py::make_tuple(variable(), evidence(), false, py::make_tuple());
        }
    }

    static void __setstate__(py::object& self, py::tuple& t) {
        auto v = t[0].cast<std::string>();
        auto p = t[1].cast<std::vector<std::string>>();

        py::gil_scoped_acquire gil;
        auto pyfactor_class = py::type::of<Factor>();
        pyfactor_class.attr("__init__")(self, v, p);

        bool is_extra = t[2].cast<bool>();
        if (is_extra) {
            pybind11::function override = pybind11::get_override(self.cast<const Factor*>(), "__setstate_extra__");
            if (override) {
                auto extra_info = t[3].cast<py::tuple>();
                override(extra_info[0]);
            } else {
                py::pybind11_fail("Tried to call \"Factor::__setstate_extra__\"");
            }
        }
    }

private:
    mutable std::shared_ptr<FactorType> m_type;
};

void pybindings_factors(py::module& root) {
    py::class_<Args>(root, "Args").def(py::init<py::args>(), R"doc(
The :class:`Args` defines a wrapper over `*args`. This class allows to distinguish between a tuple representing `*args`
or a tuple parameter while using :class:`Arguments <pybnesian.Arguments>`.

.. _Example Args/Kwargs:

Example:

.. code-block:: python

    Arguments({ 'a' : ((1, 2), {'param': 3}) })
    # or
    Arguments({ 'a' : Args((1, 2), {'param': 3}) })

defines an `*args` with 2 arguments: a tuple (1, 2) and a dict {'param': 3}. No `**kwargs` is defined.

.. code-block:: python

    Arguments({ 'a' : (Args(1, 2), Kwargs(param = 3)) })

defines an `*args` with 2 arguments: 1 and 2. It also defines a `**kwargs` with param = 3.
)doc");
    py::class_<Kwargs>(root, "Kwargs").def(py::init<py::kwargs>(), R"doc(
The :class:`Kwargs` defines a wrapper over `**kwargs`. This class allows to distinguish between a dict representing
`**kwargs` or a dict parameter while using :class:`Arguments <pybnesian.Arguments>`.

See `Example Args/Kwargs`_.
)doc");
    py::class_<Arguments>(root, "Arguments", R"doc(
The :class:`Arguments` class collects different arguments to construct :class:`Factor <pybnesian.Factor>`.

The :class:`Arguments` object is constructed from a dictionary that associates each :class:`Factor <pybnesian.Factor>`
configuration with a set of arguments.

The keys of the dictionary can be:

    - A 2-tuple (``name``, ``factor_type``) defines arguments for a :class:`Factor <pybnesian.Factor>` of variable
      ``name`` with :class:`FactorType <pybnesian.FactorType>` ``factor_type``.
    - An str defines arguments for a :class:`Factor <pybnesian.Factor>` of variable ``name``.
    - A :class:`FactorType <pybnesian.FactorType>` defines arguments for a :class:`Factor <pybnesian.Factor>` with
      :class:`FactorType <pybnesian.FactorType>` ``factor_type``.

The values of the dictionary can be:

    - A 2-tuple (:class:`Args <pybnesian.Args>`, :class:`Kwargs <pybnesian.Kwargs>`) defines `*args` and `**kwargs`.
    - An :class:`Args <pybnesian.Args>` or tuple ( ... )  defines only `*args`.
    - A :class:`Kwargs <pybnesian.Kwargs>` or dict { ... }: defines only `**kwargs`.

When searching for the defined arguments in :class:`Arguments <pybnesian.Arguments>` for a given factor with ``name``
and ``factor_type``, the most specific configurations have preference over more general ones.

    - If a 2-tuple (``name``, ``factor_type``) configuration exists, the corresponding arguments are returned.
    - Else, if a ``name`` configuration exists, the corresponding arguments are returned.
    - Else, if a ``factor_type`` configuration exists, the corresponding arguments are returned.
    - Else, empty `*args` and `**kwargs` are returned.
)doc")
        .def(py::init<>(), R"doc(
Initializes an empty :class:`Arguments`.
)doc")
        .def(py::init<py::dict>(), py::arg("dict_arguments"), R"doc(
Initializes a new :class:`Arguments` with the given configurations and arguments.

:param dict_arguments: A dictionary { configurations : arguments} that associates each
    :class:`Factor <pybnesian.Factor>` configuration with a set of arguments.
)doc")
        .def("__repr__", [](const Arguments&) { return "Arguments"; })
        .def("args", &Arguments::args, py::arg("node"), py::arg("node_type"), R"doc(
Returns the \*args and \*\*kwargs defined for a ``node`` with a given ``node_type``.

:param node: A node name.
:param node_type: :class:`FactorType <pybnesian.FactorType>` for ``node``.
:returns: 2-tuple containing ``(*args, **kwargs)``
)doc");

    py::class_<FactorType, PyFactorType, std::shared_ptr<FactorType>> factor_type(root, "FactorType", R"doc(
A representation of a :class:`Factor` type.
)doc");

    py::class_<Factor, PyFactor, std::shared_ptr<Factor>> factor(root, "Factor");

    factor_type
        .def(py::init<>(), R"doc(Initializes a new :class:`FactorType`)doc")
        // The equality operator do not compile in GCC, so it is implemented with lambdas:
        // https://github.com/pybind/pybind11/issues/1487
        .def(
            "__eq__",
            [](const FactorType& self, const FactorType& other) { return self == other; },
            py::arg("other"),
            py::is_operator())
        .def(
            "__ne__",
            [](const FactorType& self, const FactorType& other) { return self != other; },
            py::arg("other"),
            py::is_operator())
        // .def(py::self == py::self)
        // .def(py::self != py::self)
        .def("__getstate__", [](const FactorType& self) { return self.__getstate__(); })
        // Setstate for pyderived type
        .def("__setstate__", [](py::object& self, py::tuple& t) { PyFactorType::__setstate__(self, t); })
        .def("__hash__", &FactorType::hash)
        .def("__repr__", [](const FactorType& self) { return self.ToString(); })
        .def("__str__", [](const FactorType& self) { return self.ToString(); });

    {
        py::options options;
        options.disable_function_signatures();
        factor_type
            .def("new_factor",
                 py::overload_cast<const ConditionalBayesianNetworkBase&,
                                   const std::string&,
                                   const std::vector<std::string>&,
                                   py::args,
                                   py::kwargs>(&FactorType::new_factor, py::const_),
                 py::arg("model"),
                 py::arg("variable"),
                 py::arg("evidence"))
            .def("new_factor",
                 py::overload_cast<const BayesianNetworkBase&,
                                   const std::string&,
                                   const std::vector<std::string>&,
                                   py::args,
                                   py::kwargs>(&FactorType::new_factor, py::const_),
                 py::arg("model"),
                 py::arg("variable"),
                 py::arg("evidence"),
                 R"doc(
new_factor(self: pybnesian.FactorType, model: BayesianNetworkBase or ConditionalBayesianNetworkBase, variable: str, evidence: List[str], *args, **kwargs) -> pybnesian.Factor

Create a new corresponding :class:`Factor` for a ``model`` with the given ``variable`` and ``evidence``.

Note that ``evidence`` might be different from ``model.parents(variable)``.

:param model: The model that will contain the :class:`Factor`.
:param variable: Variable name.
:param evidence: List of evidence variable names.
:param args: Additional arguments to construct the :class:`Factor`.
:param kwargs: Additional keyword arguments used to construct the :class:`Factor`.
:returns: A corresponding :class:`Factor` with the given ``variable`` and ``evidence``.
)doc");
    }

    factor
        .def(py::init<const std::string&, const std::vector<std::string>&>(),
             py::arg("variable"),
             py::arg("evidence"),
             R"doc(
Initializes a new :class:`Factor` with a given ``variable`` and ``evidence``.

:param variable: Variable name.
:param evidence: List of evidence variable names.
)doc")
        .def("variable", &Factor::variable, R"doc(
Gets the variable modelled by this :class:`Factor`.

:returns: Variable name.
)doc")
        .def("evidence", &Factor::evidence, R"doc(
Gets the evidence variable list.

:returns: Evidence variable list.
)doc")
        .def("fitted", &Factor::fitted, R"doc(
Checks whether the factor is fitted.

:returns: True if the factor is fitted, False otherwise.
)doc")
        .def("type", &Factor::type, R"doc(
Returns the corresponding :class:`FactorType` of this :class:`Factor`.

:returns: :class:`FactorType` corresponding to this :class:`Factor`.
)doc")
        .def("data_type", &Factor::data_type, R"doc(
Returns the :class:`pyarrow.DataType` that represents the type of data handled by the :class:`Factor`.

For a continuous Factor, this usually returns :func:`pyarrow.float64` or :func:`pyarrow.float32`. The discrete factor
is usually a :func:`pyarrow.dictionary`.

:returns: the :class:`pyarrow.DataType` physical data type representation of the :class:`Factor`.
)doc")
        .def("fit", &Factor::fit, py::arg("df"), R"doc(
Fits the :class:`Factor` with the data in ``df``.

:param df: DataFrame to fit the :class:`Factor`.
)doc")
        .def("logl",
             &Factor::logl,
             py::arg("df"),
             R"doc(
Returns the log-likelihood of each instance in the DataFrame ``df``.

:param df: DataFrame to compute the log-likelihood.
:returns: A :class:`numpy.ndarray` vector with dtype :class:`numpy.float64`, where the i-th value is the log-likelihod
          of the i-th instance of ``df``.
)doc",
             py::return_value_policy::take_ownership)
        .def("slogl", &Factor::slogl, py::arg("df"), R"doc(
Returns the sum of the log-likelihood of each instance in the DataFrame ``df``. That is, the sum of the result of
:func:`Factor.logl`.

:param df: DataFrame to compute the sum of the log-likelihood.
:returns: The sum of log-likelihood for DataFrame ``df``.
)doc")
        .def(
            "sample",
            [](const Factor& self,
               int n,
               std::optional<const DataFrame> evidence_values,
               std::optional<unsigned int> seed) {
                if (evidence_values)
                    return self.sample(n, *evidence_values, random_seed_arg(seed));
                else
                    return self.sample(n, DataFrame(), random_seed_arg(seed));
            },
            py::arg("n"),
            py::arg("evidence_values") = std::nullopt,
            py::arg("seed") = std::nullopt,
            R"doc(
Samples ``n`` values from this :class:`Factor`. This method returns a :class:`pyarrow.Array` with ``n`` values with
the same type returned by :func:`Factor.data_type`.

If this :class:`Factor` has evidence variables, the DataFrame ``evidence_values`` contains ``n`` instances for each
evidence variable. Each sampled instance must be conditioned on ``evidence_values``.

:param n: Number of instances to sample.
:param evidence_values: DataFrame of evidence values to condition the sampling.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
)doc")
        .def("save", &Factor::save, py::arg("filename"), R"doc(
Saves the :class:`Factor` in a pickle file with the given name.

:param filename: File name of the saved graph.
)doc")
        .def("__str__", &Factor::ToString)
        .def("__repr__", &Factor::ToString)
        .def("__getstate__", [](const Factor& self) { return self.__getstate__(); })
        .def("__setstate__", [](py::object& self, py::tuple& t) { PyFactor::__setstate__(self, t); });

    py::class_<UnknownFactorType, FactorType, std::shared_ptr<UnknownFactorType>>(root, "UnknownFactorType", R"doc(
:class:`UnknownFactorType` is the representation of an unknown :class:`FactorType`. This factor type is assigned by
default to each node in an heterogeneous Bayesian network.
)doc")
        .def(py::init(&UnknownFactorType::get), R"doc(
Instantiates an :class:`UnknownFactorType`.
)doc")
        .def(py::pickle([](const UnknownFactorType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return UnknownFactorType::get(); }));

    py::class_<LinearGaussianCPDType, FactorType, std::shared_ptr<LinearGaussianCPDType>>(
        root, "LinearGaussianCPDType", R"doc(
:class:`LinearGaussianCPDType` is the corresponding CPD type of :class:`LinearGaussianCPD`.
)doc")
        .def(py::init(&LinearGaussianCPDType::get), R"doc(
Instantiates a :class:`LinearGaussianCPDType`.
)doc")
        .def(py::pickle([](const LinearGaussianCPDType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return LinearGaussianCPDType::get(); }));

    py::class_<LinearGaussianCPD, Factor, std::shared_ptr<LinearGaussianCPD>>(root, "LinearGaussianCPD", R"doc(
This is a linear Gaussian CPD:

.. math::

    \hat{f}(\text{variable} \mid \text{evidence}) = \mathcal{N}(\text{variable};
    \text{beta}_{0} + \sum_{i=1}^{|\text{evidence}|} \text{beta}_{i}\cdot \text{evidence}_{i}, \text{variance})

It is parametrized by the following attributes:

:ivar beta: The beta vector.
:ivar variance: The variance.

.. testsetup::

    import numpy as np

.. doctest::

    >>> from pybnesian import LinearGaussianCPD
    >>> cpd = LinearGaussianCPD("a", ["b"])
    >>> assert not cpd.fitted()
    >>> cpd.beta
    array([], dtype=float64)
    >>> cpd.beta = np.asarray([1., 2.])
    >>> assert not cpd.fitted()
    >>> cpd.variance = 0.5
    >>> assert cpd.fitted()
    >>> cpd.beta
    array([1., 2.])
    >>> cpd.variance
    0.5

)doc")
        .def(py::init<std::string, std::vector<std::string>>(),
             py::arg("variable"),
             py::arg("evidence"),
             R"doc(
Initializes a new :class:`LinearGaussianCPD` with a given ``variable`` and ``evidence``.

The :class:`LinearGaussianCPD` is left unfitted.

:param variable: Variable name.
:param evidence: List of evidence variable names.
)doc")
        .def(py::init<std::string, std::vector<std::string>, VectorXd, double>(),
             py::arg("variable"),
             py::arg("evidence"),
             py::arg("beta"),
             py::arg("variance"),
             R"doc(
Initializes a new :class:`LinearGaussianCPD` with a given ``variable`` and ``evidence``.

The :class:`LinearGaussianCPD` is fitted with ``beta``  and ``variance``.

:param variable: Variable name.
:param evidence: List of evidence variable names.
:param beta: Vector of parameters.
:param variance: Variance of the linear Gaussian CPD.
)doc")
        .def_property("beta", &LinearGaussianCPD::beta, &LinearGaussianCPD::set_beta, R"doc(
The beta vector of parameters. The beta vector is a :class:`numpy.ndarray` vector of type :class:`numpy.float64` with
size ``len(evidence) + 1``.

``beta[0]`` is always the intercept coefficient and ``beta[i]`` is the corresponding coefficient for the variable
``evidence[i-1]`` for ``i > 0``.
)doc")
        .def_property("variance",
                      &LinearGaussianCPD::variance,
                      &LinearGaussianCPD::set_variance,
                      R"doc(The variance of the linear Gaussian CPD. This is a :class:`float` value.)doc")
        .def("cdf", &LinearGaussianCPD::cdf, py::return_value_policy::take_ownership, py::arg("df"), R"doc(
Returns the cumulative distribution function values of each instance in the DataFrame ``df``.

:param df: DataFrame to compute the log-likelihood.
:returns: A :class:`numpy.ndarray` vector with dtype :class:`numpy.float64`, where the i-th value is the cumulative
          distribution function value of the i-th instance of ``df``.
)doc")
        .def(py::pickle([](const LinearGaussianCPD& self) { return self.__getstate__(); },
                        [](py::tuple t) { return LinearGaussianCPD::__setstate__(t); }));

    py::class_<CKDEType, FactorType, std::shared_ptr<CKDEType>>(root, "CKDEType", R"doc(
:class:`CKDEType` is the corresponding CPD type of :class:`CKDE`.
)doc")
        .def(py::init(&CKDEType::get), R"doc(
Instantiates a :class:`CKDEType`.
)doc")
        .def(py::pickle([](const CKDEType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return CKDEType::get(); }));

    py::class_<CKDE, Factor, std::shared_ptr<CKDE>>(root, "CKDE", R"doc(
A conditional kernel density estimator (CKDE) is the ratio of two KDE models [Semiparametric]_:

.. math::

    \hat{f}(\text{variable} \mid \text{evidence}) =
    \frac{\hat{f}_{K}(\text{variable}, \text{evidence})}{\hat{f}_{K}(\text{evidence})}

where :math:`\hat{f}_{K}` is a :class:`KDE` estimation.
)doc")
        .def(py::init<std::string, std::vector<std::string>>(),
             py::arg("variable"),
             py::arg("evidence"),
             R"doc(
Initializes a new :class:`CKDE` with a given ``variable`` and ``evidence``.

:param variable: Variable name.
:param evidence: List of evidence variable names.
)doc")
        .def(py::init<>([](std::string variable,
                           std::vector<std::string> evidence,
                           std::shared_ptr<BandwidthSelector> bandwidth_selector) {
                 return CKDE(variable, evidence, BandwidthSelector::keep_python_alive(bandwidth_selector));
             }),
             py::arg("variable"),
             py::arg("evidence"),
             py::arg("bandwidth_selector"),
             R"doc(
Initializes a new :class:`CKDE` with a given ``variable`` and ``evidence``.

:param variable: Variable name.
:param evidence: List of evidence variable names.
:param bandwidth_selector: Procedure to fit the bandwidth.
)doc")
        .def("num_instances", &CKDE::num_instances, R"doc(
Gets the number of training instances (:math:`N`).

:returns: Number of training instances.
)doc")
        .def("kde_joint", &CKDE::kde_joint, py::return_value_policy::reference_internal, R"doc(
Gets the joint :math:`\hat{f}_{K}(\text{variable}, \text{evidence})` :class:`KDE` model.

:returns: Joint KDE model.
)doc")
        .def("kde_marg", &CKDE::kde_marg, py::return_value_policy::reference_internal, R"doc(
Gets the marginalized :math:`\hat{f}_{K}(\text{evidence})` :class:`KDE` model.

:returns: Marginalized KDE model.
)doc")
        .def("cdf", &CKDE::cdf, py::return_value_policy::take_ownership, py::arg("df"), R"doc(
Returns the cumulative distribution function values of each instance in the DataFrame ``df``.

:param df: DataFrame to compute the log-likelihood.
:returns: A :class:`numpy.ndarray` vector with dtype :class:`numpy.float64`, where the i-th value is the cumulative
          distribution function value of the i-th instance of ``df``.
)doc")
        .def(py::pickle([](const CKDE& self) { return self.__getstate__(); },
                        [](py::tuple t) { return CKDE::__setstate__(t); }));

    py::class_<DiscreteFactorType, FactorType, std::shared_ptr<DiscreteFactorType>>(root, "DiscreteFactorType", R"doc(
:class:`DiscreteFactorType` is the corresponding CPD type of :class:`DiscreteFactor`.
)doc")
        .def(py::init(&DiscreteFactorType::get), R"doc(
Instantiates a :class:`DiscreteFactorType`.
)doc")
        .def(py::pickle([](const DiscreteFactorType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return DiscreteFactorType::get(); }));

    py::class_<DiscreteFactor, Factor, std::shared_ptr<DiscreteFactor>>(root, "DiscreteFactor", R"doc(
This is a discrete factor implemented as a conditional probability table (CPT).
)doc")
        .def(py::init<std::string, std::vector<std::string>>(), py::arg("variable"), py::arg("evidence"), R"doc(
Initializes a new :class:`DiscreteFactor` with a given ``variable`` and ``evidence``.

:param variable: Variable name.
:param evidence: List of evidence variable names.
)doc")
        .def(py::pickle([](const DiscreteFactor& self) { return self.__getstate__(); },
                        [](py::tuple t) { return DiscreteFactor::__setstate__(t); }));

    py::class_<Assignment>(root, "Assignment", R"doc(
:class:`Assignment <pybnesian.Assignment>` represents the assignment of values to a set of variables.
)doc")
        .def(py::init<std::unordered_map<std::string, AssignmentValue>>(), py::arg("assignments"), R"doc(
Initializes an :class:`Assignment <pybnesian.Assignment>` from a dict that contains the value for each variable.
The key of the dict is the name of the variable, and the value of the dict can be an ``str`` or a ``float`` value.

:param assignments: Value assignments for each variable.
)doc")
        .def("value", &Assignment::value, py::arg("variable"), R"doc(
Returns the assignment value for a given ``variable``.

:param variable: Variable name.
:returns: Value assignment of the variable.
)doc")
        .def(
            "has_variables",
            [](const Assignment& self, const std::vector<std::string> vars) {
                return self.has_variables(vars.begin(), vars.end());
            },
            py::arg("variables"),
            R"doc(
Checks whether the :class:`Assignment <pybnesian.Assignment>` contains assignments for all the ``variables``.

:param variables: Variable names.
:returns: True if the :class:`Assignment <pybnesian.Assignment>` contains values for all the given variables,
    False otherwise.
)doc")
        .def("empty", &Assignment::empty, R"doc(
Checks whether the :class:`Assignment <pybnesian.Assignment>` does not have assignments.

:returns: True if the :class:`Assignment <pybnesian.Assignment>` does not have assignments, False otherwise.
)doc")
        .def("size", &Assignment::size, R"doc(
Gets the number of assignments in the :class:`Assignment <pybnesian.Assignment>`.

:returns: The number of assignments.
)doc")
        .def(
            "insert",
            [](Assignment& self, const std::string& key, const AssignmentValue value) { self.insert({key, value}); },
            py::arg("variable"),
            py::arg("value"),
            R"doc(
Inserts a new assignment for a ``variable`` with a ``value``.

:param variable: Variable name.
:param value: Value (``str`` or ``float``) for the variable.
)doc")
        .def("remove", &Assignment::erase, py::arg("variable"), R"doc(
Removes the assignment for the ``variable``.

:param variable: Variable name.
)doc")
        .def("__iter__", [](Assignment& self) { return py::make_iterator(self.begin(), self.end()); })
        .def("__hash__", &Assignment::hash)
        .def("__str__", &Assignment::ToString)
        .def("__repr__", &Assignment::ToString)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::pickle([](const Assignment& self) { return self.__getstate__(); },
                        [](py::object& o) { return Assignment::__setstate__(o); }));

    py::class_<CLinearGaussianCPD, Factor, std::shared_ptr<CLinearGaussianCPD>>(root, "CLinearGaussianCPD", R"doc(
A conditional linear Gaussian CPD defines a :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` for each discrete
configuration of its parents.
)doc")
        .def(py::init<std::string, std::vector<std::string>>(), py::arg("variable"), py::arg("evidence"), R"doc(
Initializes a new :class:`CLinearGaussianCPD` with a given ``variable`` and ``evidence``.

The :class:`CLinearGaussianCPD` is left unfitted.

:param variable: Variable name.
:param evidence: List of evidence variable names.
)doc")
        .def(py::init<std::string, std::vector<std::string>, VectorXd, double>(),
             py::arg("variable"),
             py::arg("evidence"),
             py::arg("beta"),
             py::arg("variance"),
             R"doc(
Initializes a new :class:`CLinearGaussianCPD` with a given ``variable`` and ``evidence``. Each 
:class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` will be constructed using the provided ``beta`` and
``variance``.

Note that :class:`CLinearGaussianCPD` is left unfitted because some data is needed to extract the categories of the
discrete variables. You should call :func:`fit <pybnesian.Factor.fit>`.

:param variable: Variable name.
:param evidence: List of evidence variable names.
:param beta: Vector of parameters for each :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>`.
:param variance: Variance for each :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>`.
)doc")
        .def(py::init<std::string,
                      std::vector<std::string>,
                      std::unordered_map<Assignment, std::tuple<VectorXd, double>, AssignmentHash>>(),
             py::arg("variable"),
             py::arg("evidence"),
             py::arg("args"),
             R"doc(
Initializes a new :class:`CLinearGaussianCPD` with a given ``variable`` and ``evidence``. The
:class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` of each discrete configuration can be constructed using
different ``beta`` and ``variance``.

Note that :class:`CLinearGaussianCPD` is left unfitted because some data is needed to extract the categories of the
discrete variables. You should call :func:`fit <pybnesian.Factor.fit>`. If some discrete
configuration is not provided, the :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` will be fitted with 
:func:`LinearGaussianCPD.fit <pybnesian.Factor.fit>`

:param variable: Variable name.
:param evidence: List of evidence variable names.
:param args: Dict of of ``beta`` and ``variance`` for each discrete :class:`Assignment <pybnesian.Assignment>`.
)doc")
        .def("conditional_factor",
             &CLinearGaussianCPD::conditional_factor,
             py::return_value_policy::reference_internal,
             py::arg("assignment"),
             R"doc(
Return the corresponding :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` for the given discrete ``assignment``

:param assignment: A discrete :class:`Assignment <pybnesian.Assignment>`.
)doc")
        .def(py::pickle([](const CLinearGaussianCPD& self) { return self.__getstate__(); },
                        [](py::tuple t) { return CLinearGaussianCPD::__setstate__(t); }));

    py::class_<HCKDE, Factor, std::shared_ptr<HCKDE>>(root, "HCKDE", R"doc(
The hybrid conditional kernel density estimation (HCKDE) [HybridSemiparametric]_ defines a
:class:`CKDE <pybnesian.CKDE>` for each discrete configuration of its parents.
)doc")
        .def(py::init<std::string, std::vector<std::string>>(), py::arg("variable"), py::arg("evidence"), R"doc(
Initializes a new :class:`HCKDE` with a given ``variable`` and ``evidence``.

The :class:`HCKDE` is left unfitted.

:param variable: Variable name.
:param evidence: List of evidence variable names.
)doc")
        .def(py::init<>([](std::string variable,
                           std::vector<std::string> evidence,
                           std::shared_ptr<BandwidthSelector> bandwidth_selector) {
                 return HCKDE(variable, evidence, BandwidthSelector::keep_python_alive(bandwidth_selector));
             }),
             py::arg("variable"),
             py::arg("evidence"),
             py::arg("bandwidth_selector"),
             R"doc(
Initializes a new :class:`HCKDE` with a given ``variable`` and ``evidence``. Each :class:`HCKDE` will be constructed
using the provided ``bandwidth_selector``.

Note that :class:`HCKDE` is left unfitted because some data is needed to extract the categories of the discrete
variables and to fit each :class:`CKDE <pybnesian.CKDE>`. You should call :func:`HCKDE.fit <pybnesian.Factor.fit>`.

:param variable: Variable name.
:param evidence: List of evidence variable names.
:param bandwidth_selector: A :class:`BandwidthSelector <pybnesian.BandwidthSelector>` to use for each :class:`CKDE <pybnesian.CKDE>`.
)doc")
        .def(
            py::init<>([](std::string variable,
                          std::vector<std::string> evidence,
                          std::unordered_map<Assignment, std::tuple<std::shared_ptr<BandwidthSelector>>, AssignmentHash>
                              args) {
                for (auto& arg : args) {
                    BandwidthSelector::keep_python_alive(std::get<0>(arg.second));
                }

                return HCKDE(variable, evidence, args);
            }),
            py::arg("variable"),
            py::arg("evidence"),
            py::arg("bandwidth_selector"),
            R"doc(
Initializes a new :class:`HCKDE` with a given ``variable`` and ``evidence``. The :class:`CKDE <pybnesian.CKDE>` of each
discrete configuration can be constructed using different ``bandwidth_selector``.

Note that :class:`HCKDE` is left unfitted because some data is needed to extract the categories of the discrete
variables and to fit each :class:`CKDE <pybnesian.CKDE>`. You should call :func:`HCKDE.fit <pybnesian.Factor.fit>`. If
some discrete configuration is not provided, the :class:`CKDE <pybnesian.CKDE>` will be fitted with the
:class:`NormalRereferenceRule <pybnesian.NormalReferenceRule>`.

:param variable: Variable name.
:param evidence: List of evidence variable names.
:param args: Dict of ``bandwidth_selectors`` for each discrete :class:`Assignment <pybnesian.Assignment>`.
)doc")
        .def("conditional_factor",
             &HCKDE::conditional_factor,
             py::return_value_policy::reference_internal,
             py::arg("assignment"),
             R"doc(
Return the corresponding :class:`CKDE <pybnesian.CKDE>` for the given discrete ``assignment``

:param assignment: A discrete :class:`Assignment <pybnesian.Assignment>`.
)doc")
        .def(py::pickle([](const HCKDE& self) { return self.__getstate__(); },
                        [](py::tuple t) { return HCKDE::__setstate__(t); }));
}
