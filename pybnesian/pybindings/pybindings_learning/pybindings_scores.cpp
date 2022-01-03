#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <learning/scores/scores.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/bge.hpp>
#include <learning/scores/bde.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>
#include <learning/scores/validated_likelihood.hpp>
#include <util/util_types.hpp>

namespace py = pybind11;

using learning::scores::Score, learning::scores::ValidatedScore, learning::scores::BIC, learning::scores::BGe,
    learning::scores::BDe, learning::scores::CVLikelihood, learning::scores::HoldoutLikelihood,
    learning::scores::ValidatedLikelihood;

using learning::scores::DynamicScore, learning::scores::DynamicBIC, learning::scores::DynamicBGe,
    learning::scores::DynamicBDe, learning::scores::DynamicCVLikelihood, learning::scores::DynamicHoldoutLikelihood,
    learning::scores::DynamicValidatedLikelihood;

using util::random_seed_arg;

template <typename CppClass, typename PyClass>
void register_Score_methods(PyClass& pyclass) {
    pyclass
        .def(
            "local_score",
            [](const CppClass& self, const ConditionalBayesianNetworkBase& m, const std::string& variable) {
                return self.local_score(m, variable);
            },
            py::arg("model"),
            py::arg("variable"))
        .def(
            "local_score",
            [](const CppClass& self, const BayesianNetworkBase& m, const std::string& variable) {
                return self.local_score(m, variable);
            },
            py::arg("model"),
            py::arg("variable"),
            R"doc(
Returns the local score value of a node ``variable`` in the ``model``.

For example:

.. code-block:: python

    >>> score.local_score(m, "a")

returns the local score of node ``"a"`` in the model ``m``. This method assumes that the parents in the score are
``m.parents("a")`` and its node type is ``m.node_type("a")``.

:param model: Bayesian network model.
:param variable: A variable name.
:returns: Local score value of ``node`` in the ``model``.
)doc")
        .def(
            "local_score",
            [](const CppClass& self,
               const ConditionalBayesianNetworkBase& m,
               const std::string& variable,
               const std::vector<std::string>& evidence) { return self.local_score(m, variable, evidence); },
            py::arg("model"),
            py::arg("variable"),
            py::arg("evidence"))
        .def(
            "local_score",
            [](const CppClass& self,
               const BayesianNetworkBase& m,
               const std::string& variable,
               const std::vector<std::string>& evidence) { return self.local_score(m, variable, evidence); },
            py::arg("model"),
            py::arg("variable"),
            py::arg("evidence"),
            R"doc(
Returns the local score value of a node ``variable`` in the ``model`` if it had ``evidence`` as parents.

For example:

.. code-block:: python

    >>> score.local_score(m, "a", ["b"])

returns the local score of node ``"a"`` in the model ``m``, with ``["b"]`` as parents. This method assumes that the node
type of ``"a"`` is ``m.node_type("a")``.

:param model: Bayesian network model.
:param variable: A variable name.
:param evidence: A list of parent names.
:returns: Local score value of ``node`` in the ``model`` with ``evidence`` as parents.
)doc")
        .def(
            "local_score_node_type",
            [](CppClass& self,
               const BayesianNetworkBase& m,
               const std::shared_ptr<FactorType>& variable_type,
               const std::string& variable,
               const std::vector<std::string>& evidence) {
                return self.local_score(m, variable_type, variable, evidence);
            },
            py::arg("model"),
            py::arg("variable_type").none(false),
            py::arg("variable"),
            py::arg("evidence"),
            R"doc(
Returns the local score value of a node ``variable`` in the ``model`` if its conditional distribution were a
``variable_type`` factor and it had ``evidence`` as parents.

For example:

.. code-block:: python

    >>> score.local_score(m, LinearGaussianCPDType(), "a", ["b"])

returns the local score of node ``"a"`` in the model ``m``, with ``["b"]`` as parents assuming the conditional
distribution of ``"a"`` is a :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>`.

:param model: Bayesian network model.
:param variable_type: The :class:`FactorType <pybnesian.FactorType>` of the node ``variable``.
:param variable: A variable name.
:param evidence: A list of parent names.
:returns: Local score value of ``node`` in the ``model`` with ``evidence`` as parents and ``variable_type`` as
          conditional distribution.
)doc")
        .def("data", &Score::data, R"doc(
Returns the DataFrame used to calculate the score and local scores.

:returns: DataFrame used to calculate the score. If the score do not use data, it returns None.
)doc");

    {
        py::options options;
        options.disable_function_signatures();

        pyclass
            .def(
                "score",
                [](const CppClass& self, const ConditionalBayesianNetworkBase& m) { return self.score(m); },
                py::arg("model"))
            .def(
                "score",
                [](const CppClass& self, const BayesianNetworkBase& m) { return self.score(m); },
                py::arg("model"),
                R"doc(
score(self: pybnesian.Score, model: BayesianNetworkBase or ConditionalBayesianNetworkBase) -> float

Returns the score value of the ``model``.

:param model: Bayesian network model.
:returns: Score value of ``model``.
)doc");
    }
}

template <typename CppClass, typename PyClass>
void register_ValidatedScore_methods(PyClass& pyclass) {
    pyclass
        .def(
            "vlocal_score",
            [](const CppClass& self, const ConditionalBayesianNetworkBase& m, const std::string& variable) {
                return self.vlocal_score(m, variable);
            },
            py::arg("model"),
            py::arg("variable"))
        .def(
            "vlocal_score",
            [](const CppClass& self, const BayesianNetworkBase& m, const std::string& variable) {
                return self.vlocal_score(m, variable);
            },
            py::arg("model"),
            py::arg("variable"),
            R"doc(
vlocal_score(self: pybnesian.ValidatedScore, model: BayesianNetworkBase or ConditionalBayesianNetworkBase, variable: str) -> float

Returns the validated local score value of a node ``variable`` in the ``model``.

For example:

.. code-block:: python

    >>> score.local_score(m, "a")

returns the validated local score of node ``"a"`` in the model ``m``. This method assumes that the parents of ``"a"`` are
``m.parents("a")`` and its node type is ``m.node_type("a")``.

:param model: Bayesian network model.
:param variable: A variable name.
:returns: Validated local score value of ``node`` in the ``model``.
)doc")
        .def("vlocal_score",
             [](const CppClass& self,
                const ConditionalBayesianNetworkBase& m,
                const std::string& variable,
                const std::vector<std::string>& evidence) { return self.vlocal_score(m, variable, evidence); })
        .def(
            "vlocal_score",
            [](const CppClass& self,
               const BayesianNetworkBase& m,
               const std::string& variable,
               const std::vector<std::string>& evidence) { return self.vlocal_score(m, variable, evidence); },
            py::arg("model"),
            py::arg("variable"),
            py::arg("evidence"),
            R"doc(
vlocal_score(self: pybnesian.ValidatedScore, model: BayesianNetworkBase or ConditionalBayesianNetworkBase, variable: str, evidence: List[str]) -> float

Returns the validated local score value of a node ``variable`` in the ``model`` if it had ``evidence`` as parents.

For example:

.. code-block:: python

    >>> score.local_score(m, "a", ["b"])

returns the validated local score of node ``"a"`` in the model ``m``, with ``["b"]`` as parents. This method assumes
that the node type of ``"a"`` is ``m.node_type("a")``.

:param model: Bayesian network model.
:param variable: A variable name.
:param evidence: A list of parent names.
:returns: Validated local score value of ``node`` in the ``model`` with ``evidence`` as parents.
)doc")
        .def(
            "vlocal_score_node_type",
            [](CppClass& self,
               const BayesianNetworkBase& m,
               const std::shared_ptr<FactorType>& variable_type,
               const std::string& variable,
               const std::vector<std::string>& evidence) {
                return self.vlocal_score(m, variable_type, variable, evidence);
            },
            py::arg("model"),
            py::arg("variable_type").none(false),
            py::arg("variable"),
            py::arg("evidence"),
            R"doc(
Returns the validated local score value of a node ``variable`` in the ``model`` if its conditional distribution were a
``variable_type`` factor and it had ``evidence`` as parents.

For example:

.. code-block:: python

    >>> score.vlocal_score(m, LinearGaussianCPDType(), "a", ["b"])

returns the validated local score of node ``"a"`` in the model ``m``, with ``["b"]`` as parents assuming the conditional
distribution of ``"a"`` is a :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>`.

:param model: Bayesian network model.
:param variable_type: The :class:`FactorType <pybnesian.FactorType>` of the node ``variable``.
:param variable: A variable name.
:param evidence: A list of parent names.
:returns: Validated local score value of ``node`` in the ``model`` with ``evidence`` as parents and ``variable_type`` as
          conditional distribution.
)doc");

    {
        py::options options;
        options.disable_function_signatures();

        pyclass
            .def(
                "vscore",
                [](const CppClass& self, const ConditionalBayesianNetworkBase& m) { return self.vscore(m); },
                py::arg("model"))
            .def(
                "vscore",
                [](const CppClass& self, const BayesianNetworkBase& m) { return self.vscore(m); },
                py::arg("model"),
                R"doc(
vscore(self: pybnesian.ValidatedScore, model: BayesianNetworkBase or ConditionalBayesianNetworkBase) -> float

Returns the validated score value of the ``model``.

:param model: Bayesian network model.
:returns: Validated score value of ``model``.
)doc");
    }
}

template <typename ScoreBase = Score>
class PyScore : public ScoreBase {
public:
    using ScoreBase::local_score;
    using ScoreBase::ScoreBase;

    double score(const BayesianNetworkBase& model) const override {
        {
            py::gil_scoped_acquire gil;
            py::function override = pybind11::get_override(static_cast<const ScoreBase*>(this), "score");
            if (override) {
                auto o = override(model.shared_from_this());
                try {
                    return std::move(o).cast<double>();
                } catch (py::cast_error& e) {
                    throw std::runtime_error("The returned object of Score::score is not a double.");
                }
            }
        }

        return ScoreBase::score(model);
    }

    double local_score(const BayesianNetworkBase& model,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override {
        PYBIND11_OVERRIDE_PURE(double,      /* Return type */
                               ScoreBase,   /* Parent class */
                               local_score, /* Name of function in C++ (must match Python name) */
                               model.shared_from_this(),
                               variable, /* Argument(s) */
                               parents);
    }

    double local_score(const BayesianNetworkBase& model,
                       const std::shared_ptr<FactorType>& node_type,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override {
        PYBIND11_OVERRIDE_PURE_NAME(double,    /* Return type */
                                    ScoreBase, /* Parent class */
                                    "local_score_node_type",
                                    local_score, /* Name of function in C++ (must match Python name) */
                                    model.shared_from_this(),
                                    node_type,
                                    variable, /* Argument(s) */
                                    parents);
    }

    std::string ToString() const override {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, /* Return type */
                                    ScoreBase,   /* Parent class */
                                    "__str__",
                                    ToString, /* Name of function in C++ (must match Python name) */
        );
    }

    bool has_variables(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(bool,          /* Return type */
                               ScoreBase,     /* Parent class */
                               has_variables, /* Name of function in C++ (must match Python name) */
                               name           /* Argument(s) */
        );
    }

    bool has_variables(const std::vector<std::string>& cols) const override {
        PYBIND11_OVERRIDE_PURE(bool,          /* Return type */
                               ScoreBase,     /* Parent class */
                               has_variables, /* Name of function in C++ (must match Python name) */
                               cols           /* Argument(s) */
        );
    }

    bool compatible_bn(const BayesianNetworkBase& model) const override {
        PYBIND11_OVERRIDE_PURE(bool,          /* Return type */
                               ScoreBase,     /* Parent class */
                               compatible_bn, /* Name of function in C++ (must match Python name) */
                               &model         /* Argument(s) */
        );
    }

    bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
        PYBIND11_OVERRIDE_PURE(bool,          /* Return type */
                               ScoreBase,     /* Parent class */
                               compatible_bn, /* Name of function in C++ (must match Python name) */
                               &model         /* Argument(s) */
        );
    }

    DataFrame data() const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const ScoreBase*>(this), "data");

        if (override) {
            auto o = override();

            if (o.is(py::none())) {
                return DataFrame();
            }

            try {
                return o.cast<DataFrame>();
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of Score::data is not a DataFrame (pandas.DataFrame or pyarrow.RecordBatch).");
            }
        } else {
            return DataFrame();
        }
    }
};

template <typename ValidatedScoreBase = ValidatedScore>
class PyValidatedScore : public PyScore<ValidatedScoreBase> {
public:
    using PyScore<ValidatedScoreBase>::PyScore;
    using PyScore<ValidatedScoreBase>::vlocal_score;

    double vscore(const BayesianNetworkBase& model) const override {
        {
            py::gil_scoped_acquire gil;
            py::function override = pybind11::get_override(static_cast<const ValidatedScoreBase*>(this), "vscore");
            if (override) {
                auto o = override(model.shared_from_this());
                try {
                    return std::move(o).cast<double>();
                } catch (py::cast_error& e) {
                    throw std::runtime_error("The returned object of ValidatedScore::vscore is not a double.");
                }
            }
        }

        return ValidatedScoreBase::vscore(model);
    }

    double vlocal_score(const BayesianNetworkBase& model,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const override {
        PYBIND11_OVERRIDE_PURE(double,             /* Return type */
                               ValidatedScoreBase, /* Parent class */
                               vlocal_score,       /* Name of function in C++ (must match Python name) */
                               model.shared_from_this(),
                               variable, /* Argument(s) */
                               parents);
    }

    double vlocal_score(const BayesianNetworkBase& model,
                        const std::shared_ptr<FactorType>& node_type,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const override {
        PYBIND11_OVERRIDE_PURE_NAME(double,             /* Return type */
                                    ValidatedScoreBase, /* Parent class */
                                    "vlocal_score_node_type",
                                    vlocal_score, /* Name of function in C++ (must match Python name) */
                                    model.shared_from_this(),
                                    node_type,
                                    variable, /* Argument(s) */
                                    parents);
    }
};

template <typename DynamicScoreBase = DynamicScore>
class PyDynamicScore : public DynamicScoreBase {
    using DynamicScoreBase::DynamicScoreBase;

    Score& static_score() override { PYBIND11_OVERRIDE_PURE(Score&, DynamicScoreBase, static_score, ); }

    Score& transition_score() override { PYBIND11_OVERRIDE_PURE(Score&, DynamicScoreBase, transition_score, ); }

    bool has_variables(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(bool, DynamicScoreBase, has_variables, name);
    }

    bool has_variables(const std::vector<std::string>& cols) const override {
        PYBIND11_OVERRIDE_PURE(bool, DynamicScoreBase, has_variables, cols);
    }
};

void pybindings_scores(py::module& root) {
    // register_Score<GaussianNetwork, SemiparametricBN>(scores);
    py::class_<Score, PyScore<>, std::shared_ptr<Score>> score(root, "Score", R"doc(
A :class:`Score` scores Bayesian network structures.
)doc");
    score.def(py::init<>(), R"doc(
Initializes a :class:`Score`.
)doc");
    register_Score_methods<Score>(score);

    score.def("__str__", &Score::ToString);
    {
        py::options options;
        options.disable_function_signatures();
        score
            .def("has_variables",
                 py::overload_cast<const std::string&>(&Score::has_variables, py::const_),
                 py::arg("variables"))
            .def("has_variables",
                 py::overload_cast<const std::vector<std::string>&>(&Score::has_variables, py::const_),
                 py::arg("variables"),
                 R"doc(
has_variables(self: pybnesian.Score, variables: str or List[str]) -> bool

Checks whether this :class:`Score` has the given ``variables``.

:param variables: Name or list of variables.
:returns: True if the :class:`Score` is defined over the set of ``variables``, False otherwise.
)doc")
            .def("compatible_bn",
                 py::overload_cast<const ConditionalBayesianNetworkBase&>(&Score::compatible_bn, py::const_),
                 py::arg("model"))
            .def("compatible_bn",
                 py::overload_cast<const BayesianNetworkBase&>(&Score::compatible_bn, py::const_),
                 py::arg("model"),
                 R"doc(
compatible_bn(self: pybnesian.Score, model: BayesianNetworkBase or ConditionalBayesianNetworkBase) -> bool

Checks whether the ``model`` is compatible (can be used) with this :class:`Score`.

:param model: A Bayesian network model.
:returns: True if the Bayesian network model is compatible with this :class:`Score`, False otherwise.
)doc");
    }

    py::class_<ValidatedScore, Score, PyValidatedScore<>, std::shared_ptr<ValidatedScore>> validated_score(
        root, "ValidatedScore", R"doc(
A :class:`ValidatedScore` is a score with training and validation scores. In a :class:`ValidatedScore`, the training
is driven by the training score through the functions :func:`Score.score`, :func:`Score.local_score_variable`,
:func:`Score.local_score` and :func:`Score.local_score_node_type`). The convergence of the structure is evaluated using
a validation likelihood (usually defined over different data) through the functions :func:`ValidatedScore.vscore`,
:func:`ValidatedScore.vlocal_score_variable`, :func:`ValidatedScore.vlocal_score` and
:func:`ValidatedScore.vlocal_score_node_type`.
)doc");
    validated_score.def(py::init<>());
    // register_Score_methods<ValidatedScore>(validated_score);
    register_ValidatedScore_methods<ValidatedScore>(validated_score);

    py::class_<BIC, Score, std::shared_ptr<BIC>>(root, "BIC", R"doc(
This class implements the Bayesian Information Criterion (BIC).
)doc")
        .def(py::init<const DataFrame&>(), py::arg("df"), R"doc(
Initializes a :class:`BIC` with the given DataFrame ``df``.

:param df: DataFrame to compute the BIC score.
)doc");

    py::class_<BGe, Score, std::shared_ptr<BGe>>(root, "BGe", R"doc(
This class implements the Bayesian Gaussian equivalent (BGe).
)doc")
        .def(py::init<const DataFrame&, double, std::optional<double>, std::optional<VectorXd>>(),
             py::arg("df"),
             py::arg("iss_mu") = 1,
             py::arg("iss_w") = std::nullopt,
             py::arg("nu") = std::nullopt,
             R"doc(
Initializes a :class:`BGe` with the given DataFrame ``df``.

:param df: DataFrame to compute the BGe score.
:param iss_mu: Imaginary sample size for the normal component of the normal-Wishart prior.
:param iss_w: Imaginary sample size for the Wishart component of the normal-Wishart prior.
:param nu: Mean vector of the normal-Wishart prior.
)doc");

    py::class_<BDe, Score, std::shared_ptr<BDe>>(root, "BDe", R"doc(
This class implements the Bayesian Dirichlet equivalent (BDe).
)doc")
        .def(py::init<const DataFrame&, double>(),
             py::arg("df"),
             py::arg("iss") = 1,
             R"doc(
Initializes a :class:`BDe` with the given DataFrame ``df``.

:param df: DataFrame to compute the BDe score.
:param iss: Imaginary sample size of the Dirichlet prior.
)doc");

    py::class_<CVLikelihood, Score, std::shared_ptr<CVLikelihood>>(root, "CVLikelihood", R"doc(
This class implements an estimation of the log-likelihood on unseen data using k-fold cross validation over the data.
)doc")
        .def(py::init([](const DataFrame& df, int k, std::optional<unsigned int> seed, Arguments construction_args) {
                 return CVLikelihood(df, k, random_seed_arg(seed), construction_args);
             }),
             py::arg("df"),
             py::arg("k") = 10,
             py::arg("seed") = std::nullopt,
             py::arg("construction_args") = Arguments(),
             R"doc(
Initializes a :class:`CVLikelihood` with the given DataFrame ``df``. It uses a
:class:`CrossValidation <pybnesian.CrossValidation>` with ``k`` folds and the given ``seed``.

:param df: DataFrame to compute the score.
:param k: Number of folds of the cross validation.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
:param construction_args: Additional arguments provided to construct the :class:`Factor <pybnesian.Factor>`.
)doc")
        .def_property_readonly("cv", &CVLikelihood::cv, R"doc(
The underlying :class:`CrossValidation <pybnesian.CrossValidation>` object to compute the score.
)doc");

    py::class_<HoldoutLikelihood, Score, std::shared_ptr<HoldoutLikelihood>>(root, "HoldoutLikelihood", R"doc(
This class implements an estimation of the log-likelihood on unseen data using a holdout dataset. Thus, the parameters
are estimated using training data, and the score is estimated in the holdout data.
)doc")
        .def(py::init([](const DataFrame& df,
                         double test_ratio,
                         std::optional<unsigned int> seed,
                         Arguments construction_args) {
                 return HoldoutLikelihood(df, test_ratio, random_seed_arg(seed), construction_args);
             }),
             py::arg("df"),
             py::arg("test_ratio") = 0.2,
             py::arg("seed") = std::nullopt,
             py::arg("construction_args") = Arguments(),
             R"doc(
Initializes a :class:`HoldoutLikelihood` with the given DataFrame ``df``. It uses a
:class:`HoldOut <pybnesian.HoldOut>` with the given ``test_ratio`` and ``seed``.

:param df: DataFrame to compute the score.
:param test_ratio: Proportion of instances left for the holdout data.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
:param construction_args: Additional arguments provided to construct the :class:`Factor <pybnesian.Factor>`.
)doc")
        .def_property_readonly("holdout", &HoldoutLikelihood::holdout, R"doc(
The underlying :class:`HoldOut <pybnesian.HoldOut>` object to compute the score.
)doc")
        .def("training_data", &HoldoutLikelihood::training_data, py::return_value_policy::reference_internal, R"doc(
Gets the training data of the :class:`HoldOut <pybnesian.HoldOut>` object.
)doc")
        .def("test_data", &HoldoutLikelihood::test_data, py::return_value_policy::reference_internal, R"doc(
Gets the holdout data of the :class:`HoldOut <pybnesian.HoldOut>` object.
)doc");

    py::class_<ValidatedLikelihood, ValidatedScore, std::shared_ptr<ValidatedLikelihood>>(
        root, "ValidatedLikelihood", R"doc(
This class mixes the functionality of :class:`CVLikelihood` and :class:`HoldoutLikelihood`. First, it applies a
:class:`HoldOut <pybnesian.HoldOut>` split over the data. Then:

- It estimates the training score using a :class:`CVLikelihood` over the training data.
- It estimates the validation score using the training data to estimate the parameters and calculating the
  log-likelihood on the holdout data.
)doc")
        .def(py::init([](const DataFrame& df,
                         double test_ratio,
                         int k,
                         std::optional<unsigned int> seed,
                         Arguments construction_args) {
                 return ValidatedLikelihood(df, test_ratio, k, random_seed_arg(seed), construction_args);
             }),
             py::arg("df"),
             py::arg("test_ratio") = 0.2,
             py::arg("k") = 10,
             py::arg("seed") = std::nullopt,
             py::arg("construction_args") = Arguments(),
             R"doc(
Initializes a :class:`ValidatedLikelihood` with the given DataFrame ``df``. The
:class:`HoldOut <pybnesian.HoldOut>` is initialized with ``test_ratio`` and ``seed``. The ``CVLikelihood`` is
initialized with ``k`` and ``seed`` over the training data of the holdout :class:`HoldOut <pybnesian.HoldOut>`.

:param df: DataFrame to compute the score.
:param test_ratio: Proportion of instances left for the holdout data.
:param k: Number of folds of the cross validation.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
:param construction_args: Additional arguments provided to construct the :class:`Factor <pybnesian.Factor>`.
)doc")
        .def_property_readonly(
            "holdout_lik", &ValidatedLikelihood::holdout, py::return_value_policy::reference_internal, R"doc(
The underlying :class:`HoldoutLikelihood` to compute the validation score.
)doc")
        .def_property_readonly("cv_lik", &ValidatedLikelihood::cv, py::return_value_policy::reference_internal, R"doc(
The underlying :class:`CVLikelihood` to compute the training score.
)doc")
        .def("training_data", &ValidatedLikelihood::training_data, py::return_value_policy::reference_internal, R"doc(
The underlying training data of the :class:`HoldOut <pybnesian.HoldOut>`.
)doc")
        .def("validation_data",
             &ValidatedLikelihood::validation_data,
             py::return_value_policy::reference_internal,
             R"doc(
The underlying holdout data of the :class:`HoldOut <pybnesian.HoldOut>`.
)doc");

    py::class_<DynamicScore, PyDynamicScore<>, std::shared_ptr<DynamicScore>> dynamic_score(root, "DynamicScore", R"doc(
A :class:`DynamicScore` adapts the static :class:`Score` to learn dynamic Bayesian networks. It generates a static and a
transition score to learn the static and transition components of the dynamic Bayesian network.

The dynamic scores are usually implemented using a :class:`DynamicDataFrame <pybnesian.DynamicDataFrame>` with
the methods :func:`DynamicDataFrame.static_df <pybnesian.DynamicDataFrame.static_df>` and
:func:`DynamicDataFrame.transition_df <pybnesian.DynamicDataFrame.transition_df>`.
)doc");

    dynamic_score
        .def(py::init<>(), R"doc(
Initializes a :class:`DynamicScore`.
)doc")
        .def("static_score", &DynamicScore::static_score, py::return_value_policy::reference_internal, R"doc(
It returns the static score component of the :class:`DynamicScore`.

:returns: The static score component.
)doc")
        .def("transition_score", &DynamicScore::transition_score, py::return_value_policy::reference_internal, R"doc(
It returns the transition score component of the :class:`DynamicScore`.

:returns: The transition score component.
)doc");

    {
        py::options options;
        options.disable_function_signatures();

        dynamic_score
            .def("has_variables",
                 py::overload_cast<const std::string&>(&DynamicScore::has_variables, py::const_),
                 py::arg("variables"))
            .def("has_variables",
                 py::overload_cast<const std::vector<std::string>&>(&DynamicScore::has_variables, py::const_),
                 py::arg("variables"),
                 R"doc(
has_variables(self: pybnesian.DynamicScore, variables: str or List[str]) -> bool

Checks whether this :class:`DynamicScore` has the given ``variables``.

:param variables: Name or list of variables.
:returns: True if the :class:`DynamicScore` is defined over the set of ``variables``, False otherwise.
)doc");
    }

    py::class_<DynamicBIC, DynamicScore, std::shared_ptr<DynamicBIC>>(
        root, "DynamicBIC", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`BIC` score.
)doc")
        .def(py::init<DynamicDataFrame>(), py::keep_alive<1, 2>(), py::arg("ddf"), R"doc(
Initializes a :class:`DynamicBIC` with the given :class:`DynamicDataFrame` ``ddf``.

:param ddf: :class:`DynamicDataFrame` to compute the :class:`DynamicBIC` score.
)doc");

    py::class_<DynamicBGe, DynamicScore, std::shared_ptr<DynamicBGe>>(
        root, "DynamicBGe", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`BGe` score.
)doc")
        .def(py::init<DynamicDataFrame, double, std::optional<double>, std::optional<VectorXd>>(),
             py::keep_alive<1, 2>(),
             py::arg("ddf"),
             py::arg("iss_mu") = 1,
             py::arg("iss_w") = std::nullopt,
             py::arg("nu") = std::nullopt,
             R"doc(
Initializes a :class:`DynamicBGe` with the given :class:`DynamicDataFrame` ``ddf``.

:param ddf: :class:`DynamicDataFrame` to compute the :class:`DynamicBGe` score.
:param iss_mu: Imaginary sample size for the normal component of the normal-Wishart prior.
:param iss_w: Imaginary sample size for the Wishart component of the normal-Wishart prior.
:param nu: Mean vector of the normal-Wishart prior.
)doc");

    py::class_<DynamicBDe, DynamicScore, std::shared_ptr<DynamicBDe>>(
        root, "DynamicBDe", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`BDe` score.
)doc")
        .def(py::init<DynamicDataFrame, double>(), py::keep_alive<1, 2>(), py::arg("ddf"), py::arg("iss") = 1, R"doc(
Initializes a :class:`DynamicBDe` with the given :class:`DynamicDataFrame` ``ddf``.

:param ddf: :class:`DynamicDataFrame` to compute the :class:`DynamicBDe` score.
:param iss: Imaginary sample size of the Dirichlet prior.
)doc");

    py::class_<DynamicCVLikelihood, DynamicScore, std::shared_ptr<DynamicCVLikelihood>>(
        root, "DynamicCVLikelihood", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`CVLikelihood` score.
)doc")
        .def(py::init([](DynamicDataFrame df, int k, std::optional<unsigned int> seed) {
                 return DynamicCVLikelihood(df, k, static_cast<unsigned int>(random_seed_arg(seed)));
             }),
             py::arg("df"),
             py::arg("k") = 10,
             py::arg("seed") = std::nullopt,
             R"doc(
Initializes a :class:`DynamicCVLikelihood` with the given :class:`DynamicDataFrame` ``df``. The ``k`` and ``seed``
parameters are passed to the static and transition components of :class:`CVLikelihood`.

:param df: :class:`DynamicDataFrame` to compute the score.
:param k: Number of folds of the cross validation.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
)doc");

    py::class_<DynamicHoldoutLikelihood, DynamicScore, std::shared_ptr<DynamicHoldoutLikelihood>>(
        root, "DynamicHoldoutLikelihood", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`HoldoutLikelihood` score.
)doc")
        .def(py::init([](DynamicDataFrame df, double test_ratio, std::optional<unsigned int> seed) {
                 return DynamicHoldoutLikelihood(df, test_ratio, static_cast<unsigned int>(random_seed_arg(seed)));
             }),
             py::arg("df"),
             py::arg("test_ratio") = 0.2,
             py::arg("seed") = std::nullopt,
             R"doc(
Initializes a :class:`DynamicHoldoutLikelihood` with the given :class:`DynamicDataFrame` ``df``. The ``test_ratio`` and
``seed`` parameters are passed to the static and transition components of :class:`HoldoutLikelihood`.

:param df: :class:`DynamicDataFrame` to compute the score.
:param test_ratio: Proportion of instances left for the holdout data.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
)doc");

    py::class_<DynamicValidatedLikelihood, DynamicScore, std::shared_ptr<DynamicValidatedLikelihood>>(
        root, "DynamicValidatedLikelihood", py::multiple_inheritance(), R"doc(
The dynamic adaptation of the :class:`ValidatedLikelihood` score.
)doc")
        .def(py::init([](DynamicDataFrame df, double test_ratio, int k, std::optional<unsigned int> seed) {
                 return DynamicValidatedLikelihood(df, test_ratio, k, static_cast<unsigned int>(random_seed_arg(seed)));
             }),
             py::arg("df"),
             py::arg("test_ratio") = 0.2,
             py::arg("k") = 10,
             py::arg("seed") = std::nullopt,
             R"doc(
Initializes a :class:`DynamicValidatedLikelihood` with the given :class:`DynamicDataFrame` ``df``. The ``test_ratio``,
``k`` and ``seed`` parameters are passed to the static and transition components of :class:`ValidatedLikelihood`.

:param df: :class:`DynamicDataFrame` to compute the score.
:param test_ratio: Proportion of instances left for the holdout data.
:param k: Number of folds of the cross validation.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
)doc");
}
