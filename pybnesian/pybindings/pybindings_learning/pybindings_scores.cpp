#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <learning/scores/scores.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/bge.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>
#include <learning/scores/validated_likelihood.hpp>
#include <util/util_types.hpp>

namespace py = pybind11;

using learning::scores::Score, learning::scores::ValidatedScore, learning::scores::BIC, learning::scores::BGe,
    learning::scores::CVLikelihood, learning::scores::HoldoutLikelihood, learning::scores::ValidatedLikelihood;

using learning::scores::DynamicScore, learning::scores::DynamicBIC, learning::scores::DynamicCVLikelihood,
    learning::scores::DynamicHoldoutLikelihood, learning::scores::DynamicValidatedLikelihood;

using util::random_seed_arg;

template <typename CppClass, typename PyClass>
void register_Score_methods(PyClass& pyclass) {
    pyclass
        .def("score",
             [](const CppClass& self, const ConditionalBayesianNetworkBase& m) {
                 if (self.compatible_bn(m))
                     return self.score(m);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("score",
             [](const CppClass& self, const BayesianNetworkBase& m) {
                 if (self.compatible_bn(m))
                     return self.score(m);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("score_unsafe", [](const CppClass& self, const BayesianNetworkBase& m) { return self.score(m); })
        .def("local_score",
             [](const CppClass& self, const ConditionalBayesianNetworkBase& m, const std::string& variable) {
                 if (self.compatible_bn(m))
                     return self.local_score(m, variable);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("local_score",
             [](const CppClass& self, const BayesianNetworkBase& m, const std::string& variable) {
                 if (self.compatible_bn(m))
                     return self.local_score(m, variable);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("local_score_unsafe",
             [](const CppClass& self, const BayesianNetworkBase& m, const std::string& variable) {
                 return self.local_score(m, variable);
             })
        .def("local_score",
             [](const CppClass& self,
                const ConditionalBayesianNetworkBase& m,
                const std::string& variable,
                const std::vector<std::string>& evidence) {
                 if (self.compatible_bn(m))
                     return self.local_score(m, variable, evidence);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("local_score",
             [](const CppClass& self,
                const BayesianNetworkBase& m,
                const std::string& variable,
                const std::vector<std::string>& evidence) {
                 if (self.compatible_bn(m))
                     return self.local_score(m, variable, evidence);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("local_score_unsafe",
             [](const CppClass& self,
                const BayesianNetworkBase& m,
                const std::string& variable,
                const std::vector<std::string>& evidence) { return self.local_score(m, variable, evidence); })
        .def("local_score",
             [](CppClass& self,
                const BayesianNetworkBase& m,
                const std::shared_ptr<FactorType>& variable_type,
                const std::string& variable,
                const std::vector<std::string>& evidence) {
                 if (self.has_variables(variable) && self.has_variables(evidence))
                     return self.local_score(m, *variable_type, variable, evidence);
                 else
                     throw py::value_error("Score is incompatible with variable/evidence.");
             })
        .def("local_score_unsafe",
             [](CppClass& self,
                const BayesianNetworkBase& m,
                const std::shared_ptr<FactorType>& variable_type,
                const std::string& variable,
                const std::vector<std::string>& evidence) {
                 return self.local_score(m, *variable_type, variable, evidence);
             });
}

template <typename CppClass, typename PyClass>
void register_ValidatedScore_methods(PyClass& pyclass) {
    pyclass
        .def("vscore",
             [](const CppClass& self, const ConditionalBayesianNetworkBase& m) {
                 if (self.compatible_bn(m))
                     return self.vscore(m);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("vscore",
             [](const CppClass& self, const BayesianNetworkBase& m) {
                 if (self.compatible_bn(m))
                     return self.vscore(m);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("vscore_unsafe", [](const CppClass& self, const BayesianNetworkBase& m) { return self.vscore(m); })
        .def("vlocal_score",
             [](const CppClass& self, const ConditionalBayesianNetworkBase& m, const std::string& variable) {
                 if (self.compatible_bn(m))
                     return self.vlocal_score(m, variable);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("vlocal_score",
             [](const CppClass& self, const BayesianNetworkBase& m, const std::string& variable) {
                 if (self.compatible_bn(m))
                     return self.vlocal_score(m, variable);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("vlocal_score_unsafe",
             [](const CppClass& self, const BayesianNetworkBase& m, const std::string& variable) {
                 return self.vlocal_score(m, variable);
             })
        .def("vlocal_score",
             [](const CppClass& self,
                const ConditionalBayesianNetworkBase& m,
                const std::string& variable,
                const std::vector<std::string>& evidence) {
                 if (self.compatible_bn(m))
                     return self.vlocal_score(m, variable, evidence);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("vlocal_score",
             [](const CppClass& self,
                const BayesianNetworkBase& m,
                const std::string& variable,
                const std::vector<std::string>& evidence) {
                 if (self.compatible_bn(m))
                     return self.vlocal_score(m, variable, evidence);
                 else
                     throw py::value_error("Bayesian network is incompatible with the score.");
             })
        .def("vlocal_score_unsafe",
             [](const CppClass& self,
                const BayesianNetworkBase& m,
                const std::string& variable,
                const std::vector<std::string>& evidence) { return self.vlocal_score(m, variable, evidence); })
        .def("vlocal_score",
             [](CppClass& self,
                const BayesianNetworkBase& m,
                const std::shared_ptr<FactorType>& variable_type,
                const std::string& variable,
                const std::vector<std::string>& evidence) {
                 if (self.has_variables(variable) && self.has_variables(evidence))
                     return self.vlocal_score(m, *variable_type, variable, evidence);
                 else
                     throw py::value_error("Score is incompatible with variable/evidence.");
             })
        .def("vlocal_score_unsafe",
             [](CppClass& self,
                const BayesianNetworkBase& m,
                const std::shared_ptr<FactorType>& variable_type,
                const std::string& variable,
                const std::vector<std::string>& evidence) {
                 return self.vlocal_score(m, *variable_type, variable, evidence);
             });
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
                auto o = override(&model);
                return std::move(o).cast<double>();
            }
        }

        return ScoreBase::score(model);
    }

    double local_score(const BayesianNetworkBase& model, const std::string& variable) const override {
        {
            py::gil_scoped_acquire gil;
            py::function override = pybind11::get_override(static_cast<const ScoreBase*>(this), "local_score_variable");
            if (override) {
                auto o = override(&model, variable);
                return std::move(o).cast<double>();
            }
        }

        return ScoreBase::local_score(model, variable);
    }

    double local_score(const BayesianNetworkBase& model,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override {
        PYBIND11_OVERRIDE_PURE(double,      /* Return type */
                               ScoreBase,   /* Parent class */
                               local_score, /* Name of function in C++ (must match Python name) */
                               &model,
                               variable, /* Argument(s) */
                               parents);
    }

    double local_score(const BayesianNetworkBase& model,
                       const FactorType& node_type,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override {
        PYBIND11_OVERRIDE_PURE_NAME(double,    /* Return type */
                                    ScoreBase, /* Parent class */
                                    "local_score_node_type",
                                    local_score, /* Name of function in C++ (must match Python name) */
                                    &model,
                                    &node_type,
                                    variable, /* Argument(s) */
                                    parents);
    }

    std::string ToString() const override {
        PYBIND11_OVERRIDE_PURE(std::string, /* Return type */
                               ScoreBase,   /* Parent class */
                               ToString,    /* Name of function in C++ (must match Python name) */
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
                auto o = override(&model);
                return std::move(o).cast<double>();
            }
        }

        return ValidatedScoreBase::vscore(model);
    }

    double vlocal_score(const BayesianNetworkBase& model, const std::string& variable) const override {
        {
            py::gil_scoped_acquire gil;
            py::function override =
                pybind11::get_override(static_cast<const ValidatedScoreBase*>(this), "vlocal_score_variable");
            if (override) {
                auto o = override(&model, variable);
                return std::move(o).cast<double>();
            }
        }
        return ValidatedScoreBase::vlocal_score(model, variable);
    }

    double vlocal_score(const BayesianNetworkBase& model,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const override {
        PYBIND11_OVERRIDE_PURE(double,             /* Return type */
                               ValidatedScoreBase, /* Parent class */
                               vlocal_score,       /* Name of function in C++ (must match Python name) */
                               &model,
                               variable, /* Argument(s) */
                               parents);
    }

    double vlocal_score(const BayesianNetworkBase& model,
                        const FactorType& node_type,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const override {
        PYBIND11_OVERRIDE_PURE_NAME(double,             /* Return type */
                                    ValidatedScoreBase, /* Parent class */
                                    "vlocal_score_node_type",
                                    vlocal_score, /* Name of function in C++ (must match Python name) */
                                    &model,
                                    &node_type,
                                    variable, /* Argument(s) */
                                    parents);
    }
};

void pybindings_scores(py::module& root) {
    auto scores = root.def_submodule("scores", "Learning scores submodule.");

    // register_Score<GaussianNetwork, SemiparametricBN>(scores);
    py::class_<Score, PyScore<>, std::shared_ptr<Score>> score(scores, "Score");
    score.def(py::init<>());
    register_Score_methods<Score>(score);

    score.def("ToString", &Score::ToString)
        .def("has_variables", py::overload_cast<const std::string&>(&Score::has_variables, py::const_))
        .def("has_variables", py::overload_cast<const std::vector<std::string>&>(&Score::has_variables, py::const_))
        .def("compatible_bn",
             py::overload_cast<const ConditionalBayesianNetworkBase&>(&Score::compatible_bn, py::const_))
        .def("compatible_bn", py::overload_cast<const BayesianNetworkBase&>(&Score::compatible_bn, py::const_));

    py::class_<ValidatedScore, Score, PyValidatedScore<>, std::shared_ptr<ValidatedScore>> validated_score(
        scores, "ValidatedScore");
    validated_score.def(py::init<>());
    register_Score_methods<ValidatedScore>(validated_score);
    register_ValidatedScore_methods<ValidatedScore>(validated_score);

    py::class_<BIC, Score, std::shared_ptr<BIC>>(scores, "BIC").def(py::init<const DataFrame&>());

    py::class_<BGe, Score, std::shared_ptr<BGe>>(scores, "BGe")
        .def(py::init<const DataFrame&, double, std::optional<double>, std::optional<VectorXd>>(),
             py::arg("df"),
             py::arg("iss_mu") = 1,
             py::arg("iss_w") = std::nullopt,
             py::arg("nu") = std::nullopt);

    py::class_<CVLikelihood, Score, std::shared_ptr<CVLikelihood>>(scores, "CVLikelihood")
        .def(py::init([](const DataFrame& df, int k, std::optional<unsigned int> seed) {
                 return CVLikelihood(df, k, random_seed_arg(seed));
             }),
             py::arg("df"),
             py::arg("k") = 10,
             py::arg("seed") = std::nullopt)
        .def_property_readonly("cv", &CVLikelihood::cv);

    py::class_<HoldoutLikelihood, Score, std::shared_ptr<HoldoutLikelihood>>(scores, "HoldoutLikelihood")
        .def(py::init([](const DataFrame& df, double test_ratio, std::optional<unsigned int> seed) {
                 return HoldoutLikelihood(df, test_ratio, random_seed_arg(seed));
             }),
             py::arg("df"),
             py::arg("test_ratio") = 0.2,
             py::arg("seed") = std::nullopt)
        .def_property_readonly("holdout", &HoldoutLikelihood::holdout)
        .def("training_data", &HoldoutLikelihood::training_data, py::return_value_policy::reference_internal)
        .def("test_data", &HoldoutLikelihood::test_data, py::return_value_policy::reference_internal);

    py::class_<ValidatedLikelihood, ValidatedScore, std::shared_ptr<ValidatedLikelihood>>(scores, "ValidatedLikelihood")
        .def(py::init([](const DataFrame& df, double test_ratio, int k, std::optional<unsigned int> seed) {
                 return ValidatedLikelihood(df, test_ratio, k, random_seed_arg(seed));
             }),
             py::arg("df"),
             py::arg("test_ratio") = 0.2,
             py::arg("k") = 10,
             py::arg("seed") = std::nullopt)
        .def_property_readonly(
            "holdout_lik", &ValidatedLikelihood::holdout, py::return_value_policy::reference_internal)
        .def_property_readonly("cv_lik", &ValidatedLikelihood::cv, py::return_value_policy::reference_internal)
        .def("training_data", &ValidatedLikelihood::training_data, py::return_value_policy::reference_internal)
        .def("validation_data", &ValidatedLikelihood::validation_data, py::return_value_policy::reference_internal);

    py::class_<DynamicScore, std::shared_ptr<DynamicScore>>(scores, "DynamicScore")
        .def("static_score", &DynamicScore::static_score, py::return_value_policy::reference_internal)
        .def("transition_score", &DynamicScore::transition_score, py::return_value_policy::reference_internal)
        .def("has_variables", py::overload_cast<const std::string&>(&DynamicScore::has_variables, py::const_))
        .def("has_variables",
             py::overload_cast<const std::vector<std::string>&>(&DynamicScore::has_variables, py::const_))
        .def("compatible_bn",
             py::overload_cast<const ConditionalBayesianNetworkBase&>(&DynamicScore::compatible_bn, py::const_))
        .def("compatible_bn", py::overload_cast<const BayesianNetworkBase&>(&DynamicScore::compatible_bn, py::const_));

    py::class_<DynamicBIC, DynamicScore, std::shared_ptr<DynamicBIC>>(scores, "DynamicBIC", py::multiple_inheritance())
        .def(py::init<DynamicDataFrame>(), py::keep_alive<1, 2>());

    py::class_<DynamicCVLikelihood, DynamicScore, std::shared_ptr<DynamicCVLikelihood>>(
        scores, "DynamicCVLikelihood", py::multiple_inheritance())
        .def(py::init([](DynamicDataFrame df, int k, std::optional<unsigned int> seed) {
                 return DynamicCVLikelihood(df, k, static_cast<unsigned int>(random_seed_arg(seed)));
             }),
             py::arg("df"),
             py::arg("k") = 10,
             py::arg("seed") = std::nullopt);

    py::class_<DynamicHoldoutLikelihood, DynamicScore, std::shared_ptr<DynamicHoldoutLikelihood>>(
        scores, "DynamicHoldoutLikelihood", py::multiple_inheritance())
        .def(py::init([](DynamicDataFrame df, double test_ratio, std::optional<unsigned int> seed) {
                 return DynamicHoldoutLikelihood(df, test_ratio, static_cast<unsigned int>(random_seed_arg(seed)));
             }),
             py::arg("df"),
             py::arg("test_ratio") = 0.2,
             py::arg("seed") = std::nullopt);

    py::class_<DynamicValidatedLikelihood, DynamicScore, std::shared_ptr<DynamicValidatedLikelihood>>(
        scores, "DynamicValidatedLikelihood", py::multiple_inheritance())
        .def(py::init([](DynamicDataFrame df, double test_ratio, int k, std::optional<unsigned int> seed) {
                 return DynamicValidatedLikelihood(df, test_ratio, k, static_cast<unsigned int>(random_seed_arg(seed)));
             }),
             py::arg("df"),
             py::arg("test_ratio") = 0.2,
             py::arg("k") = 10,
             py::arg("seed") = std::nullopt);
}
