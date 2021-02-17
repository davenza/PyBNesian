#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <learning/scores/scores.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/bge.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>
#include <learning/scores/validated_likelihood.hpp>

namespace py = pybind11;

using learning::scores::Score, learning::scores::ValidatedScore,
      learning::scores::BIC, learning::scores::BGe,
      learning::scores::CVLikelihood, learning::scores::HoldoutLikelihood,
      learning::scores::ValidatedLikelihood;

using learning::scores::DynamicScore, learning::scores::DynamicBIC,
      learning::scores::DynamicCVLikelihood, learning::scores::DynamicHoldoutLikelihood,
      learning::scores::DynamicValidatedLikelihood;


template<typename CppClass, typename PyClass>
void register_Score_methods(PyClass& pyclass) {
    pyclass
        .def("score", [](const CppClass& self, const ConditionalBayesianNetworkBase& m) {
            if (self.compatible_bn(m))
                return self.score(m);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("score", [](const CppClass& self, const BayesianNetworkBase& m) {
            if (self.compatible_bn(m))
                return self.score(m);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("score_unsafe", [](const CppClass& self, const BayesianNetworkBase& m) {
            return self.score(m);
        })
        .def("local_score", [](const CppClass& self, const ConditionalBayesianNetworkBase& m, const std::string& variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const CppClass& self, const BayesianNetworkBase& m, const std::string& variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const CppClass& self, const BayesianNetworkBase& m, const std::string& variable) {
            return self.local_score(m, variable);
        })
        .def("local_score", [](const CppClass& self, const ConditionalBayesianNetworkBase& m, int variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const CppClass& self, const BayesianNetworkBase& m, int variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const CppClass& self, const BayesianNetworkBase& m, int variable) {
            return self.local_score(m, variable);
        })
        .def("local_score", [](const CppClass& self,
                               const ConditionalBayesianNetworkBase& m,
                               const std::string& variable, 
                               const std::vector<std::string>& evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const CppClass& self,
                               const BayesianNetworkBase& m,
                               const std::string& variable, 
                               const std::vector<std::string>& evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const CppClass& self,
                                      const BayesianNetworkBase& m,
                                      const std::string& variable, 
                                      const std::vector<std::string>& evidence) {
            return self.local_score(m, variable, evidence);
        })
        .def("local_score", [](const CppClass& self,
                               const ConditionalBayesianNetworkBase& m,
                               int variable, 
                               const std::vector<int>& evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const CppClass& self,
                               const BayesianNetworkBase& m,
                               int variable, 
                               const std::vector<int>& evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const CppClass& self,
                                      const BayesianNetworkBase& m,
                                      int variable, 
                                      const std::vector<int>& evidence) {
            return self.local_score(m, variable, evidence);
        });
}

template<typename CppClass, typename PyClass>
void register_ScoreSPBN_methods(PyClass& pyclass) {
    pyclass
        .def("local_score", [](CppClass& self,
                               FactorType variable_type,
                               const std::string& variable,
                               const std::vector<std::string>& evidence) {
            if (self.has_variables(variable) && self.has_variables(evidence))
                return self.local_score(variable_type, variable, evidence);
            else
                throw py::value_error("Score is incompatible with variable/evidence.");
        })
        .def("local_score_unsafe", [](CppClass& self,
                                      FactorType variable_type,
                                      const std::string& variable,
                                      const std::vector<std::string>& evidence) {
            return self.local_score(variable_type, variable, evidence);
        });
}

template<typename CppClass, typename PyClass>
void register_ValidatedScore_methods(PyClass& pyclass) {
    pyclass
        .def("vscore", [](const CppClass& self, const ConditionalBayesianNetworkBase& m) {
            if (self.compatible_bn(m))
                return self.vscore(m);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("vscore", [](const CppClass& self, const BayesianNetworkBase& m) {
            if (self.compatible_bn(m))
                return self.vscore(m);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("vscore_unsafe", [](const CppClass& self, const BayesianNetworkBase& m) {
            return self.vscore(m);
        })
        .def("vlocal_score", [](const CppClass& self, const ConditionalBayesianNetworkBase& m, const std::string& variable) {
            if (self.compatible_bn(m))
                return self.vlocal_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("vlocal_score", [](const CppClass& self, const BayesianNetworkBase& m, const std::string& variable) {
            if (self.compatible_bn(m))
                return self.vlocal_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("vlocal_score_unsafe", [](const CppClass& self, const BayesianNetworkBase& m, const std::string& variable) {
            return self.vlocal_score(m, variable);
        })
        .def("vlocal_score", [](const CppClass& self, const ConditionalBayesianNetworkBase& m, int variable) {
            if (self.compatible_bn(m))
                return self.vlocal_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("vlocal_score", [](const CppClass& self, const BayesianNetworkBase& m, int variable) {
            if (self.compatible_bn(m))
                return self.vlocal_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("vlocal_score_unsafe", [](const CppClass& self, const BayesianNetworkBase& m, int variable) {
            return self.vlocal_score(m, variable);
        })
        .def("vlocal_score", [](const CppClass& self,
                               const ConditionalBayesianNetworkBase& m,
                               const std::string& variable, 
                               const std::vector<std::string>& evidence) {
            if (self.compatible_bn(m))
                return self.vlocal_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("vlocal_score", [](const CppClass& self,
                               const BayesianNetworkBase& m,
                               const std::string& variable, 
                               const std::vector<std::string>& evidence) {
            if (self.compatible_bn(m))
                return self.vlocal_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("vlocal_score_unsafe", [](const CppClass& self,
                                      const BayesianNetworkBase& m,
                                      const std::string& variable, 
                                      const std::vector<std::string>& evidence) {
            return self.vlocal_score(m, variable, evidence);
        })
        .def("vlocal_score", [](const CppClass& self,
                               const ConditionalBayesianNetworkBase& m,
                               int variable, 
                               const std::vector<int>& evidence) {
            if (self.compatible_bn(m))
                return self.vlocal_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("vlocal_score", [](const CppClass& self,
                               const BayesianNetworkBase& m,
                               int variable, 
                               const std::vector<int>& evidence) {
            if (self.compatible_bn(m))
                return self.vlocal_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("vlocal_score_unsafe", [](const CppClass& self,
                                      const BayesianNetworkBase& m,
                                      int variable, 
                                      const std::vector<int>& evidence) {
            return self.vlocal_score(m, variable, evidence);
        });
}

template<typename CppClass, typename PyClass>
void register_ValidatedScoreSPBN_methods(PyClass& pyclass) {
    pyclass
        .def("vlocal_score", [](CppClass& self,
                               FactorType variable_type,
                               const std::string& variable,
                               const std::vector<std::string>& evidence) {
            if (self.has_variables(variable) && self.has_variables(evidence))
                return self.vlocal_score(variable_type, variable, evidence);
            else
                throw py::value_error("Score is incompatible with variable/evidence.");
        })
        .def("vlocal_score_unsafe", [](CppClass& self,
                                      FactorType variable_type,
                                      const std::string& variable,
                                      const std::vector<std::string>& evidence) {
            return self.vlocal_score(variable_type, variable, evidence);
        });
}

void pybindings_scores(py::module& root) {
    auto scores = root.def_submodule("scores", "Learning scores submodule.");

    // register_Score<GaussianNetwork, SemiparametricBN>(scores);
    py::class_<Score, std::shared_ptr<Score>> score(scores, "Score");
    register_Score_methods<Score>(score);

    score
        .def("ToString", &Score::ToString)
        .def("is_decomposable", &Score::is_decomposable)
        .def("type", &Score::type)
        .def("has_variables", py::overload_cast<const std::string&>(&Score::has_variables, py::const_))
        .def("has_variables", py::overload_cast<const std::vector<std::string>&>(&Score::has_variables, py::const_))
        .def("compatible_bn", py::overload_cast<const ConditionalBayesianNetworkBase&>(&Score::compatible_bn, py::const_))
        .def("compatible_bn", py::overload_cast<const BayesianNetworkBase&>(&Score::compatible_bn, py::const_));

    py::class_<ScoreSPBN, Score, std::shared_ptr<ScoreSPBN>> scorespbn(scores, "ScoreSPBN");
    register_Score_methods<ScoreSPBN>(scorespbn);
    register_ScoreSPBN_methods<ScoreSPBN>(scorespbn);

    py::class_<ValidatedScore, Score, std::shared_ptr<ValidatedScore>> validated_score(scores, "ValidatedScore");
    register_Score_methods<ValidatedScore>(validated_score);
    register_ValidatedScore_methods<ValidatedScore>(validated_score);


    py::class_<ValidatedScoreSPBN, ValidatedScore, ScoreSPBN, 
                std::shared_ptr<ValidatedScoreSPBN>> validated_scorespbn(scores, "ValidatedScoreSPBN");
    register_Score_methods<ValidatedScoreSPBN>(validated_scorespbn);
    register_ScoreSPBN_methods<ValidatedScoreSPBN>(validated_scorespbn);
    register_ValidatedScore_methods<ValidatedScoreSPBN>(validated_scorespbn);
    register_ValidatedScoreSPBN_methods<ValidatedScoreSPBN>(validated_scorespbn);

    py::class_<BIC, Score, std::shared_ptr<BIC>>(scores, "BIC")
        .def(py::init<const DataFrame&>());

    py::class_<BGe, Score, std::shared_ptr<BGe>>(scores, "BGe")
        .def(py::init<const DataFrame&,
                      double,
                      std::optional<double>,
                      std::optional<VectorXd>>(),
                      py::arg("df"),
                      py::arg("iss_mu") = 1,
                      py::arg("iss_w") = std::nullopt,
                      py::arg("nu") = std::nullopt);

    py::class_<CVLikelihood, ScoreSPBN, std::shared_ptr<CVLikelihood>>(scores, "CVLikelihood")
        .def(py::init<const DataFrame&, int>(),
                py::arg("df"),
                py::arg("k") = 10)
        .def(py::init<const DataFrame&, int, unsigned int>(),
                py::arg("df"),
                py::arg("k") = 10,
                py::arg("seed"))
        .def_property_readonly("cv", &CVLikelihood::cv);

    py::class_<HoldoutLikelihood, ScoreSPBN, std::shared_ptr<HoldoutLikelihood>>(scores, "HoldoutLikelihood")
        .def(py::init<const DataFrame&, double>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2)
        .def(py::init<const DataFrame&, double, unsigned int>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2,
                py::arg("seed"))
        .def_property_readonly("holdout", &HoldoutLikelihood::holdout)
        .def("training_data", &HoldoutLikelihood::training_data, py::return_value_policy::reference_internal)
        .def("test_data", &HoldoutLikelihood::test_data, py::return_value_policy::reference_internal);

    py::class_<ValidatedLikelihood, ValidatedScoreSPBN, std::shared_ptr<ValidatedLikelihood>>(scores, "ValidatedLikelihood")
        .def(py::init<const DataFrame&, double, int>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2,
                py::arg("k") = 10)
        .def(py::init<const DataFrame&, double, int, unsigned int>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2,
                py::arg("k") = 10,
                py::arg("seed"))
        .def_property_readonly("holdout_lik", &ValidatedLikelihood::holdout, py::return_value_policy::reference_internal)
        .def_property_readonly("cv_lik", &ValidatedLikelihood::cv, py::return_value_policy::reference_internal)
        .def("training_data", &ValidatedLikelihood::training_data, py::return_value_policy::reference_internal)
        .def("validation_data", &ValidatedLikelihood::validation_data, py::return_value_policy::reference_internal);

    py::class_<DynamicScore, std::shared_ptr<DynamicScore>>(scores, "DynamicScore")
        .def("static_score", &DynamicScore::static_score)
        .def("transition_score", &DynamicScore::transition_score)
        .def("has_variables", py::overload_cast<const std::string&>(&DynamicScore::has_variables, py::const_))
        .def("has_variables", py::overload_cast<const std::vector<std::string>&>(&DynamicScore::has_variables, py::const_))
        .def("compatible_bn", py::overload_cast<const ConditionalBayesianNetworkBase&>(&DynamicScore::compatible_bn, py::const_))
        .def("compatible_bn", py::overload_cast<const BayesianNetworkBase&>(&DynamicScore::compatible_bn, py::const_));

    py::class_<DynamicBIC, DynamicScore, std::shared_ptr<DynamicBIC>>
                        (scores, "DynamicBIC", py::multiple_inheritance())
        .def(py::init<DynamicDataFrame>(), py::keep_alive<1, 2>());

    py::class_<DynamicCVLikelihood, DynamicScore, std::shared_ptr<DynamicCVLikelihood>>
                        (scores, "DynamicCVLikelihood", py::multiple_inheritance())
        .def(py::init<DynamicDataFrame, int>(),
                py::arg("df"),
                py::arg("k") = 10)
        .def(py::init<DynamicDataFrame, int, unsigned int>(),
                py::arg("df"),
                py::arg("k") = 10,
                py::arg("seed"));

    py::class_<DynamicHoldoutLikelihood, DynamicScore, std::shared_ptr<DynamicHoldoutLikelihood>>
                        (scores, "DynamicHoldoutLikelihood", py::multiple_inheritance())
        .def(py::init<DynamicDataFrame, double>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2)
        .def(py::init<DynamicDataFrame, double, unsigned int>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2,
                py::arg("seed"));

    py::class_<DynamicValidatedLikelihood, DynamicScore, std::shared_ptr<DynamicValidatedLikelihood>>
                        (scores, "DynamicValidatedLikelihood", py::multiple_inheritance())
        .def(py::init<DynamicDataFrame, double, int>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2,
                py::arg("k") = 10)
        .def(py::init<DynamicDataFrame, double, int, unsigned int>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2,
                py::arg("k") = 10,
                py::arg("seed"));
}
