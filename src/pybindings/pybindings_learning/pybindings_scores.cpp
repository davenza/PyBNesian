#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <learning/scores/scores.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>

namespace py = pybind11;

using learning::scores::Score, learning::scores::BIC, learning::scores::CVLikelihood, 
      learning::scores::HoldoutLikelihood;

using learning::scores::DynamicScore, learning::scores::DynamicBIC,
      learning::scores::DynamicCVLikelihood, learning::scores::DynamicHoldoutLikelihood;

// template<typename Model, typename... Models>
// py::class_<Score, std::shared_ptr<Score>> register_Score(py::module& m) {
//     auto score = [&m](){
//         if constexpr (sizeof...(Models) == 0) {
//             py::class_<Score, std::shared_ptr<Score>> score(m, "Score");
//             score.def("is_decomposable", &Score::is_decomposable)
//             .def("type", &Score::type)
//             .def("local_score", [](Score& self, 
//                                    FactorType variable_type, 
//                                    const std::string& variable, 
//                                    const std::vector<std::string> evidence) {
//                 return self.local_score(variable_type, variable, evidence.begin(), evidence.end());
//             })
//             .def("local_score", [](Score& self, 
//                                    FactorType variable_type, 
//                                    int variable, 
//                                    const std::vector<int> evidence) {
//                 return self.local_score(variable_type, variable, evidence.begin(), evidence.end());
//             });

//             return score;
//         } else {
//             return register_Score<Models...>(m);
//         }
//     }();

//     score.def("score", [](Score& self, const Model& m) {
//         return self.score(m);
//     })
//     .def("local_score", [](Score& self, const Model& m, const std::string& variable) {
//         return self.local_score(m, variable);
//     })
//     .def("local_score", [](Score& self, const Model& m, const int variable) {
//         return self.local_score(m, variable);
//     })
//     .def("local_score", [](Score& self, const Model& m, const std::string& variable, const std::vector<std::string> evidence) {
//         return self.local_score(m, variable, evidence.begin(), evidence.end());
//     })
//     .def("local_score", [](Score& self, const Model& m, const int variable, const std::vector<int> evidence) {
//         return self.local_score(m, variable, evidence.begin(), evidence.end());
//     });

//     return score;
// }

void pybindings_scores(py::module& root) {
    auto scores = root.def_submodule("scores", "Learning scores submodule.");

    // register_Score<GaussianNetwork, SemiparametricBN>(scores);
    py::class_<Score, std::shared_ptr<Score>>(scores, "Score")
        .def("score", [](const Score& self, const ConditionalBayesianNetworkBase& m) {
            if (self.compatible_bn(m))
                return self.score(m);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("score", [](const Score& self, const BayesianNetworkBase& m) {
            if (self.compatible_bn(m))
                return self.score(m);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("score_unsafe", [](const Score& self, const BayesianNetworkBase& m) {
            return self.score(m);
        })
        .def("local_score", [](const Score& self, const ConditionalBayesianNetworkBase& m, const std::string& variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const Score& self, const BayesianNetworkBase& m, const std::string& variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const Score& self, const BayesianNetworkBase& m, const std::string& variable) {
            return self.local_score(m, variable);
        })
        .def("local_score", [](const Score& self, const ConditionalBayesianNetworkBase& m, const int variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const Score& self, const BayesianNetworkBase& m, const int variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const Score& self, const BayesianNetworkBase& m, const int variable) {
            return self.local_score(m, variable);
        })
        .def("local_score", [](const Score& self,
                               const ConditionalBayesianNetworkBase& m,
                               const std::string& variable, 
                               const std::vector<std::string> evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const Score& self,
                               const BayesianNetworkBase& m,
                               const std::string& variable, 
                               const std::vector<std::string> evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const Score& self,
                                      const BayesianNetworkBase& m,
                                      const std::string& variable, 
                                      const std::vector<std::string> evidence) {
            return self.local_score(m, variable, evidence);
        })
        .def("local_score", [](const Score& self,
                               const ConditionalBayesianNetworkBase& m,
                               const int variable, 
                               const std::vector<int> evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const Score& self,
                               const BayesianNetworkBase& m,
                               const int variable, 
                               const std::vector<int> evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const Score& self,
                                      const BayesianNetworkBase& m,
                                      const int variable, 
                                      const std::vector<int> evidence) {
            return self.local_score(m, variable, evidence);
        })
        .def("ToString", &Score::ToString)
        .def("is_decomposable", &Score::is_decomposable)
        .def("type", &Score::type)
        .def("compatible_bn", py::overload_cast<const ConditionalBayesianNetworkBase&>(&Score::compatible_bn, py::const_))
        .def("compatible_bn", py::overload_cast<const BayesianNetworkBase&>(&Score::compatible_bn, py::const_));

    py::class_<ScoreSPBN, Score, std::shared_ptr<ScoreSPBN>>(scores, "ScoreSPBN")
    //  Include parent methods.
        .def("score", [](const ScoreSPBN& self, const ConditionalBayesianNetworkBase& m) {
            if (self.compatible_bn(m))
                return self.score(m);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("score", [](const ScoreSPBN& self, const BayesianNetworkBase& m) {
            if (self.compatible_bn(m))
                return self.score(m);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("score_unsafe", [](const ScoreSPBN& self, const BayesianNetworkBase& m) {
            return self.score(m);
        })
        .def("local_score", [](const ScoreSPBN& self, const ConditionalBayesianNetworkBase& m, const std::string& variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const ScoreSPBN& self, const BayesianNetworkBase& m, const std::string& variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const ScoreSPBN& self, const BayesianNetworkBase& m, const std::string& variable) {
            return self.local_score(m, variable);
        })
        .def("local_score", [](const ScoreSPBN& self, const ConditionalBayesianNetworkBase& m, const int variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const ScoreSPBN& self, const BayesianNetworkBase& m, const int variable) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const ScoreSPBN& self, const BayesianNetworkBase& m, const int variable) {
            return self.local_score(m, variable);
        })
        .def("local_score", [](const ScoreSPBN& self,
                               const ConditionalBayesianNetworkBase& m,
                               const std::string& variable, 
                               const std::vector<std::string> evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const ScoreSPBN& self,
                               const BayesianNetworkBase& m,
                               const std::string& variable, 
                               const std::vector<std::string> evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const ScoreSPBN& self,
                                      const BayesianNetworkBase& m,
                                      const std::string& variable, 
                                      const std::vector<std::string> evidence) {
            return self.local_score(m, variable, evidence);
        })
        .def("local_score", [](const ScoreSPBN& self,
                               const ConditionalBayesianNetworkBase& m,
                               const int variable, 
                               const std::vector<int> evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score", [](const ScoreSPBN& self,
                               const BayesianNetworkBase& m,
                               const int variable, 
                               const std::vector<int> evidence) {
            if (self.compatible_bn(m))
                return self.local_score(m, variable, evidence);
            else
                throw py::value_error("Bayesian network is incompatible with the score.");
        })
        .def("local_score_unsafe", [](const ScoreSPBN& self,
                                      const BayesianNetworkBase& m,
                                      const int variable, 
                                      const std::vector<int> evidence) {
            return self.local_score(m, variable, evidence);
        })
    // SPBN methods.
        .def("local_score", [](ScoreSPBN& self, FactorType variable_type, const std::string& variable, 
                                const std::vector<std::string> evidence) {
            return self.local_score(variable_type, variable, evidence);
        });

    py::class_<BIC, Score, std::shared_ptr<BIC>>(scores, "BIC")
        .def(py::init<const DataFrame&>());

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

    py::class_<DynamicScore, std::shared_ptr<DynamicScore>>(scores, "DynamicScore")
        .def("static_score", &DynamicScore::static_score)
        .def("transition_score", &DynamicScore::transition_score);

    py::class_<DynamicBIC, DynamicScore, std::shared_ptr<DynamicBIC>>(scores, "DynamicBIC")
        .def(py::init<const DynamicDataFrame&>());

    py::class_<DynamicCVLikelihood, DynamicScore, std::shared_ptr<DynamicCVLikelihood>>(scores, "DynamicCVLikelihood")
        .def(py::init<const DynamicDataFrame&, int>(),
                py::arg("df"),
                py::arg("k") = 10)
        .def(py::init<const DynamicDataFrame&, int, unsigned int>(),
                py::arg("df"),
                py::arg("k") = 10,
                py::arg("seed"));

    py::class_<DynamicHoldoutLikelihood, DynamicScore, std::shared_ptr<DynamicHoldoutLikelihood>>(scores, "DynamicHoldoutLikelihood")
        .def(py::init<const DynamicDataFrame&, double>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2)
        .def(py::init<const DynamicDataFrame&, double, unsigned int>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2,
                py::arg("seed"));
}