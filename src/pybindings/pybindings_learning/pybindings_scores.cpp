#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <learning/scores/scores.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>

namespace py = pybind11;

using learning::scores::Score, learning::scores::BIC, learning::scores::CVLikelihood, 
      learning::scores::HoldoutLikelihood;

template<typename Model, typename... Models>
py::class_<Score, std::shared_ptr<Score>> register_Score(py::module& m) {
    auto score = [&m](){
        if constexpr (sizeof...(Models) == 0) {
            py::class_<Score, std::shared_ptr<Score>> score(m, "Score");
            score.def("is_decomposable", &Score::is_decomposable)
            .def("type", &Score::type)
            .def("local_score", [](Score& self, 
                                   FactorType variable_type, 
                                   const std::string& variable, 
                                   const std::vector<std::string> evidence) {
                return self.local_score(variable_type, variable, evidence.begin(), evidence.end());
            })
            .def("local_score", [](Score& self, 
                                   FactorType variable_type, 
                                   int variable, 
                                   const std::vector<int> evidence) {
                return self.local_score(variable_type, variable, evidence.begin(), evidence.end());
            });

            return score;
        } else {
            return register_Score<Models...>(m);
        }
    }();

    score.def("score", [](Score& self, const Model& m) {
        return self.score(m);
    })
    .def("local_score", [](Score& self, const Model& m, const std::string& variable) {
        return self.local_score(m, variable);
    })
    .def("local_score", [](Score& self, const Model& m, const int variable) {
        return self.local_score(m, variable);
    })
    .def("local_score", [](Score& self, const Model& m, const std::string& variable, const std::vector<std::string> evidence) {
        return self.local_score(m, variable, evidence.begin(), evidence.end());
    })
    .def("local_score", [](Score& self, const Model& m, const int variable, const std::vector<int> evidence) {
        return self.local_score(m, variable, evidence.begin(), evidence.end());
    });

    return score;
}

void pybindings_scores(py::module& root) {
    auto scores = root.def_submodule("scores", "Learning scores submodule.");

    register_Score<GaussianNetwork, SemiparametricBN>(scores);

    py::class_<BIC, Score, std::shared_ptr<BIC>>(scores, "BIC")
        .def(py::init<const DataFrame&>());

    py::class_<CVLikelihood, Score, std::shared_ptr<CVLikelihood>>(scores, "CVLikelihood")
        .def(py::init<const DataFrame&, int>(),
                py::arg("df"),
                py::arg("k") = 10)
        .def(py::init<const DataFrame&, int, long unsigned int>(),
                py::arg("df"),
                py::arg("k") = 10,
                py::arg("seed"))
        .def_property_readonly("cv", &CVLikelihood::cv);

    py::class_<HoldoutLikelihood, Score, std::shared_ptr<HoldoutLikelihood>>(scores, "HoldoutLikelihood")
        .def(py::init<const DataFrame&, double>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2)
        .def(py::init<const DataFrame&, double, long unsigned int>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2,
                py::arg("seed"))
        .def_property_readonly("holdout", &HoldoutLikelihood::holdout)
        .def("training_data", &HoldoutLikelihood::training_data, py::return_value_policy::reference_internal)
        .def("test_data", &HoldoutLikelihood::test_data, py::return_value_policy::reference_internal);
}