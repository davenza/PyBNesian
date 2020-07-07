#include <pybind11/pybind11.h>
// To overload operators.
#include <pybind11/operators.h>
#include <arrow/python/pyarrow.h>

#include <dataset/crossvalidation_adaptator.hpp>
#include <dataset/holdout_adaptator.hpp>

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <factors/factors.hpp>
#include <graph/dag.hpp>
#include <models/BayesianNetwork.hpp>
#include <models/GaussianNetwork.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>
// #include <learning/scores/bic.hpp>
#include <learning/parameters/mle.hpp>
#include <learning/parameters/pybindings_mle.hpp>
#include <learning/operators/operators.hpp>
#include <learning/operators/pybindings_operators.hpp>
#include <learning/algorithms/hillclimbing.hpp>
#include <util/util_types.hpp>

namespace py = pybind11;

namespace pyarrow = arrow::py;

// using namespace factors::continuous;

using dataset::DataFrame, dataset::CrossValidation, dataset::HoldOut;

using namespace ::graph;

using factors::continuous::LinearGaussianCPD;
using factors::continuous::KDE;
using factors::continuous::CKDE;
using factors::FactorType;

using models::BayesianNetwork, models::BayesianNetworkType;
using learning::scores::BIC;
using learning::scores::CVLikelihood;
using learning::scores::HoldoutLikelihood;
using learning::operators::AddArc, learning::operators::RemoveArc, learning::operators::FlipArc,
      learning::operators::ChangeNodeType, learning::operators::OperatorTabuSet, 
      learning::operators::OperatorSetType, learning::operators::OperatorSet, 
      learning::operators::ArcOperatorSet;

using util::ArcVector;


template<typename DerivedBN>
py::class_<DerivedBN, BayesianNetwork<DerivedBN>> register_BayesianNetwork(py::module& m, const char* derivedbn_name) {
    using BaseClass = BayesianNetwork<DerivedBN>;
    std::string base_name = std::string("BayesianNetwork<") + derivedbn_name + ">";
    // TODO: Implement copy operation.
    py::class_<BaseClass>(m, base_name.c_str())
        .def("num_nodes", &BaseClass::num_nodes)
        .def("num_edges", &BaseClass::num_edges)
        .def("nodes", &BaseClass::nodes, py::return_value_policy::reference_internal)
        .def("indices", &BaseClass::indices, py::return_value_policy::reference_internal)
        .def("index", py::overload_cast<const std::string&>(&BaseClass::index, py::const_))
        .def("contains_node", &BaseClass::contains_node)
        .def("name", py::overload_cast<int>(&BaseClass::name, py::const_), py::return_value_policy::reference_internal)
        .def("num_parents", py::overload_cast<const std::string&>(&BaseClass::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&BaseClass::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&BaseClass::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&BaseClass::num_children, py::const_))
        .def("parents", py::overload_cast<const std::string&>(&BaseClass::parents, py::const_), py::return_value_policy::take_ownership)
        .def("parents", py::overload_cast<int>(&BaseClass::parents, py::const_), py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<const std::string&>(&BaseClass::parent_indices, py::const_), py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<int>(&BaseClass::parent_indices, py::const_), py::return_value_policy::take_ownership)
        .def("has_edge", py::overload_cast<const std::string&, const std::string&>(&BaseClass::has_edge, py::const_))
        .def("has_edge", py::overload_cast<int, int>(&BaseClass::has_edge, py::const_))
        .def("has_path", py::overload_cast<const std::string&, const std::string&>(&BaseClass::has_path, py::const_))
        .def("has_path", py::overload_cast<int, int>(&BaseClass::has_path, py::const_))
        .def("add_edge", py::overload_cast<const std::string&, const std::string&>(&BaseClass::add_edge))
        .def("add_edge", py::overload_cast<int, int>(&BaseClass::add_edge))
        .def("can_add_edge", py::overload_cast<const std::string&, const std::string&>(&BaseClass::can_add_edge, py::const_))
        .def("can_add_edge", py::overload_cast<int, int>(&BaseClass::can_add_edge, py::const_))
        .def("can_flip_edge", py::overload_cast<const std::string&, const std::string&>(&BaseClass::can_flip_edge))
        .def("can_flip_edge", py::overload_cast<int, int>(&BaseClass::can_flip_edge))
        .def("remove_edge", py::overload_cast<const std::string&, const std::string&>(&BaseClass::remove_edge))
        .def("remove_edge", py::overload_cast<int, int>(&BaseClass::remove_edge))
        .def("fit", &BaseClass::fit)
        .def("add_cpds", &BaseClass::add_cpds)
        .def("cpd", py::overload_cast<const std::string&>(&BaseClass::cpd), py::return_value_policy::reference_internal)
        .def("cpd", py::overload_cast<int>(&BaseClass::cpd), py::return_value_policy::reference_internal)
        .def("logpdf", &BaseClass::logpdf, py::return_value_policy::take_ownership)
        .def("slogpdf", &BaseClass::slogpdf);

    return py::class_<DerivedBN, BaseClass>(m, derivedbn_name)
            .def(py::init<const std::vector<std::string>&>())
            .def(py::init<const ArcVector&>())
            .def(py::init<const std::vector<std::string>&, const ArcVector&>());
}

template<typename Model>
void register_ArcOperators(py::module& m, const char* model_name) {
    std::string operator_name = std::string("Operator<") + model_name + ">";
    py::class_<Operator<Model>>(m, operator_name.c_str())
        .def_property_readonly("delta", &Operator<Model>::delta)
        .def_property_readonly("type", &Operator<Model>::type)
        .def("apply", &Operator<Model>::apply)
        .def("opposite", &Operator<Model>::opposite, py::return_value_policy::take_ownership);

    std::string arcoperator_name = std::string("ArcOperator<") + model_name + ">";
    py::class_<ArcOperator<Model>, Operator<Model>>(m, arcoperator_name.c_str())
        .def_property_readonly("source", &ArcOperator<Model>::source)
        .def_property_readonly("target", &ArcOperator<Model>::target);

    std::string addarc_name = std::string("AddArc<") + model_name + ">";
    py::class_<AddArc<Model>, ArcOperator<Model>>(m, addarc_name.c_str())
        .def(py::init<std::string, std::string, double>())
        .def("apply", &AddArc<Model>::apply)
        .def("opposite", &AddArc<Model>::opposite, py::return_value_policy::take_ownership);

    std::string removearc_name = std::string("RemoveArc<") + model_name + ">";
    py::class_<RemoveArc<Model>, ArcOperator<Model>>(m, removearc_name.c_str())
        .def(py::init<std::string, std::string, double>())
        .def("apply", &RemoveArc<Model>::apply)
        .def("opposite", &RemoveArc<Model>::opposite, py::return_value_policy::take_ownership);

    std::string fliparc_name = std::string("FlipArc<") + model_name + ">";
    py::class_<FlipArc<Model>, ArcOperator<Model>>(m, fliparc_name.c_str())
        .def(py::init<std::string, std::string, double>())
        .def("apply", &FlipArc<Model>::apply)
        .def("opposite", &FlipArc<Model>::opposite, py::return_value_policy::take_ownership);
}

template<typename Model>
void register_OperatorTabuSet(py::module& m, const char* model_name) {
    std::string name = std::string("OperatorTabuSet<") + model_name + ">";

    // TODO: Implement copy operation.
    py::class_<OperatorTabuSet<Model>>(m, name.c_str())
        .def(py::init<>())
        .def("insert", py::overload_cast<Operator<Model>*>(&OperatorTabuSet<Model>::insert))
        .def("contains", py::overload_cast<Operator<Model>*>(&OperatorTabuSet<Model>::contains, py::const_))
        .def("clear", &OperatorTabuSet<Model>::clear)
        .def("empty", &OperatorTabuSet<Model>::empty);
}

template<typename Model>
void register_OperatorSet(py::module& m, const char* model_name) {
    std::string name = std::string("OperatorSet<") + model_name + ">";

    py::class_<OperatorSet<Model>>(m, name.c_str())
        .def("cache_scores", &OperatorSet<Model>::cache_scores)
        .def("find_max", [](OperatorSet<Model>& self, Model& model) { return self.find_max(model); }, 
            py::return_value_policy::take_ownership)
        .def("find_max", [](OperatorSet<Model>& self, Model& model, OperatorTabuSet<Model>& tabu) { 
            return self.find_max(model, tabu); 
        }, py::return_value_policy::take_ownership)
        .def("update_scores", &OperatorSet<Model>::update_scores);

    // if constexpr (util::is_gaussian_network_v<Model>) {
    //     std::string arc_bic_name = std::string("ArcOperatorSet<") + model_name + ", BIC>";


    //     py::class_<ArcOperatorSet<Model, BIC>, OperatorSet<Model>>(m, arc_bic_name.c_str())
    //         .def(py::init<Model&, const BIC, ArcVector&, ArcVector&, const VectorXd&, int>())
    //         .def("cache_scores", &ArcOperatorSet<Model, BIC>::cache_scores)
    //         .def("find_max", [](ArcOperatorSet<Model, BIC>& self, Model& model) { return self.find_max(model); }, 
    //             py::return_value_policy::take_ownership)
    //         .def("find_max", [](ArcOperatorSet<Model, BIC>& self, Model& model, OperatorTabuSet<Model>& tabu) { 
    //             return self.find_max(model, tabu); 
    //         }, py::return_value_policy::take_ownership)
    //         .def("update_scores", &ArcOperatorSet<Model, BIC>::update_scores);
    // }
}

PYBIND11_MODULE(pgm_dataset, m) {
    pyarrow::import_pyarrow();

    m.doc() = "pybind11 data plugin"; // optional module docstring

    auto dataset = m.def_submodule("dataset", "Dataset functionality.");

    py::class_<CrossValidation>(dataset, "CrossValidation")
        .def(py::init<DataFrame, int, bool>(), 
                    py::arg("df"), 
                    py::arg("k") = 10, 
                    py::arg("include_null") = false)
        .def(py::init<DataFrame, int, int, bool>(), 
                    py::arg("df"), 
                    py::arg("k") = 10, 
                    py::arg("seed"), 
                    py::arg("include_null") = false)
        .def("__iter__", [](CrossValidation& self) { 
                    return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>())
        .def("fold", &CrossValidation::fold, py::return_value_policy::take_ownership)
        .def("loc", [](CrossValidation& self, std::string name) { return self.loc(name); })
        .def("loc", [](CrossValidation& self, int idx) { return self.loc(idx); })
        .def("loc", [](CrossValidation& self, std::vector<std::string> v) { return self.loc(v); })
        .def("loc", [](CrossValidation& self, std::vector<int> v) { return self.loc(v); })
        .def("indices", [](CrossValidation& self) { 
                    return py::make_iterator(self.begin_indices(), self.end_indices()); }
                    );

    py::class_<HoldOut>(dataset, "HoldOut")
        .def(py::init<const DataFrame&, double, bool>(), 
                    py::arg("df"), 
                    py::arg("test_ratio") = 0.2, 
                    py::arg("include_null") = false)
        .def(py::init<const DataFrame&, double, int, bool>(), 
                    py::arg("df"), 
                    py::arg("test_ratio") = 0.2, 
                    py::arg("seed"), 
                    py::arg("include_null") = false)
        .def("training_data", &HoldOut::training_data, py::return_value_policy::reference_internal)
        .def("test_data", &HoldOut::test_data, py::return_value_policy::reference_internal);

    auto factors = m.def_submodule("factors", "Factors submodule.");

    py::class_<FactorType>(factors, "FactorType")
        .def_property_readonly_static("LinearGaussianCPD", [](const py::object&) { 
            return FactorType(FactorType::LinearGaussianCPD);
        })
        .def_property_readonly_static("CKDE", [](const py::object&) { 
            return FactorType(FactorType::CKDE);
        })
        .def("opposite", &FactorType::opposite)
        .def(py::self == py::self)
        .def(py::self != py::self);

    auto continuous = factors.def_submodule("continuous", "Continuous factors submodule.");

    py::class_<LinearGaussianCPD>(continuous, "LinearGaussianCPD")
        .def(py::init<const std::string, const std::vector<std::string>>())
        .def(py::init<const std::string, const std::vector<std::string>, const std::vector<double>, double>())
        .def_property_readonly("variable", &LinearGaussianCPD::variable)
        .def_property_readonly("evidence", &LinearGaussianCPD::evidence)
        .def_property("beta", &LinearGaussianCPD::beta, &LinearGaussianCPD::setBeta)
        .def_property("variance", &LinearGaussianCPD::variance, &LinearGaussianCPD::setVariance)
        .def_property_readonly("fitted", &LinearGaussianCPD::fitted)
        .def("fit", &LinearGaussianCPD::fit)
        .def("logpdf", &LinearGaussianCPD::logpdf, py::return_value_policy::take_ownership)
        .def("slogpdf", &LinearGaussianCPD::slogpdf);

    py::class_<KDE>(continuous, "KDE")
        .def(py::init<std::vector<std::string>>())
        .def_property_readonly("variables", &KDE::variables)
        .def_property_readonly("n", &KDE::num_instances)
        .def_property_readonly("d", &KDE::num_variables)
        .def_property("bandwidth", &KDE::bandwidth, &KDE::setBandwidth)
        .def_property_readonly("fitted", &KDE::fitted)
        .def("fit", (void (KDE::*)(const DataFrame&))&KDE::fit)
        .def("logpdf", &KDE::logpdf, py::return_value_policy::take_ownership)
        .def("slogpdf", &KDE::slogpdf);

    py::class_<CKDE>(continuous, "CKDE")
        .def(py::init<const std::string, const std::vector<std::string>>())
        .def_property_readonly("variable", &CKDE::variable)
        .def_property_readonly("evidence", &CKDE::evidence)
        .def_property_readonly("n", &CKDE::num_instances)
        .def_property_readonly("kde_joint", &CKDE::kde_joint)
        .def_property_readonly("kde_marg", &CKDE::kde_marg)
        .def_property_readonly("fitted", &CKDE::fitted)
        .def("fit", &CKDE::fit)
        .def("logpdf", &CKDE::logpdf, py::return_value_policy::take_ownership)
        .def("slogpdf", &CKDE::slogpdf);

    py::class_<SemiparametricCPD>(continuous, "SemiparametricCPD")
        .def(py::init<LinearGaussianCPD>())
        .def(py::init<CKDE>())
        .def_property_readonly("variable", &SemiparametricCPD::variable)
        .def_property_readonly("evidence", &SemiparametricCPD::evidence)
        .def_property_readonly("node_type", &SemiparametricCPD::node_type)
        .def_property_readonly("fitted", &SemiparametricCPD::fitted)
        .def("as_lg", &SemiparametricCPD::as_lg, py::return_value_policy::reference_internal)
        .def("as_ckde", &SemiparametricCPD::as_ckde, py::return_value_policy::reference_internal)
        .def("fit", &SemiparametricCPD::fit)
        .def("logpdf", &SemiparametricCPD::logpdf, py::return_value_policy::take_ownership)
        .def("slogpdf", &SemiparametricCPD::slogpdf);
    
    py::implicitly_convertible<LinearGaussianCPD, SemiparametricCPD>();
    py::implicitly_convertible<CKDE, SemiparametricCPD>();

    // //////////////////////////////
    // Include Different types of Graphs
    // /////////////////////////////

    auto models = m.def_submodule("models", "Models submodule.");

    py::class_<BayesianNetworkType>(models, "BayesianNetworkType")
        .def_property_readonly_static("GBN", [](const py::object&) { 
            return BayesianNetworkType(BayesianNetworkType::GBN);
        })
        .def_property_readonly_static("SPBN", [](const py::object&) { 
            return BayesianNetworkType(BayesianNetworkType::SPBN);
        })
        .def(py::self == py::self)
        .def(py::self != py::self);
    
    register_BayesianNetwork<GaussianNetwork<>>(models, "GaussianNetwork");
    auto spbn = register_BayesianNetwork<SemiparametricBN<>>(models, "SemiparametricBN");

    spbn.def(py::init<const std::vector<std::string>&, FactorTypeVector&>())
        .def(py::init<const ArcVector&, FactorTypeVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcVector&, FactorTypeVector&>())
        .def("node_type", py::overload_cast<const std::string&>(&SemiparametricBN<>::node_type, py::const_))
        .def("node_type", py::overload_cast<int>(&SemiparametricBN<>::node_type, py::const_))
        .def("set_node_type", py::overload_cast<const std::string&, FactorType>(&SemiparametricBN<>::set_node_type))
        .def("set_node_type", py::overload_cast<int, FactorType>(&SemiparametricBN<>::set_node_type));
      
    auto learning = m.def_submodule("learning", "Learning submodule");
    auto scores = learning.def_submodule("scores", "Learning scores submodule.");

    py::class_<BIC>(scores, "BIC")
        .def(py::init<const DataFrame&>())
        .def("score", &BIC::score<GaussianNetwork<>>)
        .def("local_score", [](BIC& self, GaussianNetwork<> g, std::string var) {
            return self.local_score(g, var);
        })
        .def("local_score", [](BIC& self, GaussianNetwork<> g, int idx) {
            return self.local_score(g, idx);
        })
        .def("local_score", [](BIC& self, GaussianNetwork<> g, std::string var, std::vector<std::string> evidence) {
            return self.local_score(g, var, evidence.begin(), evidence.end());
        })
        .def("local_score", [](BIC& self, GaussianNetwork<> g, int idx, std::vector<int> evidence_idx) {
            return self.local_score(g, idx, evidence_idx.begin(), evidence_idx.end());
        });

    py::class_<CVLikelihood>(scores, "CVLikelihood")
        .def(py::init<const DataFrame&, int>(),
                py::arg("df"),
                py::arg("k") = 10)
        .def(py::init<const DataFrame&, int, int>(),
                py::arg("df"),
                py::arg("k") = 10,
                py::arg("seed"))
        .def_property_readonly("cv", &CVLikelihood::cv)
        .def("score", &CVLikelihood::score<SemiparametricBN<>>)
        .def("score", &CVLikelihood::score<GaussianNetwork<>>)
        .def("local_score", [](CVLikelihood& self, SemiparametricBN<> g, std::string var) {
            return self.local_score(g, var);
        })
        .def("local_score", [](CVLikelihood& self, GaussianNetwork<> g, std::string var) {
            return self.local_score(g, var);
        })
        .def("local_score", [](CVLikelihood& self, SemiparametricBN<> g, int idx) {
            return self.local_score(g, idx);
        })
        .def("local_score", [](CVLikelihood& self, GaussianNetwork<> g, int idx) {
            return self.local_score(g, idx);
        })
        .def("local_score", [](CVLikelihood& self, SemiparametricBN<> g, std::string var, std::vector<std::string> evidence) {
            return self.local_score(g, var, evidence.begin(), evidence.end());
        })
        .def("local_score", [](CVLikelihood& self, SemiparametricBN<> g, int idx, std::vector<int> evidence_idx) {
            return self.local_score(g, idx, evidence_idx.begin(), evidence_idx.end());
        })
        .def("local_score", [](CVLikelihood& self, GaussianNetwork<> g, std::string var, std::vector<std::string> evidence) {
            return self.local_score(g, var, evidence.begin(), evidence.end());
        })
        .def("local_score", [](CVLikelihood& self, GaussianNetwork<> g, int idx, std::vector<int> evidence_idx) {
            return self.local_score(g, idx, evidence_idx.begin(), evidence_idx.end());
        })
        .def("local_score", [](CVLikelihood& self, FactorType node_type, std::string var, std::vector<std::string> evidence) {
            return self.local_score(node_type, var, evidence.begin(), evidence.end());
        })
        .def("local_score", [](CVLikelihood& self, FactorType node_type, int idx, std::vector<int> evidence_idx) {
            return self.local_score(node_type, idx, evidence_idx.begin(), evidence_idx.end());
        });

    py::class_<HoldoutLikelihood>(scores, "HoldoutLikelihood")
        .def(py::init<const DataFrame&, double>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2)
        .def(py::init<const DataFrame&, double, int>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2,
                py::arg("seed"))
        .def_property_readonly("holdout", &HoldoutLikelihood::holdout)
        .def("training_data", &HoldoutLikelihood::training_data, py::return_value_policy::reference_internal)
        .def("test_data", &HoldoutLikelihood::test_data, py::return_value_policy::reference_internal)
        .def("score", &HoldoutLikelihood::score<SemiparametricBN<>>)
        .def("score", &HoldoutLikelihood::score<GaussianNetwork<>>)
        .def("local_score", [](HoldoutLikelihood& self, SemiparametricBN<> g, std::string var) {
            return self.local_score(g, var);
        })
        .def("local_score", [](HoldoutLikelihood& self, GaussianNetwork<> g, std::string var) {
            return self.local_score(g, var);
        })
        .def("local_score", [](HoldoutLikelihood& self, SemiparametricBN<> g, int idx) {
            return self.local_score(g, idx);
        })
        .def("local_score", [](HoldoutLikelihood& self, GaussianNetwork<> g, int idx) {
            return self.local_score(g, idx);
        })
        .def("local_score", [](HoldoutLikelihood& self, SemiparametricBN<> g, std::string var, std::vector<std::string> evidence) {
            return self.local_score(g, var, evidence.begin(), evidence.end());
        })
        .def("local_score", [](HoldoutLikelihood& self, SemiparametricBN<> g, int idx, std::vector<int> evidence_idx) {
            return self.local_score(g, idx, evidence_idx.begin(), evidence_idx.end());
        })
        .def("local_score", [](HoldoutLikelihood& self, GaussianNetwork<> g, std::string var, std::vector<std::string> evidence) {
            return self.local_score(g, var, evidence.begin(), evidence.end());
        })
        .def("local_score", [](HoldoutLikelihood& self, GaussianNetwork<> g, int idx, std::vector<int> evidence_idx) {
            return self.local_score(g, idx, evidence_idx.begin(), evidence_idx.end());
        });

    auto parameters = learning.def_submodule("parameters", "Learning parameters submodule.");

    parameters.def("MLE", &learning::parameters::mle_python_wrapper, py::return_value_policy::take_ownership);

    // TODO Fit LinearGaussianCPD with ParamsClass.
    py::class_<LinearGaussianCPD::ParamsClass>(parameters, "MLELinearGaussianParams")
        .def_readwrite("beta", &LinearGaussianCPD::ParamsClass::beta)
        .def_readwrite("variance", &LinearGaussianCPD::ParamsClass::variance);

    py::class_<MLE<LinearGaussianCPD>>(parameters, "MLE<LinearGaussianCPD>")
        .def("estimate", [](MLE<LinearGaussianCPD> self, const DataFrame& df, std::string var, std::vector<std::string> evidence) {
            return self.estimate(df, var, evidence.begin(), evidence.end());
        }, py::return_value_policy::take_ownership)
        .def("estimate", [](MLE<LinearGaussianCPD> self, const DataFrame& df, int idx, std::vector<int> evidence_idx) {
            return self.estimate(df, idx, evidence_idx.begin(), evidence_idx.end());
        }, py::return_value_policy::take_ownership);

    auto operators = learning.def_submodule("operators", "Learning operators submodule");

    py::class_<OperatorType>(operators, "OperatorType")
        .def_property_readonly_static("ADD_ARC", [](const py::object&) { 
            return OperatorType(OperatorType::ADD_ARC);
        })
        .def_property_readonly_static("REMOVE_ARC", [](const py::object&) { 
            return OperatorType(OperatorType::REMOVE_ARC);
        })
        .def_property_readonly_static("FLIP_ARC", [](const py::object&) { 
            return OperatorType(OperatorType::FLIP_ARC);
        })
        .def_property_readonly_static("CHANGE_NODE_TYPE", [](const py::object&) { 
            return OperatorType(OperatorType::CHANGE_NODE_TYPE);
        })
        .def(py::self == py::self)
        .def(py::self != py::self);

    register_ArcOperators<GaussianNetwork<>>(operators, "GaussianNetwork");
    register_ArcOperators<SemiparametricBN<>>(operators, "SemiparametricBN");

    py::class_<ChangeNodeType<SemiparametricBN<>>, Operator<SemiparametricBN<>>>(m, "ChangeNodeType<SemiparametricBN>")
        .def(py::init<std::string, FactorType, double>())
        .def_property_readonly("node", &ChangeNodeType<SemiparametricBN<>>::node)
        .def_property_readonly("node_type", &ChangeNodeType<SemiparametricBN<>>::node_type)
        .def("apply", &ChangeNodeType<SemiparametricBN<>>::apply)
        .def("opposite", &ChangeNodeType<SemiparametricBN<>>::opposite);

    operators.def("AddArc", &learning::operators::addarc_wrapper_constructor, py::return_value_policy::take_ownership);
    operators.def("RemoveArc", &learning::operators::removearc_wrapper_constructor, py::return_value_policy::take_ownership);
    operators.def("FlipArc", &learning::operators::fliparc_wrapper_constructor, py::return_value_policy::take_ownership);
    operators.def("ChangeNodeType", &learning::operators::changenodetype_wrapper_constructor, py::return_value_policy::take_ownership);

    register_OperatorTabuSet<GaussianNetwork<>>(operators, "GaussianNetwork");
    register_OperatorTabuSet<SemiparametricBN<>>(operators, "SemiparametricBN");

    operators.def("OperatorTabuSet", &learning::operators::operatortabuset_wrapper_constructor, py::return_value_policy::take_ownership);

    py::class_<OperatorSetType>(operators, "OperatorSetType")
        .def_property_readonly_static("ARCS", [](const py::object&) { 
            return OperatorSetType(OperatorSetType::ARCS);
        })
        .def_property_readonly_static("NODE_TYPE", [](const py::object&) { 
            return OperatorSetType(OperatorSetType::NODE_TYPE);
        })
        .def(py::self == py::self)
        .def(py::self != py::self);

    register_OperatorSet<GaussianNetwork<>>(operators, "GaussianNetwork");



    auto algorithms = learning.def_submodule("algorithms", "Learning algorithms");

    algorithms.def("hc", &learning::algorithms::hc, "Hill climbing estimate");
    // py::class_<GreedyHillClimbing>(algorithms, "GreedyHillClimbing")
    //         .def(py::init<>())
    //         .def("estimate")
}