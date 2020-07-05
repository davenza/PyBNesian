#include <pybind11/pybind11.h>
// To overload operators.
#include <pybind11/operators.h>
#include <arrow/python/pyarrow.h>

#include <dataset/crossvalidation_adaptator.hpp>
#include <dataset/holdout_adaptator.hpp>

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <graph/dag.hpp>
#include <models/BayesianNetwork.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN_NodeType.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
// #include <learning/scores/bic.hpp>
#include <learning/parameter/mle.hpp>
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

using models::BayesianNetwork;
using models::NodeType;
using learning::scores::BIC;
using learning::scores::CVLikelihood;

using util::ArcVector;


template<typename DerivedBN>
py::class_<DerivedBN, BayesianNetwork<DerivedBN>> register_BayesianNetwork(py::module& m, const char* derivedbn_name) {
    using BaseClass = BayesianNetwork<DerivedBN>;
    std::string base_name = std::string("BayesianNetwork<") + derivedbn_name + ">";
    py::class_<BaseClass>(m, base_name.c_str())
        .def("num_nodes", &BaseClass::num_nodes)
        .def("num_edges", &BaseClass::num_edges)
        .def("nodes", &BaseClass::nodes)
        .def("indices", &BaseClass::indices)
        .def("index", py::overload_cast<const std::string&>(&BaseClass::index, py::const_))
        .def("contains_node", &BaseClass::contains_node)
        .def("name", py::overload_cast<int>(&BaseClass::name, py::const_))
        .def("num_parents", py::overload_cast<const std::string&>(&BaseClass::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&BaseClass::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&BaseClass::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&BaseClass::num_children, py::const_))
        .def("parents", py::overload_cast<const std::string&>(&BaseClass::parents, py::const_), py::return_value_policy::move)
        .def("parents", py::overload_cast<int>(&BaseClass::parents, py::const_), py::return_value_policy::move)
        .def("parent_indices", py::overload_cast<const std::string&>(&BaseClass::parent_indices, py::const_), py::return_value_policy::move)
        .def("parent_indices", py::overload_cast<int>(&BaseClass::parent_indices, py::const_), py::return_value_policy::move)
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
        .def("logpdf", &BaseClass::logpdf, py::return_value_policy::move)
        .def("slogpdf", &BaseClass::slogpdf);

    return py::class_<DerivedBN, BaseClass>(m, derivedbn_name)
            .def(py::init<const std::vector<std::string>&>())
            .def(py::init<const ArcVector&>())
            .def(py::init<const std::vector<std::string>&, const ArcVector&>());
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
                    return py::make_iterator(self.begin(), self.end()); }, 
            py::keep_alive<0, 1>())
        .def("fold", &CrossValidation::fold)
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
        .def("training_data", &HoldOut::training_data)
        .def("test_data", &HoldOut::test_data);

    auto factors = m.def_submodule("factors", "Factors submodule.");
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
        .def("logpdf", &LinearGaussianCPD::logpdf, py::return_value_policy::move)
        .def("slogpdf", &LinearGaussianCPD::slogpdf);

    py::class_<KDE>(continuous, "KDE")
        .def(py::init<std::vector<std::string>>())
        .def_property_readonly("variables", &KDE::variables)
        .def_property_readonly("n", &KDE::num_instances)
        .def_property_readonly("d", &KDE::num_variables)
        .def_property("bandwidth", &KDE::bandwidth, &KDE::setBandwidth)
        .def_property_readonly("fitted", &KDE::fitted)
        .def("fit", (void (KDE::*)(const DataFrame&))&KDE::fit)
        .def("logpdf", &KDE::logpdf, py::return_value_policy::move)
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
        .def("logpdf", &CKDE::logpdf, py::return_value_policy::move)
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
        .def("logpdf", &SemiparametricCPD::logpdf, py::return_value_policy::move)
        .def("slogpdf", &SemiparametricCPD::slogpdf);
    
    py::implicitly_convertible<LinearGaussianCPD, SemiparametricCPD>();
    py::implicitly_convertible<CKDE, SemiparametricCPD>();

    // //////////////////////////////
    // Include Different types of Graphs
    // /////////////////////////////

    auto models = m.def_submodule("models", "Models submodule.");
    
    py::class_<NodeType>(models, "NodeType")
        .def_property_readonly_static("LinearGaussianCPD", [](const py::object&) { 
            return NodeType(NodeType::LinearGaussianCPD);
        })
        .def_property_readonly_static("CKDE", [](const py::object&) { 
            return NodeType(NodeType::CKDE);
        })
        .def("opposite", &NodeType::opposite)
        .def(py::self == py::self)
        .def(py::self != py::self);

    register_BayesianNetwork<GaussianNetwork<>>(models, "GaussianNetwork");
    auto spbn = register_BayesianNetwork<SemiparametricBN<>>(models, "SemiparametricBN");

    spbn.def(py::init<const std::vector<std::string>&, NodeTypeVector&>())
        .def(py::init<const ArcVector&, NodeTypeVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcVector&, NodeTypeVector&>())
        .def("node_type", py::overload_cast<const std::string&>(&SemiparametricBN<>::node_type, py::const_))
        .def("node_type", py::overload_cast<int>(&SemiparametricBN<>::node_type, py::const_))
        .def("set_node_type", py::overload_cast<const std::string&, NodeType>(&SemiparametricBN<>::set_node_type))
        .def("set_node_type", py::overload_cast<int, NodeType>(&SemiparametricBN<>::set_node_type));
      
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
        .def(py::init<const DataFrame&, int>())
        .def(py::init<const DataFrame&, int, int>())
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
        .def("local_score", [](CVLikelihood& self, NodeType node_type, std::string var, std::vector<std::string> evidence) {
            return self.local_score(node_type, var, evidence.begin(), evidence.end());
        })
        .def("local_score", [](CVLikelihood& self, NodeType node_type, int idx, std::vector<int> evidence_idx) {
            return self.local_score(node_type, idx, evidence_idx.begin(), evidence_idx.end());
        });

    auto algorithms = learning.def_submodule("algorithms", "Learning algorithms");

    algorithms.def("hc", &learning::algorithms::hc, "Hill climbing estimate");
    // py::class_<GreedyHillClimbing>(algorithms, "GreedyHillClimbing")
    //         .def(py::init<>())
    //         .def("estimate")
}