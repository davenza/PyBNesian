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

using models::BayesianNetworkBase, models::BayesianNetwork, models::BayesianNetworkType;
using learning::scores::BIC;
using learning::scores::CVLikelihood;
using learning::scores::HoldoutLikelihood;
using learning::operators::AddArc, learning::operators::RemoveArc, learning::operators::FlipArc,
      learning::operators::ChangeNodeType, learning::operators::OperatorTabuSet, 
      learning::operators::OperatorSetType, learning::operators::OperatorSet, 
      learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet,
      learning::operators::OperatorPool;

using learning::operators::ArcOperatorSet_constructor, learning::operators::ChangeNodeTypeSet_constructor;

using util::ArcVector;


template<typename DerivedBN>
py::class_<DerivedBN, BayesianNetwork<DerivedBN>> register_BayesianNetwork(py::module& m, const char* derivedbn_name) {
    using BaseClass = BayesianNetwork<DerivedBN>;
    std::string base_name = std::string("BayesianNetwork<") + derivedbn_name + ">";
    // TODO: Implement copy operation.
    py::class_<BaseClass, BayesianNetworkBase>(m, base_name.c_str())
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

void register_ArcOperators(py::module& m) {
    py::class_<Operator, std::shared_ptr<Operator>>(m, "Operator")
        .def_property_readonly("delta", &Operator::delta)
        .def_property_readonly("type", &Operator::type)
        .def("apply", &Operator::apply)
        .def("opposite", &Operator::opposite, py::return_value_policy::take_ownership);

    py::class_<ArcOperator, Operator, std::shared_ptr<ArcOperator>>(m, "ArcOperator")
        .def_property_readonly("source", &ArcOperator::source)
        .def_property_readonly("target", &ArcOperator::target);

    py::class_<AddArc, ArcOperator, std::shared_ptr<AddArc>>(m, "AddArc")
        .def(py::init<std::string, std::string, double>())
        .def("apply", &AddArc::apply)
        .def("opposite", &AddArc::opposite, py::return_value_policy::take_ownership);

    py::class_<RemoveArc, ArcOperator, std::shared_ptr<RemoveArc>>(m, "RemoveArc")
        .def(py::init<std::string, std::string, double>())
        .def("apply", &RemoveArc::apply)
        .def("opposite", &RemoveArc::opposite, py::return_value_policy::take_ownership);

    py::class_<FlipArc, ArcOperator, std::shared_ptr<FlipArc>>(m, "FlipArc")
        .def(py::init<std::string, std::string, double>())
        .def("apply", &FlipArc::apply)
        .def("opposite", &FlipArc::opposite, py::return_value_policy::take_ownership);
}

void register_OperatorTabuSet(py::module& m) {

    // TODO: Implement copy operation.
    py::class_<OperatorTabuSet>(m, "OperatorTabuSet")
        .def(py::init<>())
        .def("insert", py::overload_cast<std::shared_ptr<Operator>>(&OperatorTabuSet::insert))
        .def("contains", py::overload_cast<std::shared_ptr<Operator>&>(&OperatorTabuSet::contains, py::const_))
        .def("clear", &OperatorTabuSet::clear)
        .def("empty", &OperatorTabuSet::empty);
}


template<typename Class, typename Model, typename... Models>
py::class_<Class, std::shared_ptr<Class>> register_OperatorSet(py::module& m, const char* class_name) {

    auto op_set = [&m, class_name]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<Class, std::shared_ptr<Class>> op_set(m, class_name);
            op_set.def(py::init<>());
            return op_set;
        } else {
            return register_OperatorSet<Class, Models...>(m, class_name);
        }
    }();

    op_set.def("cache_scores", [](Class& self, Model& model) {
        self.cache_scores(model);
    });
    op_set.def("find_max", [](Class& self, Model& model) {
        return self.find_max(model);
    });
    op_set.def("find_max", [](Class& self, Model& model, OperatorTabuSet& tabu) {
        return self.find_max(model, tabu);
    });
    op_set.def("update_scores", [](Class& self, Model& model, Operator& op) {
        return self.update_scores(model, op);
    });

    return op_set;
}

template<typename Score, typename Model, typename... Models>
py::class_<ArcOperatorSet<Score>, OperatorSet, std::shared_ptr<ArcOperatorSet<Score>>> register_ArcOperatorSet(py::module& m, const char* score_name) {
    std::string arc_score_name = std::string("ArcOperatorSet<") + score_name + ">";
    auto op_set = [&m, &arc_score_name, score_name]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<ArcOperatorSet<Score>, OperatorSet, std::shared_ptr<ArcOperatorSet<Score>>> op_set(m, arc_score_name.c_str());
            op_set.def(py::init<Model&, const Score, ArcVector&, ArcVector&, int>());
            return op_set;
        } else {
            return register_ArcOperatorSet<Score, Models...>(m, score_name);
        }
    }();

    m.def("ArcOperatorSet", [](Model& model, const Score score, ArcVector& whitelist, ArcVector& blacklist,
                    int max_indegree) {
        return ArcOperatorSet_constructor(model, score, whitelist, blacklist, max_indegree);
    });

    op_set.def("cache_scores", [](ArcOperatorSet<Score>& self, Model& model) {
        if constexpr(util::is_compatible_score_v<Model, Score>) {
            self.cache_scores(model);
        } else {
            self.OperatorSet::cache_scores(model);
        }
    });
    op_set.def("find_max", [](ArcOperatorSet<Score>& self, Model& model) {
        if constexpr(util::is_compatible_score_v<Model, Score>) {
            return self.find_max(model);
        } else {
            return self.OperatorSet::find_max(model);
        }
    });
    op_set.def("find_max", [](ArcOperatorSet<Score>& self, Model& model, OperatorTabuSet& tabu) {
        if constexpr(util::is_compatible_score_v<Model, Score>) {
            return self.find_max(model, tabu);
        } else {
            return self.OperatorSet::find_max(model, tabu);
        }
    });
    op_set.def("update_scores", [](ArcOperatorSet<Score>& self, Model& model, Operator& op) {
        if constexpr(util::is_compatible_score_v<Model, Score>) {
            self.update_scores(model, op);
        } else {
            self.OperatorSet::update_scores(model, op);
        }
    });

    return op_set;
}

template<typename Score, typename Model, typename... Models>
py::class_<ChangeNodeTypeSet<Score>, OperatorSet, std::shared_ptr<ChangeNodeTypeSet<Score>>> register_ChangeNodeTypeSet(py::module& m, const char* score_name) {
    std::string nodetype_score_name = std::string("ChangeNodeTypeSet<") + score_name + ">";
    auto op_set = [&m, &nodetype_score_name, score_name]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<ChangeNodeTypeSet<Score>, OperatorSet, std::shared_ptr<ChangeNodeTypeSet<Score>>> op_set(m, nodetype_score_name.c_str());
            op_set.def(py::init<Model&, const Score, FactorTypeVector&>());
            return op_set;
        } else {
            return register_ChangeNodeTypeSet<Score, Models...>(m, score_name);
        }
    }();


    m.def("ChangeNodeTypeSet", [](Model& model, const Score score, FactorTypeVector& type_whitelist) {
        return ChangeNodeTypeSet_constructor(model, score, type_whitelist);
    });


    op_set.def("cache_scores", [](ChangeNodeTypeSet<Score>& self, Model& model) {
        if constexpr(util::is_compatible_score_v<Model, Score>) {
            self.cache_scores(model);
        } else {
            self.OperatorSet::cache_scores(model);
        }
    });
    op_set.def("find_max", [](ChangeNodeTypeSet<Score>& self, Model& model) {
        if constexpr(util::is_compatible_score_v<Model, Score>) {
            return self.find_max(model);
        } else {
            return self.OperatorSet::find_max(model);
        }
    });
    op_set.def("find_max", [](ChangeNodeTypeSet<Score>& self, Model& model, OperatorTabuSet& tabu) {
        if constexpr(util::is_compatible_score_v<Model, Score>) {
            return self.find_max(model, tabu);
        } else {
            return self.OperatorSet::find_max(model, tabu);
        }
    });
    op_set.def("update_scores", [](ChangeNodeTypeSet<Score>& self, Model& model, Operator& op) {
        if constexpr(util::is_compatible_score_v<Model, Score>) {
            self.update_scores(model, op);
        } else {
            self.OperatorSet::update_scores(model, op);
        }
    });

    return op_set;
}


template<typename Score, typename Model, typename... Models>
py::class_<OperatorPool<Score>> register_OperatorPool(py::module& m, const char* score_name) {
    std::string pool_score_name = std::string("OperatorPool<") + score_name + ">";
    
    auto pool = [&m, &pool_score_name, score_name]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<OperatorPool<Score>> pool(m, pool_score_name.c_str());
            pool.def(py::init<Model&, const Score, std::vector<std::shared_ptr<OperatorSet>>>());
            return pool;
        } else {
            return register_OperatorPool<Score, Models...>(m, score_name);
        }
    }();

    pool.def("cache_scores", [](OperatorPool<Score>& self, Model& model) {
        self.cache_scores(model);
    });
    pool.def("find_max", [](OperatorPool<Score>& self, Model& model) {
        return self.find_max(model);
    });
    pool.def("find_max", [](OperatorPool<Score>& self, Model& model, OperatorTabuSet& tabu) {
        return self.find_max(model, tabu);
    });
    pool.def("update_scores", [](OperatorPool<Score>& self, Model& model, Operator& op) {
        self.update_scores(model, op);
    });

    return pool;
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

    py::class_<BayesianNetworkBase>(models, "BayesianNetworkBase")
        .def("num_nodes", &BayesianNetworkBase::num_nodes)
        .def("num_edges", &BayesianNetworkBase::num_edges)
        .def("nodes", &BayesianNetworkBase::nodes, py::return_value_policy::reference_internal)
        .def("indices", &BayesianNetworkBase::indices, py::return_value_policy::reference_internal)
        .def("index", py::overload_cast<const std::string&>(&BayesianNetworkBase::index, py::const_))
        .def("contains_node", &BayesianNetworkBase::contains_node)
        .def("name", py::overload_cast<int>(&BayesianNetworkBase::name, py::const_), py::return_value_policy::reference_internal)
        .def("num_parents", py::overload_cast<const std::string&>(&BayesianNetworkBase::num_parents, py::const_))
        .def("num_parents", py::overload_cast<int>(&BayesianNetworkBase::num_parents, py::const_))
        .def("num_children", py::overload_cast<const std::string&>(&BayesianNetworkBase::num_children, py::const_))
        .def("num_children", py::overload_cast<int>(&BayesianNetworkBase::num_children, py::const_))
        .def("parents", py::overload_cast<const std::string&>(&BayesianNetworkBase::parents, py::const_), py::return_value_policy::take_ownership)
        .def("parents", py::overload_cast<int>(&BayesianNetworkBase::parents, py::const_), py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<const std::string&>(&BayesianNetworkBase::parent_indices, py::const_), py::return_value_policy::take_ownership)
        .def("parent_indices", py::overload_cast<int>(&BayesianNetworkBase::parent_indices, py::const_), py::return_value_policy::take_ownership)
        .def("has_edge", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::has_edge, py::const_))
        .def("has_edge", py::overload_cast<int, int>(&BayesianNetworkBase::has_edge, py::const_))
        .def("has_path", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::has_path, py::const_))
        .def("has_path", py::overload_cast<int, int>(&BayesianNetworkBase::has_path, py::const_))
        .def("add_edge", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::add_edge))
        .def("add_edge", py::overload_cast<int, int>(&BayesianNetworkBase::add_edge))
        .def("can_add_edge", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::can_add_edge, py::const_))
        .def("can_add_edge", py::overload_cast<int, int>(&BayesianNetworkBase::can_add_edge, py::const_))
        .def("can_flip_edge", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::can_flip_edge))
        .def("can_flip_edge", py::overload_cast<int, int>(&BayesianNetworkBase::can_flip_edge))
        .def("remove_edge", py::overload_cast<const std::string&, const std::string&>(&BayesianNetworkBase::remove_edge))
        .def("remove_edge", py::overload_cast<int, int>(&BayesianNetworkBase::remove_edge))
        .def("fit", &BayesianNetworkBase::fit)
        .def("logpdf", &BayesianNetworkBase::logpdf, py::return_value_policy::take_ownership)
        .def("slogpdf", &BayesianNetworkBase::slogpdf);

    register_BayesianNetwork<GaussianNetwork<>>(models, "GaussianNetwork");
    register_BayesianNetwork<GaussianNetwork<AdjListDag>>(models, "GaussianNetwork_L");
    
    auto spbn = register_BayesianNetwork<SemiparametricBN<>>(models, "SemiparametricBN");

    spbn.def(py::init<const std::vector<std::string>&, FactorTypeVector&>())
        .def(py::init<const ArcVector&, FactorTypeVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcVector&, FactorTypeVector&>())
        .def("node_type", py::overload_cast<const std::string&>(&SemiparametricBN<>::node_type, py::const_))
        .def("node_type", py::overload_cast<int>(&SemiparametricBN<>::node_type, py::const_))
        .def("set_node_type", py::overload_cast<const std::string&, FactorType>(&SemiparametricBN<>::set_node_type))
        .def("set_node_type", py::overload_cast<int, FactorType>(&SemiparametricBN<>::set_node_type));
    
    auto spbn_l = register_BayesianNetwork<SemiparametricBN<AdjListDag>>(models, "SemiparametricBN_L");

    spbn_l.def(py::init<const std::vector<std::string>&, FactorTypeVector&>())
        .def(py::init<const ArcVector&, FactorTypeVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcVector&, FactorTypeVector&>())
        .def("node_type", py::overload_cast<const std::string&>(&SemiparametricBN<AdjListDag>::node_type, py::const_))
        .def("node_type", py::overload_cast<int>(&SemiparametricBN<AdjListDag>::node_type, py::const_))
        .def("set_node_type", py::overload_cast<const std::string&, FactorType>(&SemiparametricBN<AdjListDag>::set_node_type))
        .def("set_node_type", py::overload_cast<int, FactorType>(&SemiparametricBN<AdjListDag>::set_node_type));
      
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

    register_ArcOperators(operators);

    py::class_<ChangeNodeType, Operator, std::shared_ptr<ChangeNodeType>>(operators, "ChangeNodeType")
        .def(py::init<std::string, FactorType, double>())
        .def_property_readonly("node", &ChangeNodeType::node)
        .def_property_readonly("node_type", &ChangeNodeType::node_type)
        .def("apply", &ChangeNodeType::apply)
        .def("opposite", &ChangeNodeType::opposite);


    register_OperatorTabuSet(operators);

    register_OperatorSet<OperatorSet,
                            GaussianNetwork<>, 
                            GaussianNetwork<AdjListDag>, 
                            SemiparametricBN<>,
                            SemiparametricBN<AdjListDag>>(operators, "OperatorSet");
    register_ArcOperatorSet<BIC,
                            GaussianNetwork<>,
                            GaussianNetwork<AdjListDag>, 
                            SemiparametricBN<>,
                            SemiparametricBN<AdjListDag>>(operators, "BIC");
    register_ArcOperatorSet<CVLikelihood,
                            GaussianNetwork<>,
                            GaussianNetwork<AdjListDag>, 
                            SemiparametricBN<>,
                            SemiparametricBN<AdjListDag>>(operators, "CVLikelihood");
    register_ChangeNodeTypeSet<CVLikelihood,
                            SemiparametricBN<>,
                            SemiparametricBN<AdjListDag>>(operators, "CVLikelihood");
    
    register_OperatorPool<BIC,
                            GaussianNetwork<>,
                            GaussianNetwork<AdjListDag>>(operators, "BIC");
        
    register_OperatorPool<CVLikelihood,
                            GaussianNetwork<>,
                            GaussianNetwork<AdjListDag>,
                            SemiparametricBN<>,
                            SemiparametricBN<AdjListDag>>(operators, "CVLikelihood"); 


    py::class_<OperatorSetType>(operators, "OperatorSetType")
        .def_property_readonly_static("ARCS", [](const py::object&) { 
            return OperatorSetType(OperatorSetType::ARCS);
        })
        .def_property_readonly_static("NODE_TYPE", [](const py::object&) { 
            return OperatorSetType(OperatorSetType::NODE_TYPE);
        })
        .def(py::self == py::self)
        .def(py::self != py::self);

    auto algorithms = learning.def_submodule("algorithms", "Learning algorithms");

    algorithms.def("hc", &learning::algorithms::hc, "Hill climbing estimate");
    // py::class_<GreedyHillClimbing>(algorithms, "GreedyHillClimbing")
    //         .def(py::init<>())
    //         .def("estimate")
}