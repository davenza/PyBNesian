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
using learning::scores::Score, learning::scores::BIC, learning::scores::CVLikelihood, 
        learning::scores::HoldoutLikelihood;
using learning::operators::AddArc, learning::operators::RemoveArc, learning::operators::FlipArc,
      learning::operators::ChangeNodeType, learning::operators::OperatorTabuSet, 
      learning::operators::OperatorSetType, learning::operators::OperatorSet, 
      learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet,
      learning::operators::OperatorPool;
using learning::algorithms::GreedyHillClimbing;

// using learning::operators::ArcOperatorSet_constructor, learning::operators::ChangeNodeTypeSet_constructor, 
//       learning::operators::OperatorPool_constructor;

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
        .def("edges", &BaseClass::edges, py::return_value_policy::take_ownership)
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
        .def("opposite", &Operator::opposite, py::return_value_policy::take_ownership)
        .def("__eq__", [](const Operator& self, const Operator& other) {
            return self == other;
        }, py::is_operator())
        .def("__ne__", [](const Operator& self, const Operator& other) {
            return self != other;
        }, py::is_operator());

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

void register_OperatorTabuSet(py::module& m) {

    // TODO: Implement copy operation.
    py::class_<OperatorTabuSet>(m, "OperatorTabuSet")
        .def(py::init<>())
        .def(py::init<const OperatorTabuSet&>())
        .def("insert", py::overload_cast<std::shared_ptr<Operator>>(&OperatorTabuSet::insert))
        .def("contains", py::overload_cast<std::shared_ptr<Operator>&>(&OperatorTabuSet::contains, py::const_))
        .def("clear", &OperatorTabuSet::clear)
        .def("empty", &OperatorTabuSet::empty);
}

template<typename Model, typename... Models>
py::class_<OperatorSet, std::shared_ptr<OperatorSet>> register_OperatorSet(py::module& m) {

    auto op_set = [&m]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<OperatorSet, std::shared_ptr<OperatorSet>> op_set(m, "OperatorSet");
            // op_set.def(py::init<>());
            return op_set;
        } else {
            return register_OperatorSet<Models...>(m);
        }
    }();

    op_set.def("cache_scores", [](OperatorSet& self, Model& model) {
        self.cache_scores(model);
    });
    op_set.def("find_max", [](OperatorSet& self, Model& model) {
        return self.find_max(model);
    });
    op_set.def("find_max", [](OperatorSet& self, Model& model, OperatorTabuSet& tabu) {
        return self.find_max(model, tabu);
    });
    op_set.def("update_scores", [](OperatorSet& self, Model& model, Operator& op) {
        self.update_scores(model, op);
    });

    op_set.def("set_arc_blacklist", &OperatorSet::set_arc_blacklist);
    op_set.def("set_arc_whitelist", &OperatorSet::set_arc_whitelist);
    op_set.def("set_max_indegree", &OperatorSet::set_max_indegree);
    op_set.def("set_type_whitelist", &OperatorSet::set_type_whitelist);

    return op_set;
}

template<typename DerivedOpSet, typename Model, typename... Models>
py::class_<DerivedOpSet, OperatorSet, std::shared_ptr<DerivedOpSet>> register_DerivedOperatorSet(py::module& m, const char* name) {
    auto op_set = [&m, name]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<DerivedOpSet, OperatorSet, std::shared_ptr<DerivedOpSet>> op_set(m, name);
            return op_set;
        } else {
            return register_DerivedOperatorSet<DerivedOpSet, Models...>(m, name);
        }
    }();

    op_set.def("cache_scores", [](DerivedOpSet& self, Model& model) {
        self.cache_scores(model);
    })
    .def("find_max", [](DerivedOpSet& self, Model& model) {
        return self.find_max(model);
    })
    .def("find_max", [](DerivedOpSet& self, Model& model, OperatorTabuSet& tabu) {
        return self.find_max(model, tabu);
    })
    .def("update_scores", [](DerivedOpSet& self, Model& model, Operator& op) {
        self.update_scores(model, op);
    });

    return op_set;
}


template<typename Model, typename... Models>
py::class_<OperatorPool> register_OperatorPool(py::module& m) {
    auto pool = [&m]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<OperatorPool> pool(m, "OperatorPool");
            return pool;
        } else {
            return register_OperatorPool<Models...>(m);
        }
    }();

    pool.def(py::init<Model&, std::shared_ptr<Score>&, std::vector<std::shared_ptr<OperatorSet>>>());
    pool.def("cache_scores", [](OperatorPool& self, Model& model) {
        self.cache_scores(model);
    });
    pool.def("find_max", [](OperatorPool& self, Model& model) {
        return self.find_max(model);
    });
    pool.def("find_max", [](OperatorPool& self, Model& model, OperatorTabuSet& tabu) {
        return self.find_max(model, tabu);
    });
    pool.def("update_scores", [](OperatorPool& self, Model& model, Operator& op) {
        self.update_scores(model, op);
    });

    return pool;
}

template<typename Model, typename... Models>
py::class_<OperatorPool> register_GreedyHillClimbing(py::module& m) {
    auto hc = [&m]() {
        if constexpr (sizeof...(Models) == 0) {
            py::class_<GreedyHillClimbing> hc(m, "GreedyHillClimbing");
            hc.def(py::init<>());
            return hc;
        } else {
            return register_GreedyHillClimbing<Models...>(m);
        }
    }();

    hc.def("estimate", [](GreedyHillClimbing& self, 
                            const DataFrame& df, 
                            OperatorPool& pool,
                            const Model& start,
                            ArcVector& arc_blacklist,
                            ArcVector& arc_whitelist,
                            int max_indegree,
                            int max_iters, 
                            double epsilon) {
            return self.estimate(df, pool, start, arc_blacklist, arc_whitelist, max_indegree, max_iters, epsilon);
        },  py::arg("df"),
            py::arg("pool"),
            py::arg("start"),
            py::arg("arc_blacklist") = ArcVector(),
            py::arg("arc_whitelist") = ArcVector(),
            py::arg("max_indegree") = 0,
            py::arg("max_iters") = std::numeric_limits<int>::max(),
            py::arg("epsilon") = 0
    );
    hc.def("estimate_validation", [](GreedyHillClimbing& self, 
                                     const DataFrame& df, 
                                     OperatorPool& pool, 
                                     Score& validation_score,
                                     const Model& start,
                                     ArcVector& arc_blacklist,
                                     ArcVector& arc_whitelist,
                                     FactorTypeVector& type_whitelist,
                                     int max_indegree,
                                     int max_iters,
                                     double epsilon,
                                     int patience) {
            return self.estimate_validation(df, pool, validation_score, start, arc_blacklist, arc_whitelist, 
                                            type_whitelist, max_indegree, max_iters, epsilon, patience);
        },  py::arg("df"),
            py::arg("pool"),
            py::arg("validation_score"),
            py::arg("start"),
            py::arg("arc_blacklist") = ArcVector(),
            py::arg("arc_whitelist") = ArcVector(),
            py::arg("type_whitelist") = FactorTypeVector(),
            py::arg("max_indegree") = 0,
            py::arg("max_iters") = std::numeric_limits<int>::max(),
            py::arg("epsilon") = 0,
            py::arg("patience") = 0
    );

    return hc;
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
        .def("edges", &BayesianNetworkBase::edges, py::return_value_policy::take_ownership)
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

    register_Score<GaussianNetwork<>, 
                   GaussianNetwork<AdjListDag>, 
                   SemiparametricBN<>, 
                   SemiparametricBN<AdjListDag>>(scores);

    py::class_<BIC, Score, std::shared_ptr<BIC>>(scores, "BIC")
        .def(py::init<const DataFrame&>());

    py::class_<CVLikelihood, Score, std::shared_ptr<CVLikelihood>>(scores, "CVLikelihood")
        .def(py::init<const DataFrame&, int>(),
                py::arg("df"),
                py::arg("k") = 10)
        .def(py::init<const DataFrame&, int, int>(),
                py::arg("df"),
                py::arg("k") = 10,
                py::arg("seed"))
        .def_property_readonly("cv", &CVLikelihood::cv);

    py::class_<HoldoutLikelihood, Score, std::shared_ptr<HoldoutLikelihood>>(scores, "HoldoutLikelihood")
        .def(py::init<const DataFrame&, double>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2)
        .def(py::init<const DataFrame&, double, int>(),
                py::arg("df"),
                py::arg("test_ratio") = 0.2,
                py::arg("seed"))
        .def_property_readonly("holdout", &HoldoutLikelihood::holdout)
        .def("training_data", &HoldoutLikelihood::training_data, py::return_value_policy::reference_internal)
        .def("test_data", &HoldoutLikelihood::test_data, py::return_value_policy::reference_internal);

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

    register_OperatorSet<GaussianNetwork<>, 
                         GaussianNetwork<AdjListDag>, 
                         SemiparametricBN<>,
                         SemiparametricBN<AdjListDag>>(operators);
    auto arc_set = register_DerivedOperatorSet<ArcOperatorSet,
                                    GaussianNetwork<>,
                                    GaussianNetwork<AdjListDag>, 
                                    SemiparametricBN<>,
                                    SemiparametricBN<AdjListDag>>(operators, "ArcOperatorSet");
    arc_set.def(py::init<std::shared_ptr<Score>&, const ArcVector&, const ArcVector&, int>(),
                py::arg("score"),
                py::arg("blacklist") = ArcVector(),
                py::arg("whitelist") = ArcVector(),
                py::arg("max_indegree") = 0);


    auto nodetype = register_DerivedOperatorSet<ChangeNodeTypeSet,
                                    SemiparametricBN<>,
                                    SemiparametricBN<AdjListDag>>(operators, "ChangeNodeTypeSet");
    nodetype.def(py::init<std::shared_ptr<Score>&, FactorTypeVector>(), 
                 py::arg("score"),
                 py::arg("type_whitelist") = FactorTypeVector());

    
    register_OperatorPool<GaussianNetwork<>,
                          GaussianNetwork<AdjListDag>,
                          SemiparametricBN<>,
                          SemiparametricBN<AdjListDag>>(operators);


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

    algorithms.def("hc", &learning::algorithms::hc, "Hill climbing estimate",
                py::arg("df"),
                py::arg("bn_type") = "gbn",
                py::arg("score_type") = "bic",
                py::arg("operators_type") = std::vector<std::string>{"arcs"},
                py::arg("arc_blacklist") = ArcVector(),
                py::arg("arc_whitelist") = ArcVector(),
                py::arg("type_whitelist") = FactorTypeVector(),
                py::arg("max_indegree") = 0,
                py::arg("max_iters") = std::numeric_limits<int>::max(),
                py::arg("epsilon") = 0,
                py::arg("patience") = 0,
                py::arg("dag_type") = "matrix");

    register_GreedyHillClimbing<GaussianNetwork<>,
                                GaussianNetwork<AdjListDag>,
                                SemiparametricBN<>,
                                SemiparametricBN<AdjListDag>>(algorithms);
}