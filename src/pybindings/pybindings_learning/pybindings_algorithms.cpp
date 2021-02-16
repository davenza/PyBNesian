#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <learning/operators/operators.hpp>
#include <learning/algorithms/callbacks/callback.hpp>
#include <learning/algorithms/callbacks/save_model.hpp>
#include <learning/algorithms/hillclimbing.hpp>
#include <learning/algorithms/constraint.hpp>
#include <learning/algorithms/pc.hpp>
#include <learning/algorithms/mmpc.hpp>
#include <learning/algorithms/mmhc.hpp>
#include <learning/algorithms/dmmhc.hpp>

namespace py = pybind11;

using learning::operators::OperatorPool;
using learning::algorithms::callbacks::Callback, learning::algorithms::callbacks::SaveModel;
using learning::algorithms::GreedyHillClimbing, learning::algorithms::PC, learning::algorithms::MeekRules,
      learning::algorithms::MMPC, learning::algorithms::MMHC;

using learning::algorithms::DMMHC;

void pybindings_algorithms_callbacks(py::module& root) {
    auto callbacks = root.def_submodule("callbacks", "Callbacks for GreedyHillClimbing");

    py::class_<Callback, std::shared_ptr<Callback>>(callbacks, "Callback")
        .def("call", &Callback::call);

    py::class_<SaveModel, Callback, std::shared_ptr<SaveModel>>(callbacks, "SaveModel")
        .def(py::init<const std::string&>(),
            py::arg("folder_name"))
        .def("call", &SaveModel::call);
}


void pybindings_algorithms(py::module& root) {
    auto algorithms = root.def_submodule("algorithms", "Learning algorithms");

    pybindings_algorithms_callbacks(algorithms);

    algorithms.def("hc", &learning::algorithms::hc, "Hill climbing estimate",
                py::arg("df"),
                py::arg("start") = nullptr,
                py::arg("bn_type") = "gbn",
                py::arg("score") = std::nullopt,
                py::arg("operators") = std::nullopt,
                py::arg("arc_blacklist") = ArcStringVector(),
                py::arg("arc_whitelist") = ArcStringVector(),
                py::arg("type_whitelist") = FactorStringTypeVector(),
                py::arg("callback") = nullptr,
                py::arg("max_indegree") = 0,
                py::arg("max_iters") = std::numeric_limits<int>::max(),
                py::arg("epsilon") = 0,
                py::arg("patience") = 0,
                py::arg("seed") = std::nullopt,
                py::arg("num_folds") = 10,
                py::arg("test_holdout_ratio") = 0.2,
                py::arg("verbose") = 0);

    py::class_<GreedyHillClimbing> (algorithms, "GreedyHillClimbing")
        .def(py::init<>())
        .def("estimate", py::overload_cast<OperatorSet&,
                                            Score&,
                                            const ConditionalBayesianNetworkBase&,
                                            const ArcStringVector&,
                                            const ArcStringVector&,
                                            const FactorStringTypeVector&,
                                            const Callback*,
                                            int,
                                            int,
                                            double,
                                            int,
                                            int>(&GreedyHillClimbing::estimate),
                py::arg("op_set"),
                py::arg("score"),
                py::arg("start"),
                py::arg("arc_blacklist") = ArcStringVector(),
                py::arg("arc_whitelist") = ArcStringVector(),
                py::arg("type_whitelist") = FactorStringTypeVector(),
                py::arg("callback") = nullptr,
                py::arg("max_indegree") = 0,
                py::arg("max_iters") = std::numeric_limits<int>::max(),
                py::arg("epsilon") = 0,
                py::arg("patience") = 0,
                py::arg("verbose") = 0)
        .def("estimate", py::overload_cast<OperatorSet&,
                                            Score&,
                                            const BayesianNetworkBase&,
                                            const ArcStringVector&,
                                            const ArcStringVector&,
                                            const FactorStringTypeVector&,
                                            const Callback*,
                                            int,
                                            int,
                                            double,
                                            int,
                                            int>(&GreedyHillClimbing::estimate),
                py::arg("op_set"),
                py::arg("score"),
                py::arg("start"),
                py::arg("arc_blacklist") = ArcStringVector(),
                py::arg("arc_whitelist") = ArcStringVector(),
                py::arg("type_whitelist") = FactorStringTypeVector(),
                py::arg("callback") = nullptr,
                py::arg("max_indegree") = 0,
                py::arg("max_iters") = std::numeric_limits<int>::max(),
                py::arg("epsilon") = 0,
                py::arg("patience") = 0,
                py::arg("verbose") = 0);

    py::class_<PC>(algorithms, "PC")
        .def(py::init<>())
        .def("estimate", &PC::estimate, 
            py::arg("hypot_test"),
            py::arg("nodes") = std::vector<std::string>(),
            py::arg("arc_blacklist") = ArcStringVector(),
            py::arg("arc_whitelist") = ArcStringVector(),
            py::arg("edge_blacklist") = EdgeStringVector(),
            py::arg("edge_whitelist") = EdgeStringVector(),
            py::arg("alpha") = 0.05,
            py::arg("use_sepsets") = false,
            py::arg("ambiguous_threshold") = 0.5,
            py::arg("allow_bidirected") = true,
            py::arg("verbose") = 0)
        .def("estimate_conditional", &PC::estimate_conditional,
            py::arg("hypot_test"),
            py::arg("nodes"),
            py::arg("interface_nodes") = std::vector<std::string>(),
            py::arg("arc_blacklist") = ArcStringVector(),
            py::arg("arc_whitelist") = ArcStringVector(),
            py::arg("edge_blacklist") = EdgeStringVector(),
            py::arg("edge_whitelist") = EdgeStringVector(),
            py::arg("alpha") = 0.05,
            py::arg("use_sepsets") = false,
            py::arg("ambiguous_threshold") = 0.5,
            py::arg("allow_bidirected") = true,
            py::arg("verbose") = 0);

    py::class_<MeekRules>(algorithms, "MeekRules")
        .def_static("rule1", [](PartiallyDirectedGraph& graph) {
            return MeekRules::rule1(graph);
        })
        .def_static("rule1", [](ConditionalPartiallyDirectedGraph& graph) {
            return MeekRules::rule1(graph);
        })
        .def_static("rule2", [](PartiallyDirectedGraph& graph) {
            return MeekRules::rule2(graph);
        })
        .def_static("rule2", [](ConditionalPartiallyDirectedGraph& graph) {
            return MeekRules::rule2(graph);
        })
        .def_static("rule3", [](PartiallyDirectedGraph& graph) {
            return MeekRules::rule3(graph);
        })
        .def_static("rule3", [](ConditionalPartiallyDirectedGraph& graph) {
            return MeekRules::rule3(graph);
        });
    
    py::class_<MMPC>(algorithms, "MMPC")
        .def(py::init<>())
        .def("estimate", &MMPC::estimate, 
            py::arg("hypot_test"),
            py::arg("nodes") = std::vector<std::string>(),
            py::arg("arc_blacklist") = ArcStringVector(),
            py::arg("arc_whitelist") = ArcStringVector(),
            py::arg("edge_blacklist") = EdgeStringVector(),
            py::arg("edge_whitelist") = EdgeStringVector(),
            py::arg("alpha") = 0.05,
            py::arg("ambiguous_threshold") = 0.5,
            py::arg("allow_bidirected") = true,
            py::arg("verbose") = 0)
        .def("estimate_conditional", &MMPC::estimate_conditional, 
            py::arg("hypot_test"),
            py::arg("nodes"),
            py::arg("interface_nodes") = std::vector<std::string>(),
            py::arg("arc_blacklist") = ArcStringVector(),
            py::arg("arc_whitelist") = ArcStringVector(),
            py::arg("edge_blacklist") = EdgeStringVector(),
            py::arg("edge_whitelist") = EdgeStringVector(),
            py::arg("alpha") = 0.05,
            py::arg("ambiguous_threshold") = 0.5,
            py::arg("allow_bidirected") = true,
            py::arg("verbose") = 0);

    py::class_<MMHC>(algorithms, "MMHC")
        .def(py::init<>())
        .def("estimate", &MMHC::estimate, 
            py::arg("hypot_test"),
            py::arg("op_set"),
            py::arg("score"),
            py::arg("nodes") = std::vector<std::string>(),
            py::arg("bn_type") = "gbn",
            py::arg("arc_blacklist") = ArcStringVector(),
            py::arg("arc_whitelist") = ArcStringVector(),
            py::arg("edge_blacklist") = EdgeStringVector(),
            py::arg("edge_whitelist") = EdgeStringVector(),
            py::arg("type_whitelist") = FactorStringTypeVector(),
            py::arg("max_indegree") = 0,
            py::arg("max_iters") = std::numeric_limits<int>::max(),
            py::arg("epsilon") = 0,
            py::arg("patience") = 0,
            py::arg("alpha") = 0.05,
            py::arg("verbose") = 0)
        .def("estimate_conditional", &MMHC::estimate_conditional,
            py::arg("hypot_test"),
            py::arg("op_set"),
            py::arg("score"),
            py::arg("nodes") = std::vector<std::string>(),
            py::arg("interface_nodes") = std::vector<std::string>(),
            py::arg("bn_type") = "gbn",
            py::arg("arc_blacklist") = ArcStringVector(),
            py::arg("arc_whitelist") = ArcStringVector(),
            py::arg("edge_blacklist") = EdgeStringVector(),
            py::arg("edge_whitelist") = EdgeStringVector(),
            py::arg("type_whitelist") = FactorStringTypeVector(),
            py::arg("max_indegree") = 0,
            py::arg("max_iters") = std::numeric_limits<int>::max(),
            py::arg("epsilon") = 0,
            py::arg("patience") = 0,
            py::arg("alpha") = 0.05,
            py::arg("verbose") = 0);

    py::class_<DMMHC>(algorithms, "DMMHC")
        .def(py::init<>())
        .def("estimate", &DMMHC::estimate, 
            py::arg("hypot_test"),
            py::arg("op_set"),
            py::arg("score"),
            py::arg("variables") = std::vector<std::string>(),
            py::arg("bn_type") = "gbn",
            py::arg("markovian_order") = 1,
            py::arg("max_indegree") = 0,
            py::arg("max_iters") = std::numeric_limits<int>::max(),
            py::arg("epsilon") = 0,
            py::arg("patience") = 0,
            py::arg("alpha") = 0.05,
            py::arg("verbose") = 0);

}