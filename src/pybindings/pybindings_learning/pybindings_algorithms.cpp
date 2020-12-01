#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <learning/operators/operators.hpp>
#include <learning/algorithms/hillclimbing.hpp>
#include <learning/algorithms/constraint.hpp>
#include <learning/algorithms/pc.hpp>
#include <learning/algorithms/mmpc.hpp>

namespace py = pybind11;

using learning::operators::OperatorPool;
using learning::algorithms::GreedyHillClimbing, learning::algorithms::PC, learning::algorithms::MeekRules,
      learning::algorithms::MMPC;

template<typename Model, typename... Models>
py::class_<GreedyHillClimbing> register_GreedyHillClimbing(py::module& m) {
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
                            ArcStringVector& arc_blacklist,
                            ArcStringVector& arc_whitelist,
                            int max_indegree,
                            int max_iters, 
                            double epsilon,
                            int verbose) {
            return self.estimate(df, pool, start, arc_blacklist, arc_whitelist, max_indegree, max_iters, epsilon, verbose);
        },  py::arg("df"),
            py::arg("pool"),
            py::arg("start"),
            py::arg("arc_blacklist") = ArcStringVector(),
            py::arg("arc_whitelist") = ArcStringVector(),
            py::arg("max_indegree") = 0,
            py::arg("max_iters") = std::numeric_limits<int>::max(),
            py::arg("epsilon") = 0,
            py::arg("verbose") = 0
    );

    hc.def("estimate_validation", [](GreedyHillClimbing& self, 
                                     const DataFrame& df, 
                                     OperatorPool& pool, 
                                     Score& validation_score,
                                     const Model& start,
                                     ArcStringVector& arc_blacklist,
                                     ArcStringVector& arc_whitelist,
                                     FactorStringTypeVector& type_whitelist,
                                     int max_indegree,
                                     int max_iters,
                                     double epsilon,
                                     int patience,
                                     int verbose) {
            return self.estimate_validation(df, pool, validation_score, start, arc_blacklist, arc_whitelist, 
                                            type_whitelist, max_indegree, max_iters, epsilon, patience, verbose);
        },  py::arg("df"),
            py::arg("pool"),
            py::arg("validation_score"),
            py::arg("start"),
            py::arg("arc_blacklist") = ArcStringVector(),
            py::arg("arc_whitelist") = ArcStringVector(),
            py::arg("type_whitelist") = FactorStringTypeVector(),
            py::arg("max_indegree") = 0,
            py::arg("max_iters") = std::numeric_limits<int>::max(),
            py::arg("epsilon") = 0,
            py::arg("patience") = 0,
            py::arg("verbose") = 0
    );

    return hc;
}



void pybindings_algorithms(py::module& root) {
    auto algorithms = root.def_submodule("algorithms", "Learning algorithms");

    algorithms.def("hc", &learning::algorithms::hc, "Hill climbing estimate",
                py::arg("df"),
                py::arg("bn_type") = "gbn",
                py::arg("score_type") = "bic",
                py::arg("operators_type") = std::vector<std::string>{"arcs"},
                py::arg("arc_blacklist") = ArcStringVector(),
                py::arg("arc_whitelist") = ArcStringVector(),
                py::arg("type_whitelist") = FactorStringTypeVector(),
                py::arg("max_indegree") = 0,
                py::arg("max_iters") = std::numeric_limits<int>::max(),
                py::arg("epsilon") = 0,
                py::arg("patience") = 0,
                py::arg("verbose") = 0);

    register_GreedyHillClimbing<GaussianNetwork, SemiparametricBN>(algorithms);

    py::class_<PC>(algorithms, "PC")
        .def(py::init<>())
        .def("estimate", &PC::estimate, 
            py::arg("hypot_test"),
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
        .def_static("rule1", &MeekRules::rule1)
        .def_static("rule2", &MeekRules::rule2)
        .def_static("rule3", &MeekRules::rule3);
    
    py::class_<MMPC>(algorithms, "MMPC")
        .def(py::init<>())
        .def("estimate", &MMPC::estimate, 
            py::arg("hypot_test"),
            py::arg("arc_blacklist") = ArcStringVector(),
            py::arg("arc_whitelist") = ArcStringVector(),
            py::arg("edge_blacklist") = EdgeStringVector(),
            py::arg("edge_whitelist") = EdgeStringVector(),
            py::arg("alpha") = 0.05,
            py::arg("ambiguous_threshold") = 0.5,
            py::arg("allow_bidirected") = true,
            py::arg("verbose") = 0);


}