#include <pybind11/pybind11.h>
#include <learning/operators/operators.hpp>
#include <learning/algorithms/hillclimbing.hpp>

namespace py = pybind11;

using learning::operators::OperatorPool;
using learning::algorithms::GreedyHillClimbing;

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



void pybindings_algorithms(py::module& root) {
    auto algorithms = root.def_submodule("algorithms", "Learning algorithms");

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