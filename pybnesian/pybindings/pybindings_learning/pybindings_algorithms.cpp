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

using models::GaussianNetworkType;

using learning::algorithms::GreedyHillClimbing, learning::algorithms::PC, learning::algorithms::MeekRules,
    learning::algorithms::MMPC, learning::algorithms::MMHC;
using learning::algorithms::callbacks::Callback, learning::algorithms::callbacks::SaveModel;
using learning::operators::OperatorPool;

using learning::algorithms::DMMHC;

class PyCallback : public Callback {
public:
    using Callback::Callback;

    void call(BayesianNetworkBase& model, Operator* new_operator, Score& score, int num_iter) const override {
        PYBIND11_OVERRIDE_PURE(void, Callback, call, model.shared_from_this(), new_operator, &score, num_iter);
    }
};

void pybindings_algorithms_callbacks(py::module& root) {
    py::class_<Callback, PyCallback, std::shared_ptr<Callback>>(root, "Callback", R"doc(
A :class:`Callback` object is called after each iteration of a
:class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`.
)doc")
        .def(py::init<>(), R"doc(
Initializes a :class:`Callback`.
)doc")
        .def("call",
             &Callback::call,
             py::arg("model"),
             py::arg("operator"),
             py::arg("score"),
             py::arg("iteration"),
             R"doc(
This method is called after each iteration of
:class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`.

:param model: The model in the current ``iteration`` of the
              :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`.
:param operator: The last operator applied to the ``model``. It is ``None`` at the start and at the end of the
                 algorithm.
:param score: The score used in the :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`.
:param iteration: Iteration number of the
                  :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`. It is 0 at the start.
)doc");

    py::class_<SaveModel, Callback, std::shared_ptr<SaveModel>>(root, "SaveModel", R"doc(
Saves the model on each iteration of :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`
using :func:`BayesianNetworkBase.save() <pybnesian.BayesianNetworkBase.save>`. Each model is named after the
iteration number.
)doc")
        .def(py::init<const std::string&>(), py::arg("folder_name"), R"doc(
Initializes a :class:`SaveModel`. It saves all the models in the folder ``folder_name``.

:param folder_name: Name of the folder where the models will be saved.
)doc");
}

void pybindings_algorithms(py::module& root) {
    pybindings_algorithms_callbacks(root);

    root.def("hc",
             &learning::algorithms::hc,
             py::arg("df"),
             py::arg("bn_type") = nullptr,
             py::arg("start") = nullptr,
             py::arg("score") = std::nullopt,
             py::arg("operators") = std::nullopt,
             py::arg("arc_blacklist") = ArcStringVector(),
             py::arg("arc_whitelist") = ArcStringVector(),
             py::arg("type_blacklist") = FactorTypeVector(),
             py::arg("type_whitelist") = FactorTypeVector(),
             py::arg("callback") = nullptr,
             py::arg("max_indegree") = 0,
             py::arg("max_iters") = std::numeric_limits<int>::max(),
             py::arg("epsilon") = 0,
             py::arg("patience") = 0,
             py::arg("seed") = std::nullopt,
             py::arg("num_folds") = 10,
             py::arg("test_holdout_ratio") = 0.2,
             py::arg("verbose") = 0,
             R"doc(
Executes a greedy hill-climbing algorithm. This calls :func:`GreedyHillClimbing.estimate`.

:param df: DataFrame used to learn a Bayesian network model.
:param bn_type: :class:`BayesianNetworkType` of the returned model. If ``start`` is given, ``bn_type`` is ignored.
:param start: Initial structure of the :class:`GreedyHillClimbing`. If ``None``, a new Bayesian network model is
              created.
:param score: A string representing the score used to drive the search. The possible options are:
              "bic" for :class:`BIC <pybnesian.BIC>`, "bge" for
              :class:`BGe <pybnesian.BGe>`, "cv-lik" for
              :class:`CVLikelihood <pybnesian.CVLikelihood>`, "holdout-lik" for
              :class:`HoldoutLikelihood <pybnesian.HoldoutLikelihood>`, "validated-lik for
              :class:`ValidatedLikelihood <pybnesian.ValidatedLikelihood>`.
:param operators: Set of operators in the search process.
:param arc_blacklist: List of arcs blacklist (forbidden arcs).
:param arc_whitelist: List of arcs whitelist (forced arcs).
:param type_blacklist: List of type blacklist (forbidden :class:`FactorType <pybnesian.FactorType>`).
:param type_whitelist: List of type whitelist (forced :class:`FactorType <pybnesian.FactorType>`).
:param callback: Callback object that is called after each iteration.
:param max_indegree: Maximum indegree allowed in the graph.
:param max_iters: Maximum number of search iterations
:param epsilon: Minimum delta score allowed for each operator. If the new operator is less than epsilon, the search
                process is stopped.
:param patience: The patience parameter (only used with
                :class:`ValidatedScore <pybnesian.ValidatedScore>`). See `patience`_.
:param seed: Seed parameter of the score (if neeeded).
:param num_folds: Number of folds for the :class:`CVLikelihood <pybnesian.CVLikelihood>` and
                  :class:`ValidatedLikelihood <pybnesian.ValidatedLikelihood>` scores.
:param test_holdout_ratio: Parameter for the :class:`HoldoutLikelihood <pybnesian.HoldoutLikelihood>`
                           and :class:`ValidatedLikelihood <pybnesian.ValidatedLikelihood>` scores.
:param verbose: If True the progress will be displayed, otherwise nothing will be displayed.
:returns: The estimated Bayesian network structure.
)doc");

    py::class_<GreedyHillClimbing> hc(root, "GreedyHillClimbing", R"doc(
This class implements a greedy hill-climbing algorithm. It finds the best structure applying small local changes
iteratively. The best operator is found using a delta score.

.. _patience:

Patience parameter:

When the score is a :class:`ValidatedScore <pybnesian.ValidatedScore>`, a tabu set is used to improve the
exploration during the search process if the score does not improve. This is because it is allowed to continue
the search process even if the training delta score of the
:class:`ValidatedScore <pybnesian.ValidatedScore>` is negative. The existence of the validation delta
score in the :class:`ValidatedScore <pybnesian.ValidatedScore>` can help to control the uncertainty of
the training score (the training delta score can be negative because it is a bad operator or because there is
uncertainty in the data). Thus, only if both the training and validation delta scores are negative for ``patience``
iterations, the search is stopped and the best found model is returned.
)doc");
    hc.def(py::init<>(), R"doc(
Initializes a :class:`GreedyHillClimbing`.
)doc");

    {
        py::options options;
        options.disable_function_signatures();
        hc.def("estimate",
               py::overload_cast<OperatorSet&,
                                 Score&,
                                 const ConditionalBayesianNetworkBase&,
                                 const ArcStringVector&,
                                 const ArcStringVector&,
                                 const FactorTypeVector&,
                                 const FactorTypeVector&,
                                 const std::shared_ptr<Callback>,
                                 int,
                                 int,
                                 double,
                                 int,
                                 int>(&GreedyHillClimbing::estimate<ConditionalBayesianNetworkBase>),
               py::arg("operators"),
               py::arg("score"),
               py::arg("start"),
               py::arg("arc_blacklist") = ArcStringVector(),
               py::arg("arc_whitelist") = ArcStringVector(),
               py::arg("type_blacklist") = FactorTypeVector(),
               py::arg("type_whitelist") = FactorTypeVector(),
               py::arg("callback") = nullptr,
               py::arg("max_indegree") = 0,
               py::arg("max_iters") = std::numeric_limits<int>::max(),
               py::arg("epsilon") = 0,
               py::arg("patience") = 0,
               py::arg("verbose") = 0)
            .def("estimate",
                 py::overload_cast<OperatorSet&,
                                   Score&,
                                   const BayesianNetworkBase&,
                                   const ArcStringVector&,
                                   const ArcStringVector&,
                                   const FactorTypeVector&,
                                   const FactorTypeVector&,
                                   const std::shared_ptr<Callback>,
                                   int,
                                   int,
                                   double,
                                   int,
                                   int>(&GreedyHillClimbing::estimate<BayesianNetworkBase>),
                 py::arg("operators"),
                 py::arg("score"),
                 py::arg("start"),
                 py::arg("arc_blacklist") = ArcStringVector(),
                 py::arg("arc_whitelist") = ArcStringVector(),
                 py::arg("type_blacklist") = FactorTypeVector(),
                 py::arg("type_whitelist") = FactorTypeVector(),
                 py::arg("callback") = nullptr,
                 py::arg("max_indegree") = 0,
                 py::arg("max_iters") = std::numeric_limits<int>::max(),
                 py::arg("epsilon") = 0,
                 py::arg("patience") = 0,
                 py::arg("verbose") = 0,
                 R"doc(
estimate(self: pybnesian.GreedyHillClimbing, operators: pybnesian.OperatorSet, score: pybnesian.Score, start: BayesianNetworkBase or ConditionalBayesianNetworkBase, arc_blacklist: List[Tuple[str, str]] = [], arc_whitelist: List[Tuple[str, str]] = [], type_blacklist: List[Tuple[str, pybnesian.FactorType]] = [], type_whitelist: List[Tuple[str, pybnesian.FactorType]] = [], callback: pybnesian.Callback = None, max_indegree: int = 0, max_iters: int = 2147483647, epsilon: float = 0, patience: int = 0, verbose: int = 0) -> type[start]

Estimates the structure of a Bayesian network. The estimated Bayesian network is of the same type as ``start``. The set
of operators allowed in the search is ``operators``. The delta score of each operator is evaluated using the ``score``.
The initial structure of the algorithm is the model ``start``.

There are many optional parameters that restricts to the learning process.

:param operators: Set of operators in the search process.
:param score: :class:`Score <pybnesian.Score>` that drives the search.
:param start: Initial structure. A :class:`BayesianNetworkBase <pybnesian.BayesianNetworkBase>` or
              :class:`ConditionalBayesianNetworkBase <pybnesian.ConditionalBayesianNetworkBase>`
:param arc_blacklist: List of arcs blacklist (forbidden arcs).
:param arc_whitelist: List of arcs whitelist (forced arcs)
:param type_blacklist: List of type blacklist (forbidden :class:`FactorType <pybnesian.FactorType>`).
:param type_whitelist: List of type whitelist (forced :class:`FactorType <pybnesian.FactorType>`).
:param callback: Callback object that is called after each iteration.
:param max_indegree: Maximum indegree allowed in the graph.
:param max_iters: Maximum number of search iterations
:param epsilon: Minimum delta score allowed for each operator. If the new operator is less than epsilon, the search
                process is stopped.
:param patience: The patience parameter (only used with
                :class:`ValidatedScore <pybnesian.ValidatedScore>`). See `patience`_.
:param verbose: If True the progress will be displayed, otherwise nothing will be displayed.
:returns: The estimated Bayesian network structure of the same type as ``start``.
)doc");
    }

    py::class_<PC>(root, "PC", R"doc(
This class implements the PC learning algorithm. The PC algorithm finds the best partially directed graph that expresses
the conditional independences in the data.

It implements the PC-stable version of [pc-stable]_. This implementation is parametrized to execute the conservative PC
(CPC) or the majority PC (MPC) variant.

This class can return an unconditional partially directed graph (using :func:`PC.estimate`) and a conditional partially
directed graph (using :func:`PC.estimate_conditional`).
)doc")
        .def(py::init<>(), R"doc(
Initializes a :class:`PC`.
)doc")
        .def("estimate",
             &PC::estimate,
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
             py::arg("verbose") = 0,
             R"doc(
Estimates the skeleton (the partially directed graph) using the PC algorithm.

:param hypot_test: The :class:`IndependenceTest <pybnesian.IndependenceTest>` object used to
                   execute the conditional independence tests.
:param nodes: The list of nodes of the returned skeleton. If empty (the default value), the node names are extracted
              from :func:`IndependenceTest.variable_names() <pybnesian.IndependenceTest.variable_names>`.
:param arc_blacklist: List of arcs blacklist (forbidden arcs).
:param arc_whitelist: List of arcs whitelist (forced arcs).
:param edge_blacklist: List of edge blacklist (forbidden edges). This also implicitly applies a double arc
                       blacklist.
:param edge_whitelist: List of edge whitelist (forced edges).
:param alpha: The type I error of each independence test.
:param use_sepsets: If True, it detects the v-structures using the cached sepsets in Algorithm 4.1 of [pc-stable]_.
                    Otherwise, it searches among all the possible sepsets (as in CPC and MPC).
:param ambiguous_threshold: If ``use_sepsets`` is ``False``, the ``ambiguous_threshold`` sets the threshold on the ratio
                            of sepsets needed to declare a v-structure. If ``ambiguous_threshold = 0``, it is equivalent
                            to CPC (the v-structure is detected if no sepset contains the v-node).
                            If ``ambiguous_threshold = 0.5``, it is equivalent to MPC (the v-structure is detected if
                            less than half of the sepsets contain the v-node).
:param allow_bidirected: If True, it allows bi-directed arcs. This ensures that the result of the algorithm is
                         order-independent while applying v-structures (as in LCPC and LMPC in [pc-stable]_). Otherwise,
                         it does not return bi-directed arcs.
:param verbose: If True the progress will be displayed, otherwise nothing will be displayed.
:returns: A :class:`PartiallyDirectedGraph <pybnesian.PartiallyDirectedGraph>` trained by PC that represents
          the conditional independences in ``hypot_test``.
)doc")
        .def("estimate_conditional",
             &PC::estimate_conditional,
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
             py::arg("verbose") = 0,
             R"doc(
Estimates the conditional skeleton (the conditional partially directed graph) using the PC algorithm.

:param hypot_test: The :class:`IndependenceTest <pybnesian.IndependenceTest>` object used to
                   execute the conditional independence tests.
:param nodes: The list of nodes of the returned skeleton.
:param interface_nodes: The list of interface nodes of the returned skeleton.
:param arc_blacklist: List of arcs blacklist (forbidden arcs).
:param arc_whitelist: List of arcs whitelist (forced arcs).
:param edge_blacklist: List of edge blacklist (forbidden edges). This also implicitly applies a double arc
                       blacklist.
:param edge_whitelist: List of edge whitelist (forced edges).
:param alpha: The type I error of each independence test.
:param use_sepsets: If True, it detects the v-structures using the cached sepsets in Algorithm 4.1 of [pc-stable]_.
                    Otherwise, it searches among all the possible sepsets (as in CPC and MPC).
:param ambiguous_threshold: If ``use_sepsets`` is ``False``, the ``ambiguous_threshold`` sets the threshold on the ratio
                            of sepsets needed to declare a v-structure. If ``ambiguous_threshold = 0``, it is equivalent
                            to CPC (the v-structure is detected if no sepset contains the v-node).
                            If ``ambiguous_threshold = 0.5``, it is equivalent to MPC (the v-structure is detected if
                            less than half of the sepsets contain the v-node).
:param allow_bidirected: If True, it allows bi-directed arcs. This ensures that the result of the algorithm is
                         order-independent while applying v-structures (as in LCPC and LMPC in [pc-stable]_). Otherwise,
                         it does not return bi-directed arcs.
:param verbose: If True the progress will be displayed, otherwise nothing will be displayed.
:returns: A :class:`ConditionalPartiallyDirectedGraph <pybnesian.ConditionalPartiallyDirectedGraph>` trained by PC
          that represents the conditional independences in ``hypot_test``.
)doc");

    py::class_<MeekRules> meek(root, "MeekRules", R"doc(
This class implements the Meek rules [meek]_. These rules direct some edges in a partially directed graph to create an
equivalence class of Bayesian networks.
)doc");
    {
        py::options options;
        options.disable_function_signatures();

        meek.def_static(
                "rule1", [](PartiallyDirectedGraph& graph) { return MeekRules::rule1(graph); }, py::arg("graph"))
            .def_static(
                "rule1",
                [](ConditionalPartiallyDirectedGraph& graph) { return MeekRules::rule1(graph); },
                py::arg("graph"),
                R"doc(
rule1(graph: pybnesian.PartiallyDirectedGraph or pybnesian.ConditionalPartiallyDirectedGraph) -> bool

Applies the rule 1 to ``graph``.

:param graph: Graph to apply the rule 1.
:returns: True if the rule changed the graph, False otherwise.
)doc")
            .def_static(
                "rule2", [](PartiallyDirectedGraph& graph) { return MeekRules::rule2(graph); }, py::arg("graph"))
            .def_static(
                "rule2",
                [](ConditionalPartiallyDirectedGraph& graph) { return MeekRules::rule2(graph); },
                py::arg("graph"),
                R"doc(
rule2(graph: pybnesian.PartiallyDirectedGraph or pybnesian.ConditionalPartiallyDirectedGraph) -> bool

Applies the rule 2 to ``graph``.

:param graph: Graph to apply the rule 2.
:returns: True if the rule changed the graph, False otherwise.
)doc")
            .def_static(
                "rule3", [](PartiallyDirectedGraph& graph) { return MeekRules::rule3(graph); }, py::arg("graph"))
            .def_static(
                "rule3",
                [](ConditionalPartiallyDirectedGraph& graph) { return MeekRules::rule3(graph); },
                py::arg("graph"),
                R"doc(
rule3(graph: pybnesian.PartiallyDirectedGraph or pybnesian.ConditionalPartiallyDirectedGraph) -> bool

Applies the rule 3 to ``graph``.

:param graph: Graph to apply the rule 3.
:returns: True if the rule changed the graph, False otherwise.
)doc");
    }

    py::class_<MMPC>(root, "MMPC", R"doc(
This class implements Max-Min Parent Children (MMPC) [mmhc]_. The MMPC algorithm finds the sets of parents and children
of each node using a measure of association. With this estimate, it constructs a skeleton (an undirected graph).
Then, this algorithm searches for v-structures as in :class:`PC`. The final product of this algorithm is a partially
directed graph.

This implementation uses the p-value as a measure of association. A lower p-value is a higher association value and
viceversa.
)doc")
        .def(py::init<>(), R"doc(
Initializes a :class:`MMPC`.
)doc")
        .def("estimate",
             &MMPC::estimate,
             py::arg("hypot_test"),
             py::arg("nodes") = std::vector<std::string>(),
             py::arg("arc_blacklist") = ArcStringVector(),
             py::arg("arc_whitelist") = ArcStringVector(),
             py::arg("edge_blacklist") = EdgeStringVector(),
             py::arg("edge_whitelist") = EdgeStringVector(),
             py::arg("alpha") = 0.05,
             py::arg("ambiguous_threshold") = 0.5,
             py::arg("allow_bidirected") = true,
             py::arg("verbose") = 0,
             R"doc(
Estimates the skeleton (the partially directed graph) using the MMPC algorithm.

:param hypot_test: The :class:`IndependenceTest <pybnesian.IndependenceTest>` object used to
                   execute the conditional independence tests.
:param nodes: The list of nodes of the returned skeleton. If empty (the default value), the node names are extracted
              from :func:`IndependenceTest.variable_names() <pybnesian.IndependenceTest.variable_names>`.
:param arc_blacklist: List of arcs blacklist (forbidden arcs).
:param arc_whitelist: List of arcs whitelist (forced arcs).
:param edge_blacklist: List of edge blacklist (forbidden edges). This also implicitly applies a double arc
                       blacklist.
:param edge_whitelist: List of edge whitelist (forced edges).
:param alpha: The type I error of each independence test.
:param ambiguous_threshold: The ``ambiguous_threshold`` sets the threshold on the ratio of sepsets needed to declare a
                            v-structure. This is equal to ``ambiguous_threshold`` in :func:`PC.estimate`.
:param allow_bidirected: If True, it allows bi-directed arcs. This ensures that the result of the algorithm is
                         order-independent while applying v-structures (as in LCPC and LMPC in [pc-stable]_). Otherwise,
                         it does not return bi-directed arcs.
:param verbose: If True the progress will be displayed, otherwise nothing will be displayed.
:returns: A :class:`PartiallyDirectedGraph <pybnesian.PartiallyDirectedGraph>` trained by MMPC.
)doc")
        .def("estimate_conditional",
             &MMPC::estimate_conditional,
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
             py::arg("verbose") = 0,
             R"doc(
Estimates the conditional skeleton (the conditional partially directed graph) using the MMPC algorithm.

:param hypot_test: The :class:`IndependenceTest <pybnesian.IndependenceTest>` object used to
                   execute the conditional independence tests.
:param nodes: The list of nodes of the returned skeleton.
:param interface_nodes: The list of interface nodes of the returned skeleton.
:param arc_blacklist: List of arcs blacklist (forbidden arcs).
:param arc_whitelist: List of arcs whitelist (forced arcs).
:param edge_blacklist: List of edge blacklist (forbidden edges). This also implicitly applies a double arc
                       blacklist.
:param edge_whitelist: List of edge whitelist (forced edges).
:param alpha: The type I error of each independence test.
:param ambiguous_threshold: The ``ambiguous_threshold`` sets the threshold on the ratio of sepsets needed to declare a
                            v-structure. This is equal to ``ambiguous_threshold`` in :func:`PC.estimate_conditional`.
:param allow_bidirected: If True, it allows bi-directed arcs. This ensures that the result of the algorithm is
                         order-independent while applying v-structures (as in LCPC and LMPC in [pc-stable]_). Otherwise,
                         it does not return bi-directed arcs.
:param verbose: If True the progress will be displayed, otherwise nothing will be displayed.
:returns: A :class:`PartiallyDirectedGraph <pybnesian.PartiallyDirectedGraph>` trained by MMPC.
)doc");

    py::class_<MMHC>(root, "MMHC", R"doc(
This class implements Max-Min Hill-Climbing (MMHC) [mmhc]_. The MMHC algorithm finds the sets of possible arcs using the
:class:`MMPC` algorithm. Then, it trains the structure using a greedy hill-climbing algorithm
(:class:`GreedyHillClimbing`) blacklisting all the possible arcs not found by MMPC.
)doc")
        .def(py::init<>())
        .def("estimate",
             &MMHC::estimate,
             py::arg("hypot_test"),
             py::arg("operators"),
             py::arg("score"),
             py::arg("nodes") = std::vector<std::string>(),
             py::arg("bn_type") = GaussianNetworkType::get(),
             py::arg("arc_blacklist") = ArcStringVector(),
             py::arg("arc_whitelist") = ArcStringVector(),
             py::arg("edge_blacklist") = EdgeStringVector(),
             py::arg("edge_whitelist") = EdgeStringVector(),
             py::arg("type_blacklist") = FactorTypeVector(),
             py::arg("type_whitelist") = FactorTypeVector(),
             py::arg("callback") = nullptr,
             py::arg("max_indegree") = 0,
             py::arg("max_iters") = std::numeric_limits<int>::max(),
             py::arg("epsilon") = 0,
             py::arg("patience") = 0,
             py::arg("alpha") = 0.05,
             py::arg("verbose") = 0,
             R"doc(
Estimates the structure of a Bayesian network. This implementation calls :class:`MMPC` and :class:`GreedyHillClimbing`
with the set of parameters provided.

:param hypot_test: The :class:`IndependenceTest <pybnesian.IndependenceTest>` object used to
                   execute the conditional independence tests (for :class:`MMPC`).
:param operators: Set of operators in the search process (for :class:`GreedyHillClimbing`).
:param score: :class:`Score <pybnesian.Score>` that drives the search (for :class:`GreedyHillClimbing`).
:param nodes: The list of nodes of the returned skeleton. If empty (the default value), the node names are extracted
              from :func:`IndependenceTest.variable_names() <pybnesian.IndependenceTest.variable_names>`.
:param bn_type: A :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>`.
:param arc_blacklist: List of arcs blacklist (forbidden arcs).
:param arc_whitelist: List of arcs whitelist (forced arcs).
:param edge_blacklist: List of edge blacklist (forbidden edges). This also implicitly applies a double arc
                       blacklist.
:param edge_whitelist: List of edge whitelist (forced edges).
:param type_blacklist: List of type blacklist (forbidden :class:`FactorType <pybnesian.FactorType>`).
:param type_whitelist: List of type whitelist (forced :class:`FactorType <pybnesian.FactorType>`).
:param callback: Callback object that is called after each iteration of :class:`GreedyHillClimbing`.
:param max_indegree: Maximum indegree allowed in the graph (for :class:`GreedyHillClimbing`).
:param max_iters: Maximum number of search iterations (for :class:`GreedyHillClimbing`).
:param epsilon: Minimum delta score allowed for each operator. If the new operator is less than epsilon, the search
                process is stopped (for :class:`GreedyHillClimbing`).
:param patience: The patience parameter (only used with
                :class:`ValidatedScore <pybnesian.ValidatedScore>`). See `patience`_ (for
                :class:`GreedyHillClimbing`).
:param alpha: The type I error of each independence test (for :class:`MMPC`).
:param verbose: If True the progress will be displayed, otherwise nothing will be displayed.
:returns: The Bayesian network structure learned by MMHC.
)doc")
        .def("estimate_conditional",
             &MMHC::estimate_conditional,
             py::arg("hypot_test"),
             py::arg("operators"),
             py::arg("score"),
             py::arg("nodes") = std::vector<std::string>(),
             py::arg("interface_nodes") = std::vector<std::string>(),
             py::arg("bn_type") = GaussianNetworkType::get(),
             py::arg("arc_blacklist") = ArcStringVector(),
             py::arg("arc_whitelist") = ArcStringVector(),
             py::arg("edge_blacklist") = EdgeStringVector(),
             py::arg("edge_whitelist") = EdgeStringVector(),
             py::arg("type_blacklist") = FactorTypeVector(),
             py::arg("type_whitelist") = FactorTypeVector(),
             py::arg("callback") = nullptr,
             py::arg("max_indegree") = 0,
             py::arg("max_iters") = std::numeric_limits<int>::max(),
             py::arg("epsilon") = 0,
             py::arg("patience") = 0,
             py::arg("alpha") = 0.05,
             py::arg("verbose") = 0,
             R"doc(
Estimates the structure of a conditional Bayesian network. This implementation calls :class:`MMPC` and
:class:`GreedyHillClimbing` with the set of parameters provided.

:param hypot_test: The :class:`IndependenceTest <pybnesian.IndependenceTest>` object used to
                   execute the conditional independence tests (for :class:`MMPC`).
:param operators: Set of operators in the search process (for :class:`GreedyHillClimbing`).
:param score: :class:`Score <pybnesian.Score>` that drives the search (for :class:`GreedyHillClimbing`).
:param nodes: The list of nodes of the returned skeleton.
:param interface_nodes: The list of interface nodes of the returned skeleton.
:param bn_type: A :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>`.
:param arc_blacklist: List of arcs blacklist (forbidden arcs).
:param arc_whitelist: List of arcs whitelist (forced arcs).
:param edge_blacklist: List of edge blacklist (forbidden edges). This also implicitly applies a double arc
                       blacklist.
:param edge_whitelist: List of edge whitelist (forced edges).
:param type_blacklist: List of type blacklist (forbidden :class:`FactorType <pybnesian.FactorType>`).
:param type_whitelist: List of type whitelist (forced :class:`FactorType <pybnesian.FactorType>`).
:param callback: Callback object that is called after each iteration of :class:`GreedyHillClimbing`.
:param max_indegree: Maximum indegree allowed in the graph (for :class:`GreedyHillClimbing`).
:param max_iters: Maximum number of search iterations (for :class:`GreedyHillClimbing`).
:param epsilon: Minimum delta score allowed for each operator. If the new operator is less than epsilon, the search
                process is stopped (for :class:`GreedyHillClimbing`).
:param patience: The patience parameter (only used with
                :class:`ValidatedScore <pybnesian.ValidatedScore>`). See `patience`_ (for
                :class:`GreedyHillClimbing`).
:param alpha: The type I error of each independence test (for :class:`MMPC`).
:param verbose: If True the progress will be displayed, otherwise nothing will be displayed.
:returns: The conditional Bayesian network structure learned by MMHC.
)doc");

    py::class_<DMMHC>(root, "DMMHC", R"doc(
This class implements the Dynamic Max-Min Hill-Climbing (DMMHC) [dmmhc]_. This algorithm uses the :class:`MMHC` to
train the static and transition components of the dynamic Bayesian network.
)doc")
        .def(py::init<>())
        .def("estimate",
             &DMMHC::estimate,
             py::arg("hypot_test"),
             py::arg("operators"),
             py::arg("score"),
             py::arg("variables") = std::vector<std::string>(),
             py::arg("bn_type") = GaussianNetworkType::get(),
             py::arg("markovian_order") = 1,
             py::arg("static_callback") = nullptr,
             py::arg("transition_callback") = nullptr,
             py::arg("max_indegree") = 0,
             py::arg("max_iters") = std::numeric_limits<int>::max(),
             py::arg("epsilon") = 0,
             py::arg("patience") = 0,
             py::arg("alpha") = 0.05,
             py::arg("verbose") = 0,
             R"doc(
Estimates a dynamic Bayesian network. This implementation uses :class:`MMHC` to estimate both the static and transition
Bayesian networks. This set of parameters are provided to the functions :func:`MMHC.estimate` and
:func:`MMHC.estimate_conditional`.

:param hypot_test: The :class:`DynamicIndependenceTest <pybnesian.DynamicIndependenceTest>`
                   object used to execute the conditional independence tests (for :class:`MMPC`).
:param operators: Set of operators in the search process (for :class:`GreedyHillClimbing`).
:param score: :class:`DynamicScore <pybnesian.DynamicScore>` that drives the search
              (for :class:`GreedyHillClimbing`).
:param variables: The list of variables of the dynamic Bayesian network. If empty (the default value), the variable
                  names are extracted from :func:`DynamicIndependenceTest.variable_names() <pybnesian.DynamicIndependenceTest.variable_names>`.
:param bn_type: A :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>`.
:param markovian_order: The markovian order of the dynamic Bayesian network.
:param static_callback: Callback object that is called after each iteration of :class:`GreedyHillClimbing` to learn the
                        static component of the dynamic Bayesian network.
:param transition_callback: Callback object that is called after each iteration of :class:`GreedyHillClimbing` to learn
                            the transition component of the dynamic Bayesian network.
:param max_indegree: Maximum indegree allowed in the graph (for :class:`GreedyHillClimbing`).
:param max_iters: Maximum number of search iterations (for :class:`GreedyHillClimbing`).
:param epsilon: Minimum delta score allowed for each operator. If the new operator is less than epsilon, the search
                process is stopped (for :class:`GreedyHillClimbing`).
:param patience: The patience parameter (only used with
                :class:`ValidatedScore <pybnesian.ValidatedScore>`). See `patience`_ (for
                :class:`GreedyHillClimbing`).
:param alpha: The type I error of each independence test (for :class:`MMPC`).
:param verbose: If True the progress will be displayed, otherwise nothing will be displayed.
:returns: The dynamic Bayesian network structure learned by DMMHC.
)doc");
}
