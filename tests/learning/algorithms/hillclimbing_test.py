import numpy as np
import util_test

import pybnesian as pbn
from pybnesian import BayesianNetwork, BayesianNetworkType

df = util_test.generate_normal_data(1000)
# TODO: Add tests for normal data with dependencies
# dep_df = util_test.generate_normal_data_dep(1000)


def test_hc_estimate():
    bic = pbn.BIC(df)
    column_names = list(df.columns.values)
    start = pbn.GaussianNetwork(column_names)

    # Check algorithm with BN with nodes removed.
    column_names.insert(1, "e")
    column_names.insert(3, "f")
    start_removed_nodes = pbn.GaussianNetwork(column_names)
    start_removed_nodes.remove_node("e")
    start_removed_nodes.remove_node("f")

    arc_set = pbn.ArcOperatorSet()

    hc = pbn.GreedyHillClimbing()

    res = hc.estimate(arc_set, bic, start, max_iters=1)
    assert res.num_arcs() == 1
    added_arc = res.arcs()[0]
    op_delta = bic.score(res) - bic.score(start)

    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, max_iters=1)
    assert res.num_arcs() == 1
    added_arc_removed = res_removed.arcs()[0]
    assert added_arc == added_arc_removed or added_arc == added_arc_removed[::-1]
    assert np.isclose(op_delta, bic.score(res_removed) - bic.score(start_removed_nodes))

    # BIC is score equivalent, so if we blacklist the added_arc, its reverse will be added.
    res = hc.estimate(arc_set, bic, start, max_iters=1, arc_blacklist=[added_arc])
    assert res.num_arcs() == 1
    reversed_arc = res.arcs()[0][::-1]
    assert added_arc == reversed_arc

    res_removed = hc.estimate(
        arc_set,
        bic,
        start_removed_nodes,
        max_iters=1,
        arc_blacklist=[added_arc_removed],
    )
    assert res.num_arcs() == 1
    reversed_arc_removed = res_removed.arcs()[0][::-1]
    assert added_arc_removed == reversed_arc_removed

    assert np.isclose(
        op_delta,
        bic.local_score(res, added_arc[1], [added_arc[0]])
        - bic.local_score(res, added_arc[1], []),
    )
    assert np.isclose(
        op_delta,
        bic.local_score(res, added_arc_removed[1], [added_arc_removed[0]])
        - bic.local_score(res, added_arc_removed[1], []),
    )

    res = hc.estimate(arc_set, bic, start, epsilon=(op_delta + 0.01))
    assert res.num_arcs() == start.num_arcs()

    res_removed = hc.estimate(
        arc_set, bic, start_removed_nodes, epsilon=(op_delta + 0.01)
    )
    assert res_removed.num_arcs() == start_removed_nodes.num_arcs()

    # Can't compare models because the arcs could be oriented in different direction,
    # leading to a different search path. Execute the code, just to check no error is given.
    res = hc.estimate(arc_set, bic, start, verbose=False)
    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, verbose=False)


def test_hc_conditional_estimate():
    bic = pbn.BIC(df)
    column_names = list(df.columns.values)

    start = pbn.ConditionalGaussianNetwork(column_names[2:], column_names[:2])

    nodes = column_names[2:]
    nodes.insert(1, "e")
    interface_nodes = column_names[:2]
    interface_nodes.insert(1, "f")
    start_removed_nodes = pbn.ConditionalGaussianNetwork(nodes, interface_nodes)
    start_removed_nodes.remove_node("e")
    start_removed_nodes.remove_interface_node("f")

    arc_set = pbn.ArcOperatorSet()
    hc = pbn.GreedyHillClimbing()

    res = hc.estimate(arc_set, bic, start, max_iters=1, verbose=False)
    assert res.num_arcs() == 1
    added_arc = res.arcs()[0]
    op_delta = bic.score(res) - bic.score(start)

    res_removed = hc.estimate(
        arc_set, bic, start_removed_nodes, max_iters=1, verbose=False
    )
    assert res_removed.num_arcs() == 1
    added_arc_removed = res_removed.arcs()[0]
    assert added_arc == added_arc_removed or added_arc == added_arc_removed[::-1]
    assert np.isclose(op_delta, bic.score(res_removed) - bic.score(start_removed_nodes))

    assert np.isclose(
        op_delta,
        bic.local_score(res, added_arc[1], [added_arc[0]])
        - bic.local_score(res, added_arc[1], []),
    )
    assert np.isclose(
        op_delta,
        bic.local_score(res, added_arc_removed[1], [added_arc_removed[0]])
        - bic.local_score(res, added_arc_removed[1], []),
    )

    res = hc.estimate(arc_set, bic, start, epsilon=(op_delta + 0.01))
    assert res.num_arcs() == start.num_arcs()
    res_removed = hc.estimate(
        arc_set, bic, start_removed_nodes, epsilon=(op_delta + 0.01)
    )
    assert res_removed.num_arcs() == start_removed_nodes.num_arcs()

    res = hc.estimate(arc_set, bic, start, verbose=False)
    assert all(map(lambda arc: not res.is_interface(arc[1]), res.arcs()))
    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, verbose=False)
    assert all(
        map(lambda arc: not res_removed.is_interface(arc[1]), res_removed.arcs())
    )


def test_hc_estimate_validation():
    column_names = list(df.columns.values)
    start = pbn.GaussianNetwork(column_names)

    column_names.insert(1, "e")
    column_names.insert(4, "f")
    start_removed_nodes = pbn.GaussianNetwork(column_names)
    start_removed_nodes.remove_node("e")
    start_removed_nodes.remove_node("f")

    vl = pbn.ValidatedLikelihood(df)
    arc_set = pbn.ArcOperatorSet()

    hc = pbn.GreedyHillClimbing()

    res = hc.estimate(arc_set, vl, start, max_iters=1)
    assert res.num_arcs() == 1
    added_arc = res.arcs()[0]
    op_delta = vl.cv_lik.score(res) - vl.cv_lik.score(start)

    res_removed = hc.estimate(arc_set, vl, start_removed_nodes, max_iters=1)
    assert res_removed.num_arcs() == 1
    added_arc_removed = res_removed.arcs()[0]
    assert added_arc == added_arc_removed or added_arc == added_arc_removed[::-1]
    assert np.isclose(
        op_delta, vl.cv_lik.score(res_removed) - vl.cv_lik.score(start_removed_nodes)
    )

    assert np.isclose(
        op_delta,
        vl.cv_lik.local_score(res, added_arc[1], [added_arc[0]])
        - vl.cv_lik.local_score(res, added_arc[1], []),
    )
    assert np.isclose(
        op_delta,
        vl.cv_lik.local_score(res, added_arc_removed[1], [added_arc_removed[0]])
        - vl.cv_lik.local_score(res, added_arc_removed[1], []),
    )

    # CV is score equivalent for GBNs, so if we blacklist the added_edge, its reverse will be added.
    res = hc.estimate(arc_set, vl, start, max_iters=1, arc_blacklist=[added_arc])
    assert res.num_arcs() == 1
    reversed_arc = res.arcs()[0][::-1]
    assert added_arc == reversed_arc

    res_removed = hc.estimate(
        arc_set, vl, start_removed_nodes, max_iters=1, arc_blacklist=[added_arc_removed]
    )
    assert res_removed.num_arcs() == 1
    reversed_arc_removed = res_removed.arcs()[0][::-1]
    assert reversed_arc == reversed_arc_removed

    res = hc.estimate(arc_set, vl, start, epsilon=(op_delta + 0.01))
    assert res.num_arcs() == start.num_arcs()

    res_removed = hc.estimate(
        arc_set, vl, start_removed_nodes, epsilon=(op_delta + 0.01)
    )
    assert res_removed.num_arcs() == start_removed_nodes.num_arcs()

    # Can't compare models because the arcs could be oriented in different direction,
    # leading to a different search path. Execute the code, just to check no error is given.
    res = hc.estimate(arc_set, vl, start, verbose=False)
    res_removed = hc.estimate(arc_set, vl, start_removed_nodes, verbose=False)


def test_hc_shortcut_function():
    model = pbn.hc(df, bn_type=pbn.GaussianNetworkType())
    assert type(model) == pbn.GaussianNetwork

    model = pbn.hc(
        df, bn_type=MyRestrictedGaussianNetworkType(), score="bic", operators=["arcs"]
    )
    assert type(model) == NewBN


# # NOTE: Deprecated test for PyBNesian with full covariance matrices
# def test_hc_arc_singular_covariance():
#     """Function to test if with the GBN, KDE and SPBN, the HC algorithm raises an exception when the covariance matrix is singular. Then we check if the learnt model is valid."""
#     column_names = list(dep_df.columns.values)
#     # GBN
#     gbn = pbn.GaussianNetwork(nodes=column_names)
#     gbn = pbn.hc(
#         dep_df,
#         start=gbn,
#         max_iters=int(1e4),
#         verbose=True,
#     )
#     gbn.fit(dep_df)
#     assert gbn.num_arcs() == 0
#     assert np.count_nonzero(np.isnan(gbn.logl(dep_df))) == 0
#     for c in column_names:
#         print(f"{gbn.cpd(c)}")

#     # KDE
#     kde = pbn.KDENetwork(nodes=column_names)
#     kde = pbn.hc(
#         dep_df,
#         start=kde,
#         max_iters=int(1e4),
#         verbose=True,
#     )
#     kde.fit(dep_df)
#     assert kde.num_arcs() == 0
#     assert np.count_nonzero(np.isnan(kde.logl(dep_df))) == 0
#     for c in column_names:
#         print(f"{kde.cpd(c)}")

#     # SPBN
#     spbn = pbn.SemiparametricBN(nodes=column_names)
#     spbn = pbn.hc(
#         dep_df,
#         start=spbn,
#         max_iters=int(1e4),
#         verbose=True,
#     )
#     spbn.fit(dep_df)
#     assert spbn.num_arcs() == 0
#     assert np.count_nonzero(np.isnan(spbn.logl(dep_df))) == 0
#     for c in column_names:
#         print(f"{spbn.cpd(c)}")


# TODO: Test for when one variable has 0 variance in k-fold cross-validation for CKDEType


class MyRestrictedGaussianNetworkType(BayesianNetworkType):
    def __init__(self):
        BayesianNetworkType.__init__(self)

    def is_homogeneous(self):
        return True

    def default_node_type(self):
        return pbn.LinearGaussianCPDType()

    def can_have_arc(self, model, source, target):
        return "a" in source

    def new_bn(self, nodes):
        return NewBN(nodes)

    def __str__(self):
        return "MyRestrictedGaussianNetworkType"


class NewBN(BayesianNetwork):
    def __init__(self, variables, arcs=None):
        if arcs is None:
            BayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables)
        else:
            BayesianNetwork.__init__(
                self, MyRestrictedGaussianNetworkType(), variables, arcs
            )

        self.extra_data = "extra"

    def __getstate_extra__(self):
        return self.extra_data

    def __setstate_extra__(self, extra):
        self.extra_data = extra


def test_newbn_estimate_validation():
    start = NewBN(["a", "b", "c", "d"])
    hc = pbn.GreedyHillClimbing()
    arc = pbn.ArcOperatorSet()
    bic = pbn.BIC(df)

    estimated = hc.estimate(arc, bic, start)

    assert type(start) == type(estimated)
    assert estimated.extra_data == "extra"
