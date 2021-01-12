from pybnesian.learning.algorithms import GreedyHillClimbing, hc
from pybnesian.learning.operators import ArcOperatorSet, OperatorPool
from pybnesian.learning.scores import BIC, HoldoutLikelihood, CVLikelihood
from pybnesian.models import GaussianNetwork, ConditionalGaussianNetwork
import util_test

df = util_test.generate_normal_data(1000)

def test_hc_estimate():
    bic = BIC(df)
    column_names = list(df.columns.values)
    start = GaussianNetwork(column_names)

    # Check algorithm with BN with nodes removed.
    column_names.insert(1, 'e')
    column_names.insert(3, 'f')
    start_removed_nodes = GaussianNetwork(column_names)
    start_removed_nodes.remove_node('e')
    start_removed_nodes.remove_node('f')

    arc_set = ArcOperatorSet()

    hc = GreedyHillClimbing()

    res = hc.estimate(arc_set, bic, start, max_iters=1)
    assert res.num_arcs() == 1
    added_arc = res.arcs()[0]
    op_delta = bic.score(res) - bic.score(start)

    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, max_iters=1)
    assert res.num_arcs() == 1
    assert added_arc == res_removed.arcs()[0]
    assert op_delta == bic.score(res_removed) - bic.score(start_removed_nodes)

    # BIC is score equivalent, so if we blacklist the added_arc, its reverse will be added.
    res = hc.estimate(arc_set, bic, start, max_iters=1, arc_blacklist=[added_arc])
    assert res.num_arcs() == 1
    reversed_arc = res.arcs()[0]
    assert added_arc == reversed_arc[::-1]

    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, max_iters=1, arc_blacklist=[added_arc])
    assert res.num_arcs() == 1
    assert reversed_arc == res_removed.arcs()[0]

    res = hc.estimate(arc_set, bic, start, epsilon=(op_delta + 0.01))
    assert res.num_arcs() == start.num_arcs()

    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, epsilon=(op_delta + 0.01))
    assert res_removed.num_arcs() == start_removed_nodes.num_arcs()

def test_hc_conditional_estimate():
    bic = BIC(df)
    column_names = list(df.columns.values)

    start = ConditionalGaussianNetwork(column_names[2:], column_names[:2])
    
    nodes = column_names[2:]
    nodes.insert(1, 'e')
    interface_nodes = column_names[:2]
    interface_nodes.insert(1, 'f')
    start_removed_nodes = ConditionalGaussianNetwork(nodes, interface_nodes)
    start_removed_nodes.remove_node('e')
    start_removed_nodes.remove_interface_node('f')
    
    arc_set = ArcOperatorSet()
    hc = GreedyHillClimbing()

    res = hc.estimate(arc_set, bic, start, max_iters=1, verbose=False)
    assert res.num_arcs() == 1
    added_edge = res.arcs()[0]
    op_delta = bic.score(res) - bic.score(start)

    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, max_iters=1, verbose=False)
    assert res_removed.num_arcs() == 1
    assert added_edge == res_removed.arcs()[0]
    assert op_delta == bic.score(res_removed) - bic.score(start_removed_nodes)

    assert op_delta == bic.local_score(res, added_edge[1], [added_edge[0]]) - bic.local_score(res, added_edge[1], [])

    res = hc.estimate(arc_set, bic, start, epsilon=(op_delta + 0.01))
    assert res.num_arcs() == start.num_arcs()
    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, epsilon=(op_delta + 0.01))
    assert res_removed.num_arcs() == start_removed_nodes.num_arcs()

    res = hc.estimate(arc_set, bic, start, verbose=False)
    assert all(map(lambda arc : not res.is_interface(arc[1]), res.arcs()))
    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, verbose=False)
    assert all(map(lambda arc : not res_removed.is_interface(arc[1]), res_removed.arcs()))
    assert set(res.arcs()) == set(res_removed.arcs())

def test_hc_estimate_validation():
    start = GaussianNetwork(df.columns.values)
    
    holdout = HoldoutLikelihood(df)
    cv = CVLikelihood(holdout.training_data())
    arc_set = ArcOperatorSet()

    hc = GreedyHillClimbing()

    res = hc.estimate_validation(arc_set, cv, holdout, start, max_iters=1)
    assert res.num_arcs() == 1
    added_edge = res.arcs()[0]
    op_delta = cv.score(res) - cv.score(start)

    # CV is score equivalent for GBNs, so if we blacklist the added_edge, its reverse will be added.
    res = hc.estimate_validation(arc_set, cv, holdout, start, max_iters=1, arc_blacklist=[added_edge])
    assert res.num_arcs() == 1
    reversed_edge = res.arcs()[0]
    assert added_edge == reversed_edge[::-1]

    res = hc.estimate_validation(arc_set, cv, holdout, start, epsilon=(op_delta + 0.01))
    assert res.num_arcs() == start.num_arcs()