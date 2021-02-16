import numpy as np
from pybnesian.learning.algorithms import GreedyHillClimbing, hc
from pybnesian.learning.operators import ArcOperatorSet, OperatorPool
from pybnesian.learning.scores import BIC, ValidatedLikelihood
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
    added_arc_removed = res_removed.arcs()[0]
    assert added_arc == added_arc_removed or added_arc == added_arc_removed[::-1]
    assert np.isclose(op_delta, bic.score(res_removed) - bic.score(start_removed_nodes))

    # BIC is score equivalent, so if we blacklist the added_arc, its reverse will be added.
    res = hc.estimate(arc_set, bic, start, max_iters=1, arc_blacklist=[added_arc])
    assert res.num_arcs() == 1
    reversed_arc = res.arcs()[0][::-1]
    assert added_arc == reversed_arc

    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, max_iters=1, arc_blacklist=[added_arc_removed])
    assert res.num_arcs() == 1
    reversed_arc_removed = res_removed.arcs()[0][::-1]
    assert added_arc_removed == reversed_arc_removed

    assert np.isclose(op_delta, bic.local_score(res, added_arc[1], [added_arc[0]]) - bic.local_score(res, added_arc[1], []))
    assert np.isclose(op_delta, bic.local_score(res, added_arc_removed[1], [added_arc_removed[0]]) - bic.local_score(res, added_arc_removed[1], []))

    res = hc.estimate(arc_set, bic, start, epsilon=(op_delta + 0.01))
    assert res.num_arcs() == start.num_arcs()

    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, epsilon=(op_delta + 0.01))
    assert res_removed.num_arcs() == start_removed_nodes.num_arcs()

    # Can't compare models because the arcs could be oriented in different direction, 
    # leading to a different search path. Execute the code, just to check no error is given.
    res = hc.estimate(arc_set, bic, start, verbose=False)
    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, verbose=False)

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
    added_arc = res.arcs()[0]
    op_delta = bic.score(res) - bic.score(start)

    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, max_iters=1, verbose=False)
    assert res_removed.num_arcs() == 1
    added_arc_removed = res_removed.arcs()[0]
    assert added_arc == added_arc_removed or added_arc == added_arc_removed[::-1]
    assert np.isclose(op_delta, bic.score(res_removed) - bic.score(start_removed_nodes))

    assert np.isclose(op_delta, bic.local_score(res, added_arc[1], [added_arc[0]]) -
                                bic.local_score(res, added_arc[1], []))
    assert np.isclose(op_delta, bic.local_score(res, added_arc_removed[1], [added_arc_removed[0]]) -
                                bic.local_score(res, added_arc_removed[1], []))

    res = hc.estimate(arc_set, bic, start, epsilon=(op_delta + 0.01))
    assert res.num_arcs() == start.num_arcs()
    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, epsilon=(op_delta + 0.01))
    assert res_removed.num_arcs() == start_removed_nodes.num_arcs()

    res = hc.estimate(arc_set, bic, start, verbose=False)
    assert all(map(lambda arc : not res.is_interface(arc[1]), res.arcs()))
    res_removed = hc.estimate(arc_set, bic, start_removed_nodes, verbose=False)
    assert all(map(lambda arc : not res_removed.is_interface(arc[1]), res_removed.arcs()))

def test_hc_estimate_validation():
    column_names = list(df.columns.values)
    start = GaussianNetwork(column_names)

    column_names.insert(1, 'e')
    column_names.insert(4, 'f')
    start_removed_nodes = GaussianNetwork(column_names)
    start_removed_nodes.remove_node('e')
    start_removed_nodes.remove_node('f')
    
    vl = ValidatedLikelihood(df)
    arc_set = ArcOperatorSet()

    hc = GreedyHillClimbing()

    res = hc.estimate(arc_set, vl, start, max_iters=1)
    assert res.num_arcs() == 1
    added_arc = res.arcs()[0]
    op_delta = vl.cv_lik.score(res) - vl.cv_lik.score(start)

    res_removed = hc.estimate(arc_set, vl, start_removed_nodes, max_iters=1)
    assert res_removed.num_arcs() == 1
    added_arc_removed = res_removed.arcs()[0]
    assert added_arc == added_arc_removed or added_arc == added_arc_removed[::-1]
    assert np.isclose(op_delta, vl.cv_lik.score(res_removed) - vl.cv_lik.score(start_removed_nodes))

    assert np.isclose(op_delta, vl.cv_lik.local_score(res, added_arc[1], [added_arc[0]]) - 
                                vl.cv_lik.local_score(res, added_arc[1], []))
    assert np.isclose(op_delta, vl.cv_lik.local_score(res, added_arc_removed[1], [added_arc_removed[0]]) -
                                vl.cv_lik.local_score(res, added_arc_removed[1], []))

    # CV is score equivalent for GBNs, so if we blacklist the added_edge, its reverse will be added.
    res = hc.estimate(arc_set, vl, start, max_iters=1, arc_blacklist=[added_arc])
    assert res.num_arcs() == 1
    reversed_arc = res.arcs()[0][::-1]
    assert added_arc == reversed_arc

    res_removed = hc.estimate(arc_set, vl, start_removed_nodes, max_iters=1, arc_blacklist=[added_arc_removed])
    assert res_removed.num_arcs() == 1
    reversed_arc_removed = res_removed.arcs()[0][::-1]
    assert reversed_arc == reversed_arc_removed
    
    res = hc.estimate(arc_set, vl, start, epsilon=(op_delta + 0.01))
    assert res.num_arcs() == start.num_arcs()

    res_removed = hc.estimate(arc_set, vl, start_removed_nodes, epsilon=(op_delta + 0.01))
    assert res_removed.num_arcs() == start_removed_nodes.num_arcs()

    # Can't compare models because the arcs could be oriented in different direction, 
    # leading to a different search path. Execute the code, just to check no error is given.
    res = hc.estimate(arc_set, vl, start, verbose=False)
    res_removed = hc.estimate(arc_set, vl, start_removed_nodes, verbose=False)