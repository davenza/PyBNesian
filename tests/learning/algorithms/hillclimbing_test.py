import pyarrow as pa
from pybnesian.learning.algorithms import GreedyHillClimbing, hc
from pybnesian.learning.operators import ArcOperatorSet, OperatorPool
from pybnesian.learning.scores import BIC, HoldoutLikelihood, CVLikelihood
from pybnesian.models import GaussianNetwork
import util_test

df = util_test.generate_normal_data(1000)

def test_hc_estimate():
    bic = BIC(df)
    start = GaussianNetwork(df.columns.values)
    arc_set = ArcOperatorSet()

    hc = GreedyHillClimbing()

    res = hc.estimate(arc_set, bic, start, max_iters=1)
    assert res.num_arcs() == 1
    added_edge = res.arcs()[0]
    op_delta = bic.score(res) - bic.score(start)

    # BIC is score equivalent, so if we blacklist the added_edge, its reverse will be added.
    res = hc.estimate(arc_set, bic, start, max_iters=1, arc_blacklist=[added_edge])
    assert res.num_arcs() == 1
    reversed_edge = res.arcs()[0]
    assert added_edge == reversed_edge[::-1]

    res = hc.estimate(arc_set, bic, start, epsilon=(op_delta + 0.01))
    assert res.num_arcs() == start.num_arcs()

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