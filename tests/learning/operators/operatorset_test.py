import pytest
import pyarrow as pa
from pybnesian.learning.operators import ArcOperatorSet, ChangeNodeTypeSet, OperatorTabuSet, AddArc, OperatorType
from pybnesian.learning.scores import BIC, CVLikelihood
from pybnesian.models import GaussianNetwork, SemiparametricBN
from pybnesian.factors import FactorType
import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)


def test_create():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])

    bic = BIC(df)
    arc_bic = ArcOperatorSet(bic)
    
    cv = CVLikelihood(df)
    arc_cv = ArcOperatorSet(cv)

    with pytest.raises(ValueError) as ex:
        arc_bic.cache_scores(spbn)
    "Invalid score" in str(ex.value)

    node_cv = ChangeNodeTypeSet(cv)

    with pytest.raises(TypeError) as ex:
        node_cv.cache_scores(gbn)
    "incompatible function arguments" in str(ex.value)

    node_bic = ChangeNodeTypeSet(bic)

    with pytest.raises(ValueError) as ex:
        node_bic.cache_scores(spbn)
    "Invalid score" in str(ex.value)

def test_lists():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])
    bic = BIC(df)
    arc_bic = ArcOperatorSet(bic)

    arc_bic.set_arc_blacklist([("b", "a")])
    arc_bic.set_arc_whitelist([("b", "c")])
    arc_bic.set_max_indegree(3)
    arc_bic.set_type_whitelist([("a", FactorType.LinearGaussianCPD)])

    arc_bic.cache_scores(gbn)

    arc_bic.set_arc_blacklist([("e", "a")])

    with pytest.raises(ValueError) as ex:
        arc_bic.cache_scores(gbn)
    "present in the blacklist, but not" in str(ex.value)

    arc_bic.set_arc_whitelist([("e", "a")])

    with pytest.raises(ValueError) as ex:
        arc_bic.cache_scores(gbn)
    "present in the whitelist, but not" in str(ex.value)


def test_check_max_score():
    gbn = GaussianNetwork(['a', 'b'])

    bic = BIC(df)
    arc_bic = ArcOperatorSet(bic)

    arc_bic.cache_scores(gbn)
    op = arc_bic.find_max(gbn)

    assert op.delta == (bic.local_score(gbn, 'b', ['a']) - bic.local_score(gbn, 'b'))

    # arc_gbn_bic = ArcOperatorSet( bic, [(op.source, op.target)], [], 0)
    arc_bic.set_arc_blacklist([(op.source, op.target)])
    arc_bic.cache_scores(gbn)
    
    op2 = arc_bic.find_max(gbn)

    assert op.source == op2.target
    assert op.target == op2.source
    assert (op.type == op2.type) and (op.type == OperatorType.ADD_ARC)

def test_nomax():
    gbn = GaussianNetwork(['a', 'b'])

    bic = BIC(df)
    arc_bic = ArcOperatorSet(bic, whitelist=[("a", "b")])
    arc_bic.cache_scores(gbn)

    op = arc_bic.find_max(gbn)

    assert op is None



