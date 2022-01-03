import pytest
import numpy as np
import pybnesian as pbn
import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)

def test_create_change_node():
    gbn = pbn.GaussianNetwork(['a', 'b', 'c', 'd'])
    
    cv = pbn.CVLikelihood(df)

    node_op = pbn.ChangeNodeTypeSet()

    with pytest.raises(ValueError) as ex:
        node_op.cache_scores(gbn, cv)
    assert "can only be used with non-homogeneous" in str(ex.value)

def test_lists():
    gbn = pbn.GaussianNetwork(['a', 'b', 'c', 'd'])
    bic = pbn.BIC(df)
    arc_op = pbn.ArcOperatorSet()

    arc_op.set_arc_blacklist([("b", "a")])
    arc_op.set_arc_whitelist([("b", "c")])
    arc_op.set_max_indegree(3)
    arc_op.set_type_whitelist([("a", pbn.LinearGaussianCPDType())])

    arc_op.cache_scores(gbn, bic)

    arc_op.set_arc_blacklist([("e", "a")])

    with pytest.raises(ValueError) as ex:
        arc_op.cache_scores(gbn, bic)
    assert "not present in the graph" in str(ex.value)

    arc_op.set_arc_whitelist([("e", "a")])

    with pytest.raises(ValueError) as ex:
        arc_op.cache_scores(gbn, bic)
    assert "not present in the graph" in str(ex.value)


def test_check_max_score():
    gbn = pbn.GaussianNetwork(['c', 'd'])

    bic = pbn.BIC(df)
    arc_op = pbn.ArcOperatorSet()

    arc_op.cache_scores(gbn, bic)
    op = arc_op.find_max(gbn)

    assert np.isclose(op.delta(), (bic.local_score(gbn, 'd', ['c']) - bic.local_score(gbn, 'd')))

    # BIC is decomposable so the best operation is the arc in reverse direction.
    arc_op.set_arc_blacklist([(op.source(), op.target())])
    arc_op.cache_scores(gbn, bic)
    
    op2 = arc_op.find_max(gbn)

    assert op.source() == op2.target()
    assert op.target() == op2.source()
    assert (type(op) == type(op2)) and (type(op) == pbn.AddArc)

def test_nomax():
    gbn = pbn.GaussianNetwork(['a', 'b'])

    bic = pbn.BIC(df)
    arc_op = pbn.ArcOperatorSet(whitelist=[("a", "b")])
    arc_op.cache_scores(gbn, bic)

    op = arc_op.find_max(gbn)

    assert op is None



