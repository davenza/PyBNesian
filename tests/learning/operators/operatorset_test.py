import pytest
import pyarrow as pa
from pgm_dataset.learning.operators import ArcOperatorSet, ChangeNodeTypeSet, OperatorTabuSet, AddArc, OperatorType
from pgm_dataset.learning.scores import BIC, CVLikelihood
from pgm_dataset.models import GaussianNetwork, SemiparametricBN
import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)


def test_create():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])

    bic = BIC(df)
    arc_gbn_bic = ArcOperatorSet(gbn, bic, [], [], 0)
    arc_gbn_bic = ArcOperatorSet(gbn, bic)

    with pytest.raises(ValueError) as ex:
        arc_spbn_bic = ArcOperatorSet(spbn, bic, [], [], 0)
    "Invalid score" in str(ex.value)

    cv = CVLikelihood(df)
    arc_gbn_cv = ArcOperatorSet(gbn, cv, [], [], 0)
    arc_gbn_cv = ArcOperatorSet(gbn, cv)
    arc_spbn_cv = ArcOperatorSet(spbn, cv, [], [], 0)
    arc_spbn_cv = ArcOperatorSet(spbn, cv)

    with pytest.raises(TypeError) as ex:
        node_gbn_cv = ChangeNodeTypeSet(gbn, cv, [])
    "incompatible function arguments" in str(ex.value)

    with pytest.raises(TypeError) as ex:
        node_spbn_bic = ChangeNodeTypeSet(spbn, bic, [])
    "incompatible function arguments" in str(ex.value)

    node_spbn_cv = ChangeNodeTypeSet(spbn, cv, [])
    node_spbn_cv = ChangeNodeTypeSet(spbn, cv, [])


def test_check_max_score():
    gbn = GaussianNetwork(['a', 'b'])

    bic = BIC(df)
    arc_gbn_bic = ArcOperatorSet(gbn, bic, [], [], 0)

    arc_gbn_bic.cache_scores(gbn)
    op = arc_gbn_bic.find_max(gbn)

    assert op.delta == (bic.local_score(gbn, 'b', ['a']) - bic.local_score(gbn, 'b'))

    arc_gbn_bic = ArcOperatorSet(gbn, bic, [(op.source, op.target)], [], 0)
    arc_gbn_bic.cache_scores(gbn)
    
    op2 = arc_gbn_bic.find_max(gbn)

    assert op.source == op2.target
    assert op.target == op2.source
    assert (op.type == op2.type) and (op.type == OperatorType.ADD_ARC)

def test_nomax():
    gbn = GaussianNetwork(['a', 'b'])

    bic = BIC(df)
    arc_gbn_bic = ArcOperatorSet(gbn, bic, [], [('a', 'b')], 0)
    arc_gbn_bic.cache_scores(gbn)

    op = arc_gbn_bic.find_max(gbn)

    assert op is None



