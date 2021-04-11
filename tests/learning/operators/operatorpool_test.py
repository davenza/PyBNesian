import pytest
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import CVLikelihood
from pybnesian.models import SemiparametricBN
import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)

def test_create():
    arcs = ArcOperatorSet()
    node_type = ChangeNodeTypeSet()
    pool = OperatorPool([arcs, node_type])

    with pytest.raises(ValueError) as ex:
        pool = OperatorPool([])
    assert "cannot be empty" in str(ex.value)

def test_find_max():
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    cv = CVLikelihood(df)
    arcs = ArcOperatorSet()
    node_type = ChangeNodeTypeSet()
    
    arcs.cache_scores(spbn, cv)
    node_type.cache_scores(spbn, cv)
    
    arcs_max = arcs.find_max(spbn)
    node_max = node_type.find_max(spbn)

    pool = OperatorPool([arcs, node_type])
    pool.cache_scores(spbn, cv)

    op_combined = pool.find_max(spbn)

    if arcs_max.delta() >= node_max.delta():
        assert op_combined == arcs_max
    else:
        assert op_combined == node_max

    