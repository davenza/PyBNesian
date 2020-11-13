import pytest
import pyarrow as pa
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import CVLikelihood
from pybnesian.models import SemiparametricBN
import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)

def test_create():
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    cv = CVLikelihood(df)
    arcs = ArcOperatorSet(cv)
    node_type = ChangeNodeTypeSet(cv)
    pool = OperatorPool(spbn, cv, [arcs, node_type])

    with pytest.raises(ValueError) as ex:
        pool = OperatorPool(spbn, cv, [])
    "cannot be empty" in str(ex.value)

def test_find_max():
    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    cv = CVLikelihood(df)
    arcs = ArcOperatorSet(cv)
    node_type = ChangeNodeTypeSet(cv)
    
    arcs.cache_scores(spbn)
    node_type.cache_scores(spbn)
    
    arcs_max = arcs.find_max(spbn)
    node_max = node_type.find_max(spbn)

    
    pool = OperatorPool(spbn, cv, [arcs, node_type])
    pool.cache_scores(spbn)

    op_combined = pool.find_max(spbn)

    if arcs_max.delta >= node_max.delta:
        assert op_combined == arcs_max
    else:
        assert op_combined == node_max

    