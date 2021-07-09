import pytest
import pybnesian as pbn
import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)

def test_create():
    arcs = pbn.ArcOperatorSet()
    node_type = pbn.ChangeNodeTypeSet()
    pool = pbn.OperatorPool([arcs, node_type])

    with pytest.raises(ValueError) as ex:
        pool = pbn.OperatorPool([])
    assert "cannot be empty" in str(ex.value)

def test_find_max():
    spbn = pbn.SemiparametricBN(['a', 'b', 'c', 'd'])
    cv = pbn.CVLikelihood(df)
    arcs = pbn.ArcOperatorSet()
    node_type = pbn.ChangeNodeTypeSet()
    
    arcs.cache_scores(spbn, cv)
    spbn.set_unknown_node_types(df)
    node_type.cache_scores(spbn, cv)
    
    arcs_max = arcs.find_max(spbn)
    node_max = node_type.find_max(spbn)

    pool = pbn.OperatorPool([arcs, node_type])
    pool.cache_scores(spbn, cv)

    op_combined = pool.find_max(spbn)

    if arcs_max.delta() >= node_max.delta():
        assert op_combined == arcs_max
    else:
        assert op_combined == node_max

    