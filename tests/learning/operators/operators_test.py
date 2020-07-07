import pytest
import pyarrow as pa
from pgm_dataset.factors import FactorType
from pgm_dataset.models import BayesianNetworkType, GaussianNetwork, SemiparametricBN
from pgm_dataset.learning.operators import OperatorType, AddArc, RemoveArc, FlipArc, ChangeNodeType


def test_create():
    o = AddArc(BayesianNetworkType.GBN, "a", "b", 1)
    assert o.source == 'a'
    assert o.target == 'b'
    assert o.delta == 1
    assert o.type == OperatorType.ADD_ARC

    o = AddArc(BayesianNetworkType.SPBN, "a", "b", 1)
    assert o.source == 'a'
    assert o.target == 'b'
    assert o.delta == 1
    assert o.type == OperatorType.ADD_ARC

    o = RemoveArc(BayesianNetworkType.GBN, "a", "b", 1)
    assert o.source == 'a'
    assert o.target == 'b'
    assert o.delta == 1
    assert o.type == OperatorType.REMOVE_ARC

    o = RemoveArc(BayesianNetworkType.SPBN, "a", "b", 1)
    assert o.source == 'a'
    assert o.target == 'b'
    assert o.delta == 1
    assert o.type == OperatorType.REMOVE_ARC

    o = FlipArc(BayesianNetworkType.GBN, "a", "b", 1)
    assert o.source == 'a'
    assert o.target == 'b'
    assert o.delta == 1
    assert o.type == OperatorType.FLIP_ARC

    o = FlipArc(BayesianNetworkType.SPBN, "a", "b", 1)
    assert o.source == 'a'
    assert o.target == 'b'
    assert o.delta == 1
    assert o.type == OperatorType.FLIP_ARC

    with pytest.raises(ValueError) as ex:
        o = ChangeNodeType(BayesianNetworkType.GBN, "a", FactorType.CKDE, 1)
    "not available for BayesianNetwork " in str(ex.value)

    o = ChangeNodeType(BayesianNetworkType.SPBN, "a", FactorType.CKDE, 1)
    assert o.node == 'a'
    assert o.node_type == FactorType.CKDE
    assert o.delta == 1
    assert o.type == OperatorType.CHANGE_NODE_TYPE

def test_apply():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])
    assert gbn.num_edges() == 0
    assert not gbn.has_edge('a', 'b')

    o = AddArc(BayesianNetworkType.GBN, "a", "b", 1)
    o.apply(gbn)
    assert gbn.num_edges() == 1
    assert gbn.has_edge('a', 'b')
    
    o = FlipArc(BayesianNetworkType.GBN, "a", "b", 1)
    o.apply(gbn)
    assert gbn.num_edges() == 1
    assert not gbn.has_edge('a', 'b')
    assert gbn.has_edge('b', 'a')

    o = RemoveArc(BayesianNetworkType.GBN, "b", "a", 1)
    o.apply(gbn)
    assert gbn.num_edges() == 0
    assert not gbn.has_edge('b', 'a')

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    assert spbn.num_edges() == 0

    o = ChangeNodeType(BayesianNetworkType.SPBN, "a", FactorType.CKDE, 1)
    assert(spbn.node_type('a') == FactorType.LinearGaussianCPD)
    o.apply(spbn)
    assert(spbn.node_type('a') == FactorType.CKDE)

    assert not spbn.has_edge('a', 'b')
    o = AddArc(BayesianNetworkType.SPBN, "a", "b", 1)
    o.apply(spbn)
    assert spbn.num_edges() == 1
    assert spbn.has_edge('a', 'b')
    
    o = FlipArc(BayesianNetworkType.SPBN, "a", "b", 1)
    o.apply(spbn)
    assert spbn.num_edges() == 1
    assert not spbn.has_edge('a', 'b')
    assert spbn.has_edge('b', 'a')

    o = RemoveArc(BayesianNetworkType.SPBN, "b", "a", 1)
    o.apply(spbn)
    assert spbn.num_edges() == 0
    assert not spbn.has_edge('b', 'a')

def test_opposite():
    o = AddArc(BayesianNetworkType.GBN, "a", "b", 1)
    oppo = o.opposite()
    assert oppo.source == 'a'
    assert oppo.target == 'b'
    assert oppo.delta == -1
    assert oppo.type == OperatorType.REMOVE_ARC

    o = RemoveArc(BayesianNetworkType.GBN, "a", "b", 1)
    oppo = o.opposite()
    assert oppo.source == 'a'
    assert oppo.target == 'b'
    assert oppo.delta == -1
    assert oppo.type == OperatorType.ADD_ARC

    o = FlipArc(BayesianNetworkType.GBN, "a", "b", 1)
    oppo = o.opposite()
    assert oppo.source == 'b'
    assert oppo.target == 'a'
    assert oppo.delta == -1
    assert oppo.type == OperatorType.FLIP_ARC

    o = AddArc(BayesianNetworkType.SPBN, "a", "b", 1)
    oppo = o.opposite()
    assert oppo.source == 'a'
    assert oppo.target == 'b'
    assert oppo.delta == -1
    assert oppo.type == OperatorType.REMOVE_ARC

    o = RemoveArc(BayesianNetworkType.SPBN, "a", "b", 1)
    oppo = o.opposite()
    assert oppo.source == 'a'
    assert oppo.target == 'b'
    assert oppo.delta == -1
    assert oppo.type == OperatorType.ADD_ARC

    o = FlipArc(BayesianNetworkType.SPBN, "a", "b", 1)
    oppo = o.opposite()
    assert oppo.source == 'b'
    assert oppo.target == 'a'
    assert oppo.delta == -1
    assert oppo.type == OperatorType.FLIP_ARC

    o = ChangeNodeType(BayesianNetworkType.SPBN, "a", FactorType.CKDE, 1)
    oppo = o.opposite()
    assert oppo.node == 'a'
    assert oppo.node_type == FactorType.LinearGaussianCPD
    assert oppo.delta == -1
    assert oppo.type == OperatorType.CHANGE_NODE_TYPE