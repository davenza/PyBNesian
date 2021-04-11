import pytest
from pybnesian.factors.continuous import LinearGaussianCPDType, CKDEType
from pybnesian.models import GaussianNetwork, SemiparametricBN
from pybnesian.learning.operators import AddArc, RemoveArc, FlipArc, ChangeNodeType


def test_create():
    o = AddArc("a", "b", 1)
    assert o.source() == 'a'
    assert o.target() == 'b'
    assert o.delta() == 1

    o = RemoveArc("a", "b", 2)
    assert o.source() == 'a'
    assert o.target() == 'b'
    assert o.delta() == 2

    o = FlipArc("a", "b", 3)
    assert o.source() == 'a'
    assert o.target() == 'b'
    assert o.delta() == 3

    o = ChangeNodeType("a", CKDEType(), 4)
    assert o.node() == 'a'
    assert o.node_type() == CKDEType()
    assert o.delta() == 4

def test_apply():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'])
    assert gbn.num_arcs() == 0
    assert not gbn.has_arc('a', 'b')

    o = AddArc("a", "b", 1)
    o.apply(gbn)
    assert gbn.num_arcs() == 1
    assert gbn.has_arc('a', 'b')
    
    o = FlipArc("a", "b", 1)
    o.apply(gbn)
    assert gbn.num_arcs() == 1
    assert not gbn.has_arc('a', 'b')
    assert gbn.has_arc('b', 'a')

    o = RemoveArc("b", "a", 1)
    o.apply(gbn)
    assert gbn.num_arcs() == 0
    assert not gbn.has_arc('b', 'a')

    o = ChangeNodeType("a", CKDEType(), 1)
    with pytest.raises(ValueError) as ex:
        o.apply(gbn)
    assert "Wrong factor type" in str(ex.value)

    spbn = SemiparametricBN(['a', 'b', 'c', 'd'])
    assert spbn.num_arcs() == 0

    o = ChangeNodeType("a", CKDEType(), 1)
    assert(spbn.node_type('a') == LinearGaussianCPDType())
    o.apply(spbn)
    assert(spbn.node_type('a') == CKDEType())

    assert not spbn.has_arc('a', 'b')
    o = AddArc("a", "b", 1)
    o.apply(spbn)
    assert spbn.num_arcs() == 1
    assert spbn.has_arc('a', 'b')
    
    o = FlipArc("a", "b", 1)
    o.apply(spbn)
    assert spbn.num_arcs() == 1
    assert not spbn.has_arc('a', 'b')
    assert spbn.has_arc('b', 'a')

    o = RemoveArc("b", "a", 1)
    o.apply(spbn)
    assert spbn.num_arcs() == 0
    assert not spbn.has_arc('b', 'a')

def test_opposite():
    o = AddArc("a", "b", 1)
    oppo = o.opposite()
    assert oppo.source() == 'a'
    assert oppo.target() == 'b'
    assert oppo.delta() == -1
    assert type(oppo) == RemoveArc

    o = RemoveArc("a", "b", 1)
    oppo = o.opposite()
    assert oppo.source() == 'a'
    assert oppo.target() == 'b'
    assert oppo.delta() == -1
    assert type(oppo) == AddArc

    o = FlipArc("a", "b", 1)
    oppo = o.opposite()
    assert oppo.source() == 'b'
    assert oppo.target() == 'a'
    assert oppo.delta() == -1
    assert type(oppo) == FlipArc

    o = ChangeNodeType("a", CKDEType(), 1)
    oppo = o.opposite()
    assert oppo.node() == 'a'
    assert oppo.node_type() == LinearGaussianCPDType()
    assert oppo.delta() == -1
    assert type(oppo) == ChangeNodeType