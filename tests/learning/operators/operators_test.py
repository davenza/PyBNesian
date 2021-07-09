import pytest
import pybnesian as pbn

def test_create():
    o = pbn.AddArc("a", "b", 1)
    assert o.source() == 'a'
    assert o.target() == 'b'
    assert o.delta() == 1

    o = pbn.RemoveArc("a", "b", 2)
    assert o.source() == 'a'
    assert o.target() == 'b'
    assert o.delta() == 2

    o = pbn.FlipArc("a", "b", 3)
    assert o.source() == 'a'
    assert o.target() == 'b'
    assert o.delta() == 3

    o = pbn.ChangeNodeType("a", pbn.CKDEType(), 4)
    assert o.node() == 'a'
    assert o.node_type() == pbn.CKDEType()
    assert o.delta() == 4

def test_apply():
    gbn = pbn.GaussianNetwork(['a', 'b', 'c', 'd'])
    assert gbn.num_arcs() == 0
    assert not gbn.has_arc('a', 'b')

    o = pbn.AddArc("a", "b", 1)
    o.apply(gbn)
    assert gbn.num_arcs() == 1
    assert gbn.has_arc('a', 'b')
    
    o = pbn.FlipArc("a", "b", 1)
    o.apply(gbn)
    assert gbn.num_arcs() == 1
    assert not gbn.has_arc('a', 'b')
    assert gbn.has_arc('b', 'a')

    o = pbn.RemoveArc("b", "a", 1)
    o.apply(gbn)
    assert gbn.num_arcs() == 0
    assert not gbn.has_arc('b', 'a')

    o = pbn.ChangeNodeType("a", pbn.CKDEType(), 1)
    with pytest.raises(ValueError) as ex:
        o.apply(gbn)
    assert "Wrong factor type" in str(ex.value)

    spbn = pbn.SemiparametricBN(['a', 'b', 'c', 'd'])
    assert spbn.num_arcs() == 0

    o = pbn.ChangeNodeType("a", pbn.CKDEType(), 1)
    assert(spbn.node_type('a') == pbn.UnknownFactorType())
    o.apply(spbn)
    assert(spbn.node_type('a') == pbn.CKDEType())

    assert not spbn.has_arc('a', 'b')
    o = pbn.AddArc("a", "b", 1)
    o.apply(spbn)
    assert spbn.num_arcs() == 1
    assert spbn.has_arc('a', 'b')
    
    o = pbn.FlipArc("a", "b", 1)
    o.apply(spbn)
    assert spbn.num_arcs() == 1
    assert not spbn.has_arc('a', 'b')
    assert spbn.has_arc('b', 'a')

    o = pbn.RemoveArc("b", "a", 1)
    o.apply(spbn)
    assert spbn.num_arcs() == 0
    assert not spbn.has_arc('b', 'a')

def test_opposite():
    bn = pbn.SemiparametricBN(["a", "b"])
    o = pbn.AddArc("a", "b", 1)
    oppo = o.opposite(bn)
    assert oppo.source() == 'a'
    assert oppo.target() == 'b'
    assert oppo.delta() == -1
    assert type(oppo) == pbn.RemoveArc

    o = pbn.RemoveArc("a", "b", 1)
    oppo = o.opposite(bn)
    assert oppo.source() == 'a'
    assert oppo.target() == 'b'
    assert oppo.delta() == -1
    assert type(oppo) == pbn.AddArc

    o = pbn.FlipArc("a", "b", 1)
    oppo = o.opposite(bn)
    assert oppo.source() == 'b'
    assert oppo.target() == 'a'
    assert oppo.delta() == -1
    assert type(oppo) == pbn.FlipArc

    bn.set_node_type("a", pbn.LinearGaussianCPDType())
    o = pbn.ChangeNodeType("a", pbn.CKDEType(), 1)
    oppo = o.opposite(bn)
    assert oppo.node() == 'a'
    assert oppo.node_type() == pbn.LinearGaussianCPDType()
    assert oppo.delta() == -1
    assert type(oppo) == pbn.ChangeNodeType