import pytest
import pybnesian as pbn
from pybnesian import FactorType, Factor

def test_factor_type():
    lg1 = pbn.LinearGaussianCPD("a", [])
    lg2 = pbn.LinearGaussianCPD("b", ["a"])
    lg3 = pbn.LinearGaussianCPD("c", ["b", "a"])

    assert lg1.type() == pbn.LinearGaussianCPDType()
    assert lg1.type() == lg2.type()
    assert lg1.type() == lg3.type()
    assert lg2.type() == lg3.type()

    c1 = pbn.CKDE("a", [])
    c2 = pbn.CKDE("b", ["a"])
    c3 = pbn.CKDE("c", ["b", "a"])

    assert c1.type() == pbn.CKDEType()
    assert c1.type() == c2.type()
    assert c1.type() == c3.type()
    assert c2.type() == c3.type()

    d1 = pbn.DiscreteFactor("a", [])
    d2 = pbn.DiscreteFactor("b", ["a"])
    d3 = pbn.DiscreteFactor("c", ["b", "a"])

    assert d1.type() == pbn.DiscreteFactorType()
    assert d1.type() == d2.type()
    assert d1.type() == d3.type()
    assert d2.type() == d3.type()

    assert lg1.type() != c1.type()
    assert lg1.type() != d1.type()
    assert c1.type() != d1.type()

def test_new_factor_type():
    class A(FactorType):
        def __init__(self):
            FactorType.__init__(self)

    a1 = A()
    a2 = A()
    a3 = A()

    assert a1 == a2
    assert a1 == a3
    assert a2 == a3

    class B(FactorType):
        def __init__(self):
            FactorType.__init__(self)

    b1 = B()
    b2 = B()
    b3 = B()

    assert b1 == b2
    assert b1 == b3
    assert b2 == b3

    assert a1 != b1

def test_factor_defined_factor_type():
    class F_type(FactorType):
        def __init__(self):
            FactorType.__init__(self)

        def __str__(self):
            return "FType"

    class F(Factor):
        def __init__(self, variable, evidence):
            Factor.__init__(self, variable, evidence)

        def type(self):
            return F_type()

    f1 = F("a", [])
    f2 = F("b", ["a"])
    f3 = F("c", ["a", "b"])

    assert f1.type() == f2.type()
    assert f1.type() == f3.type()
    assert f2.type() == f3.type()

    assert str(f1.type()) == str(f2.type()) == str(f3.type()) == "FType"

    dummy_network = pbn.GaussianNetwork(["a", "b", "c", "d"])
    with pytest.raises(RuntimeError) as ex:
        f4 = f1.type().new_factor(dummy_network, "d", ["a", "b", "c"])
    assert 'Tried to call pure virtual function "FactorType::new_factor"' in str(ex.value)

    class G_type(FactorType):
        def __init__(self):
            FactorType.__init__(self)
            
        def new_factor(self, model, variable, evidence):
            return G(variable, evidence)

        def __str__(self):
            return "GType"

    class G(Factor):
        def __init__(self, variable, evidence):
            Factor.__init__(self, variable, evidence)

        def type(self):
            return G_type()

    g1 = G("a", [])
    g2 = G("b", ["a"])
    g3 = G("c", ["a", "b"])

    assert g1.type() == g2.type()
    assert g1.type() == g3.type()
    assert g2.type() == g3.type()

    assert f1.type() != g1.type()

    assert str(g1.type()) == str(g2.type()) == str(g3.type()) == "GType"

    g4 = g1.type().new_factor(dummy_network, "d", ["a", "b", "c"])

    assert g1.type() == g4.type()
    assert g4.variable() == "d"
    assert g4.evidence() == ["a", "b", "c"]