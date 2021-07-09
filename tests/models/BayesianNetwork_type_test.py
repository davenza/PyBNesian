import pybnesian as pbn
from pybnesian import BayesianNetworkType, BayesianNetwork, ConditionalBayesianNetwork, GaussianNetwork,\
    SemiparametricBN, KDENetwork, DiscreteBN
import util_test

def test_bn_type():
    g1 = GaussianNetwork(["a", "b", "c", "d"])
    g2 = GaussianNetwork(["a", "b", "c", "d"])
    g3 = GaussianNetwork(["a", "b", "c", "d"])

    assert g1.type() == pbn.GaussianNetworkType()
    assert g1.type() == g2.type()
    assert g1.type() == g3.type()
    assert g2.type() == g3.type()

    s1 = SemiparametricBN(["a", "b", "c", "d"])
    s2 = SemiparametricBN(["a", "b", "c", "d"])
    s3 = SemiparametricBN(["a", "b", "c", "d"])

    assert s1.type() == pbn.SemiparametricBNType()
    assert s1.type() == s2.type()
    assert s1.type() == s3.type()
    assert s2.type() == s3.type()

    k1 = KDENetwork(["a", "b", "c", "d"])
    k2 = KDENetwork(["a", "b", "c", "d"])
    k3 = KDENetwork(["a", "b", "c", "d"])

    assert k1.type() == pbn.KDENetworkType()
    assert k1.type() == k2.type()
    assert k1.type() == k3.type()
    assert k2.type() == k3.type()

    d1 = DiscreteBN(["a", "b", "c", "d"])
    d2 = DiscreteBN(["a", "b", "c", "d"])
    d3 = DiscreteBN(["a", "b", "c", "d"])

    assert d1.type() == pbn.DiscreteBNType()
    assert d1.type() == d2.type()
    assert d1.type() == d3.type()
    assert d2.type() == d3.type()

    assert g1.type() != s1.type()
    assert g1.type() != k1.type()
    assert g1.type() != d1.type()
    assert s1.type() != k1.type()
    assert s1.type() != d1.type()
    assert k1.type() != d1.type()

def test_new_bn_type():
    class MyGaussianNetworkType(BayesianNetworkType):
        def __init__(self):
            BayesianNetworkType.__init__(self)

        def is_homogeneous(self):
            return True

        def can_have_arc(self, model, source, target):
            return source == "a"

    a1 = MyGaussianNetworkType()
    a2 = MyGaussianNetworkType()
    a3 = MyGaussianNetworkType()

    assert a1 == a2
    assert a1 == a3
    assert a2 == a3

    class MySemiparametricBNType(BayesianNetworkType):
        def __init__(self):
            BayesianNetworkType.__init__(self)
    
    b1 = MySemiparametricBNType()
    b2 = MySemiparametricBNType()
    b3 = MySemiparametricBNType()

    assert b1 == b2
    assert b1 == b3
    assert b2 == b3

    assert a1 != b1

    mybn = BayesianNetwork(a1, ["a", "b", "c", "d"])

    # This type omits the arcs that do not have "a" as source.
    assert mybn.can_add_arc("a", "b")
    assert not mybn.can_add_arc("b", "a")
    assert not mybn.can_add_arc("c", "d")


class MyRestrictedGaussianNetworkType(BayesianNetworkType):
    def __init__(self):
        BayesianNetworkType.__init__(self)

    def is_homogeneous(self):
        return True

    def default_node_type(self):
        return pbn.LinearGaussianCPDType()

    def can_have_arc(self, model, source, target):
        return source == "a"

    def __str__(self):
        return "MyRestrictedGaussianNetworkType"

class SpecificNetwork(BayesianNetwork):
    def __init__(self, variables, arcs=None):
        if arcs is None:
            BayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables)
        else:
            BayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables, arcs)

class ConditionalSpecificNetwork(ConditionalBayesianNetwork):
    def __init__(self, variables, interface, arcs=None):
        if arcs is None:
            ConditionalBayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables, interface)
        else:
            ConditionalBayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables, interface, arcs)

def test_new_specific_bn_type():
    sp1 = SpecificNetwork(["a", "b", "c", "d"])
    sp2 = SpecificNetwork(["a", "b", "c", "d"], [("a", "b")])
    sp3 = SpecificNetwork(["a", "b", "c", "d"])

    assert sp1.type() == sp2.type()
    assert sp1.type() == sp3.type()
    assert sp2.type() == sp3.type()

    assert sp1.can_add_arc("a", "b")
    assert not sp1.can_add_arc("b", "a")
    assert not sp1.can_add_arc("c", "d")

    assert sp1.num_arcs() == sp3.num_arcs() == 0
    assert sp2.arcs() == [("a", "b")]

    df = util_test.generate_normal_data_indep(1000)
    bic = pbn.BIC(df)

    start = SpecificNetwork(["a", "b", "c", "d"])

    hc = pbn.GreedyHillClimbing()
    estimated = hc.estimate(pbn.ArcOperatorSet(), bic, start)
    assert estimated.type() == start.type()
    assert all([s == "a" for s, t in estimated.arcs()])

    # #######################
    # Conditional BN
    # #######################
 
    csp1 = ConditionalSpecificNetwork(["a", "b"], ["c", "d"])
    csp2 = ConditionalSpecificNetwork(["a", "b"], ["c", "d"], [("a", "b")])
    csp3 = ConditionalSpecificNetwork(["a", "b"], ["c", "d"])

    assert csp1.type() == csp2.type()
    assert csp1.type() == csp3.type()
    assert csp2.type() == csp3.type()

    assert csp1.can_add_arc("a", "b")
    assert not csp1.can_add_arc("b", "a")
    assert not csp1.can_add_arc("c", "d")

    assert csp1.num_arcs() == csp3.num_arcs() == 0
    assert csp2.arcs() == [("a", "b")]

    cstart = ConditionalSpecificNetwork(["a", "c"], ["b", "d"])

    hc = pbn.GreedyHillClimbing()
    cestimated = hc.estimate(pbn.ArcOperatorSet(), bic, cstart)
    assert cestimated.type() == cstart.type()
    assert all([s == "a" for s, t in cestimated.arcs()])
