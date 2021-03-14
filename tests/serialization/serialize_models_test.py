import pytest
from pybnesian.factors.continuous import *
from pybnesian.factors.discrete import *
from pybnesian.models import *
import pickle
import util_test

@pytest.fixture
def gaussian_bytes():
    gaussian = GaussianNetwork(["a", "b", "c", "d"], [("a", "b")])
    return pickle.dumps(gaussian)

@pytest.fixture
def spbn_bytes():
    spbn = SemiparametricBN(["a", "b", "c", "d"], [("a", "b")], [("b", CKDEType())])
    return pickle.dumps(spbn)

@pytest.fixture
def kde_bytes():
    kde = KDENetwork(["a", "b", "c", "d"], [("a", "b")])
    return pickle.dumps(kde)

@pytest.fixture
def discrete_bytes():
    discrete = DiscreteBN(["a", "b", "c", "d"], [("a", "b")])
    return pickle.dumps(discrete)

class MyRestrictedGaussianNetworkType(BayesianNetworkType):
    def __init__(self):
        BayesianNetworkType.__init__(self)

    def is_homogeneous(self):
        return True

    def default_node_type(self):
        return LinearGaussianCPDType()

    def can_add_arc(self, model, source, target):
        return "a" in source

    def can_flip_arc(self, model, source, target):
        return not "a" in target

    def ToString(self):
        return "MyRestrictedGaussianNetworkType"

@pytest.fixture
def genericbn_bytes():
    gen = BayesianNetwork(MyRestrictedGaussianNetworkType(), ["a", "b", "c", "d"], [("a", "b")])
    return pickle.dumps(gen)

class NewBN(BayesianNetwork):
    def __init__(self, variables, arcs=None):
        if arcs is None:
            BayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables)
        else:
            BayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables, arcs)

@pytest.fixture
def newbn_bytes():
    new = NewBN(["a", "b", "c", "d"], [("a", "b")])
    return pickle.dumps(new)

class NonHomogeneousType(BayesianNetworkType):
    def __init__(self):
        BayesianNetworkType.__init__(self)
    
    def is_homogeneous(self):
        return False

    def default_node_type(self):
        return LinearGaussianCPDType()

    def ToString(self):
        return "NonHomogeneousType"


class OtherBN(BayesianNetwork):
    def __init__(self, variables, arcs=None, node_types=None):
        if arcs is None:
            if node_types is None:
                BayesianNetwork.__init__(self, NonHomogeneousType(), variables)
            else:
                BayesianNetwork.__init__(self, NonHomogeneousType(), variables, node_types)
        else:
            if node_types is None:
                BayesianNetwork.__init__(self, NonHomogeneousType(), variables, arcs)
            else:
                BayesianNetwork.__init__(self, NonHomogeneousType(), variables, arcs, node_types)

        self.extra_info = "extra"

    def __getstate_extra__(self):
        return self.extra_info

    def __setstate_extra__(self, t):
        self.extra_info = t

@pytest.fixture
def otherbn_bytes():
    other = OtherBN(["a", "b", "c", "d"], [("a", "b")], [("b", LinearGaussianCPDType()),
                                                         ("c", CKDEType()),
                                                         ("d", DiscreteFactorType())])
    return pickle.dumps(other)

def test_serialization_bn_model(gaussian_bytes, spbn_bytes, kde_bytes, discrete_bytes, genericbn_bytes, newbn_bytes, otherbn_bytes):
    loaded_g = pickle.loads(gaussian_bytes)
    assert set(loaded_g.nodes()) == set(["a", "b", "c", "d"])
    assert loaded_g.arcs() == [("a", "b")]
    assert loaded_g.type() == GaussianNetworkType()

    loaded_s = pickle.loads(spbn_bytes)
    assert set(loaded_s.nodes()) == set(["a", "b", "c", "d"])
    assert loaded_s.arcs() == [("a", "b")]
    assert loaded_s.type() == SemiparametricBNType()
    assert loaded_s.node_types() == {'a': LinearGaussianCPDType(),
                                     'b': CKDEType(),
                                     'c': LinearGaussianCPDType(),
                                     'd': LinearGaussianCPDType()}

    loaded_k = pickle.loads(kde_bytes)
    assert set(loaded_k.nodes()) == set(["a", "b", "c", "d"])
    assert loaded_k.arcs() == [("a", "b")]
    assert loaded_k.type() == KDENetworkType()

    loaded_d = pickle.loads(discrete_bytes)
    assert set(loaded_d.nodes()) == set(["a", "b", "c", "d"])
    assert loaded_d.arcs() == [("a", "b")]
    assert loaded_d.type() == DiscreteBNType()

    loaded_gen = pickle.loads(genericbn_bytes)
    assert set(loaded_gen.nodes()) == set(["a", "b", "c", "d"])
    assert loaded_gen.arcs() == [("a", "b")]
    assert loaded_gen.type() == MyRestrictedGaussianNetworkType()

    loaded_nn = pickle.loads(newbn_bytes)
    assert set(loaded_g.nodes()) == set(["a", "b", "c", "d"])
    assert loaded_nn.arcs() == [("a", "b")]
    assert loaded_nn.type() == MyRestrictedGaussianNetworkType()

    loaded_o = pickle.loads(otherbn_bytes)
    assert set(loaded_g.nodes()) == set(["a", "b", "c", "d"])
    assert loaded_o.arcs() == [("a", "b")]
    assert loaded_o.type() == NonHomogeneousType()
    assert loaded_o.node_types() == {'a': LinearGaussianCPDType(),
                                     'b': LinearGaussianCPDType(),
                                     'c': CKDEType(),
                                     'd': DiscreteFactorType()}
    assert loaded_o.extra_info == "extra"

    assert loaded_nn.type() != loaded_o.type()

@pytest.fixture
def gaussian_partial_fit_bytes():
    gaussian = GaussianNetwork(["a", "b", "c", "d"], [("a", "b")])
    lg = LinearGaussianCPD("b", ["a"], [1, 2], 2)
    gaussian.add_cpds([lg])
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)

@pytest.fixture
def gaussian_fit_bytes():
    gaussian = GaussianNetwork(["a", "b", "c", "d"], [("a", "b")])
    lg_a = LinearGaussianCPD("a", [], [0], 0.5)
    lg_b = LinearGaussianCPD("b", ["a"], [1, 2], 2)
    lg_c = LinearGaussianCPD("c", [], [2], 1)
    lg_d = LinearGaussianCPD("d", [], [3], 1.5)
    gaussian.add_cpds([lg_a, lg_b, lg_c, lg_d])
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)

@pytest.fixture
def other_partial_fit_bytes():
    other = OtherBN(["a", "b", "c", "d"], [("a", "b")], [("b", LinearGaussianCPDType()),
                                                         ("c", CKDEType()),
                                                         ("d", DiscreteFactorType())])
    lg = LinearGaussianCPD("b", ["a"], [1, 2], 2)
    other.add_cpds([lg])
    other.include_cpd = True
    return pickle.dumps(other)

@pytest.fixture
def other_fit_bytes():
    other = OtherBN(["a", "b", "c", "d"], [("a", "b")], [("b", LinearGaussianCPDType()),
                                                         ("c", CKDEType()),
                                                         ("d", DiscreteFactorType())])
    cpd_a = LinearGaussianCPD("a", [], [0], 0.5)
    cpd_b = LinearGaussianCPD("b", ["a"], [1, 2], 2)
    
    df_continuous = util_test.generate_normal_data_indep(100)
    cpd_c = CKDE("c", [])
    cpd_c.fit(df_continuous)

    df_discrete = util_test.generate_discrete_data_dependent(100)
    df_discrete.columns = df_discrete.columns.str.lower()
    cpd_d = DiscreteFactor("d", [])
    cpd_d.fit(df_discrete)
    
    other.add_cpds([cpd_a, cpd_b, cpd_c, cpd_d])

    other.include_cpd = True
    return pickle.dumps(other)

def test_serialization_fitted_bn(gaussian_partial_fit_bytes, gaussian_fit_bytes, other_partial_fit_bytes, other_fit_bytes):
    # ####################
    # Gaussian partial fit
    # ####################
    loaded_partial = pickle.loads(gaussian_partial_fit_bytes)
    assert not loaded_partial.fitted
    cpd = loaded_partial.cpd("b")
    assert cpd.variable == "b"
    assert cpd.evidence == ["a"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    # ####################
    # Gaussian fit
    # ####################
    loaded_fitted = pickle.loads(gaussian_fit_bytes)
    assert loaded_fitted.fitted

    cpd_a = loaded_fitted.cpd("a")
    assert cpd_a.variable == "a"
    assert cpd_a.evidence == []
    assert cpd_a.beta == [0]
    assert cpd_a.variance == 0.5

    cpd_b = loaded_fitted.cpd("b")
    assert cpd_b.variable == "b"
    assert cpd_b.evidence == ["a"]
    assert list(cpd_b.beta) == [1, 2]
    assert cpd_b.variance == 2

    cpd_c = loaded_fitted.cpd("c")
    assert cpd_c.variable == "c"
    assert cpd_c.evidence == []
    assert cpd_c.beta == [2]
    assert cpd_c.variance == 1
    
    cpd_d = loaded_fitted.cpd("d")
    assert cpd_d.variable == "d"
    assert cpd_d.evidence == []
    assert cpd_d.beta == [3]
    assert cpd_d.variance == 1.5

    # ####################
    # OtherBN homogeneous partial fit
    # ####################
    loaded_other = pickle.loads(other_partial_fit_bytes)
    assert not loaded_other.fitted
    cpd = loaded_partial.cpd("b")
    assert cpd.variable == "b"
    assert cpd.evidence == ["a"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    # ####################
    # OtherBN homogeneous fit
    # ####################
    loaded_other_fitted = pickle.loads(other_fit_bytes)
    assert loaded_other_fitted.fitted

    cpd_a = loaded_other_fitted.cpd("a")
    assert cpd_a.variable == "a"
    assert cpd_a.evidence == []
    assert cpd_a.beta == [0]
    assert cpd_a.variance == 0.5
    assert cpd_a.type() == LinearGaussianCPDType()

    cpd_b = loaded_other_fitted.cpd("b")
    assert cpd_b.variable == "b"
    assert cpd_b.evidence == ["a"]
    assert list(cpd_b.beta) == [1, 2]
    assert cpd_b.variance == 2
    assert cpd_b.type() == LinearGaussianCPDType()

    cpd_c = loaded_other_fitted.cpd("c")
    assert cpd_c.variable == "c"
    assert cpd_c.evidence == []
    assert cpd_c.fitted
    assert cpd_c.N == 100
    assert cpd_c.type() == CKDEType()

    cpd_d = loaded_other_fitted.cpd("d")
    assert cpd_d.variable == "d"
    assert cpd_d.evidence == []
    assert cpd_d.fitted
    assert cpd_d.type() == DiscreteFactorType()


# ##########################
# Conditional BN
# ##########################

@pytest.fixture
def cond_gaussian_bytes():
    gaussian = ConditionalGaussianNetwork(["c", "d"], ["a", "b"], [("a", "c")])
    return pickle.dumps(gaussian)

@pytest.fixture
def cond_spbn_bytes():
    spbn = ConditionalSemiparametricBN(["c", "d"], ["a", "b"], [("a", "c")], [("c", CKDEType())])
    return pickle.dumps(spbn)

@pytest.fixture
def cond_kde_bytes():
    kde = ConditionalKDENetwork(["c", "d"], ["a", "b"], [("a", "c")])
    return pickle.dumps(kde)

@pytest.fixture
def cond_discrete_bytes():
    discrete = ConditionalDiscreteBN(["c", "d"], ["a", "b"], [("a", "c")])
    return pickle.dumps(discrete)

@pytest.fixture
def cond_genericbn_bytes():
    gen = ConditionalBayesianNetwork(MyRestrictedGaussianNetworkType(), ["c", "d"], ["a", "b"], [("a", "c")])
    return pickle.dumps(gen)

class ConditionalNewBN(ConditionalBayesianNetwork):
    def __init__(self, variables, interface, arcs=None):
        if arcs is None:
            ConditionalBayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables, interface)
        else:
            ConditionalBayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables, interface, arcs)

@pytest.fixture
def cond_newbn_bytes():
    new = ConditionalNewBN(["c", "d"], ["a", "b"], [("a", "c")])
    return pickle.dumps(new)

class ConditionalOtherBN(ConditionalBayesianNetwork):
    def __init__(self, variables, interface, arcs=None, node_types=None):
        if arcs is None:
            if node_types is None:
                ConditionalBayesianNetwork.__init__(self, NonHomogeneousType(), variables, interface)
            else:
                ConditionalBayesianNetwork.__init__(self, NonHomogeneousType(), variables, interface, node_types)
        else:
            if node_types is None:
                ConditionalBayesianNetwork.__init__(self, NonHomogeneousType(), variables, interface, arcs)
            else:
                ConditionalBayesianNetwork.__init__(self, NonHomogeneousType(), variables, interface, arcs, node_types)

        self.extra_info = "extra"

    def __getstate_extra__(self):
        return self.extra_info

    def __setstate_extra__(self, t):
        self.extra_info = t

@pytest.fixture
def cond_otherbn_bytes():
    other = ConditionalOtherBN(["c", "d"], ["a", "b"], [("a", "c")], [("b", LinearGaussianCPDType()),
                                                                      ("c", CKDEType()),
                                                                      ("d", DiscreteFactorType())])
    return pickle.dumps(other)



def test_serialization_conditional_bn_model(cond_gaussian_bytes, cond_spbn_bytes, cond_kde_bytes, 
                                            cond_discrete_bytes, cond_genericbn_bytes,
                                            cond_newbn_bytes, cond_otherbn_bytes,
                                            newbn_bytes, otherbn_bytes):
    loaded_g = pickle.loads(cond_gaussian_bytes)
    assert set(loaded_g.nodes()) == set(["c", "d"])
    assert set(loaded_g.interface_nodes()) == set(["a", "b"])
    assert loaded_g.arcs() == [("a", "c")]
    assert loaded_g.type() == GaussianNetworkType()

    loaded_s = pickle.loads(cond_spbn_bytes)
    assert set(loaded_s.nodes()) == set(["c", "d"])
    assert set(loaded_s.interface_nodes()) == set(["a", "b"])
    assert loaded_s.arcs() == [("a", "c")]
    assert loaded_s.type() == SemiparametricBNType()
    assert loaded_s.node_types() == {'c': CKDEType(),
                                     'd': LinearGaussianCPDType()}

    loaded_k = pickle.loads(cond_kde_bytes)
    assert set(loaded_k.nodes()) == set(["c", "d"])
    assert set(loaded_k.interface_nodes()) == set(["a", "b"])
    assert loaded_k.arcs() == [("a", "c")]
    assert loaded_k.type() == KDENetworkType()

    loaded_d = pickle.loads(cond_discrete_bytes)
    assert set(loaded_d.nodes()) == set(["c", "d"])
    assert set(loaded_d.interface_nodes()) == set(["a", "b"])
    assert loaded_d.arcs() == [("a", "c")]
    assert loaded_d.type() == DiscreteBNType()

    loaded_gen = pickle.loads(cond_genericbn_bytes)
    assert set(loaded_gen.nodes()) == set(["c", "d"])
    assert set(loaded_gen.interface_nodes()) == set(["a", "b"])
    assert loaded_gen.arcs() == [("a", "c")]
    assert loaded_gen.type() == MyRestrictedGaussianNetworkType()

    loaded_nn = pickle.loads(cond_newbn_bytes)
    assert set(loaded_nn.nodes()) == set(["c", "d"])
    assert set(loaded_nn.interface_nodes()) == set(["a", "b"])
    assert loaded_nn.arcs() == [("a", "c")]
    assert loaded_nn.type() == MyRestrictedGaussianNetworkType()

    loaded_o = pickle.loads(cond_otherbn_bytes)
    assert set(loaded_o.nodes()) == set(["c", "d"])
    assert set(loaded_o.interface_nodes()) == set(["a", "b"])
    assert loaded_o.arcs() == [("a", "c")]
    assert loaded_o.type() == NonHomogeneousType()
    assert loaded_o.node_types() == {'c': CKDEType(),
                                     'd': DiscreteFactorType()}
    assert loaded_o.extra_info == "extra"

    assert loaded_nn.type() != loaded_o.type()

    loaded_unconditional_nn = pickle.loads(newbn_bytes)
    loaded_unconditional_o = pickle.loads(otherbn_bytes)

    assert loaded_nn.type() == loaded_unconditional_nn.type()
    assert loaded_o.type() == loaded_unconditional_o.type()

@pytest.fixture
def cond_gaussian_partial_fit_bytes():
    gaussian = ConditionalGaussianNetwork(["c", "d"], ["a", "b"], [("a", "c")])
    lg = LinearGaussianCPD("c", ["a"], [1, 2], 2)
    gaussian.add_cpds([lg])
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)

@pytest.fixture
def cond_gaussian_fit_bytes():
    gaussian = ConditionalGaussianNetwork(["c", "d"], ["a", "b"], [("a", "c")])
    lg_c = LinearGaussianCPD("c", ["a"], [1, 2], 2)
    lg_d = LinearGaussianCPD("d", [], [3], 1.5)
    gaussian.add_cpds([lg_c, lg_d])
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)

@pytest.fixture
def cond_other_partial_fit_bytes():
    other = ConditionalOtherBN(["c", "d"], ["a", "b"], [("a", "c")], [("c", CKDEType()),
                                                                      ("d", LinearGaussianCPDType())])
    lg = LinearGaussianCPD("d", [], [3], 1.5)
    other.add_cpds([lg])
    other.include_cpd = True
    return pickle.dumps(other)

@pytest.fixture
def cond_other_fit_bytes():
    other = ConditionalOtherBN(["c", "d"], ["a", "b"], [("a", "c")], [("c", CKDEType()),
                                                                      ("d", DiscreteFactorType())])
    cpd_c = CKDE("c", ["a"])
    cpd_d = DiscreteFactor("d", [])
    
    df_continuous = util_test.generate_normal_data_indep(100)
    cpd_c.fit(df_continuous)

    df_discrete = util_test.generate_discrete_data_dependent(100)
    df_discrete.columns = df_discrete.columns.str.lower()
    cpd_d = DiscreteFactor("d", [])
    cpd_d.fit(df_discrete)

    other.add_cpds([cpd_c, cpd_d])
    
    other.include_cpd = True
    return pickle.dumps(other)

def test_serialization_fitted_conditional_bn(cond_gaussian_partial_fit_bytes, cond_gaussian_fit_bytes,
                                             cond_other_partial_fit_bytes, cond_other_fit_bytes):
    # ####################
    # Gaussian partial fit
    # ####################
    loaded_partial = pickle.loads(cond_gaussian_partial_fit_bytes)
    assert not loaded_partial.fitted
    cpd = loaded_partial.cpd("c")
    assert cpd.variable == "c"
    assert cpd.evidence == ["a"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    # ####################
    # Gaussian fit
    # ####################
    loaded_fitted = pickle.loads(cond_gaussian_fit_bytes)
    assert loaded_fitted.fitted

    cpd_c = loaded_fitted.cpd("c")
    assert cpd_c.variable == "c"
    assert cpd_c.evidence == ["a"]
    assert list(cpd_c.beta) == [1, 2]
    assert cpd_c.variance == 2
    
    cpd_d = loaded_fitted.cpd("d")
    assert cpd_d.variable == "d"
    assert cpd_d.evidence == []
    assert cpd_d.beta == [3]
    assert cpd_d.variance == 1.5

    # ####################
    # OtherBN homogeneous partial fit
    # ####################
    loaded_other = pickle.loads(cond_other_partial_fit_bytes)
    assert not loaded_other.fitted
    cpd = loaded_other.cpd("d")
    assert cpd.variable == "d"
    assert cpd.evidence == []
    assert cpd.beta == [3]
    assert cpd.variance == 1.5

    # ####################
    # OtherBN homogeneous fit
    # ####################
    loaded_other_fitted = pickle.loads(cond_other_fit_bytes)
    assert loaded_other_fitted.fitted

    cpd_c = loaded_other_fitted.cpd("c")
    assert cpd_c.variable == "c"
    assert cpd_c.evidence == ["a"]
    assert cpd_c.fitted
    assert cpd_c.N == 100
    assert cpd_c.type() == CKDEType()

    cpd_d = loaded_other_fitted.cpd("d")
    assert cpd_d.variable == "d"
    assert cpd_d.evidence == []
    assert cpd_d.fitted
    assert cpd_d.type() == DiscreteFactorType()

    assert loaded_other_fitted.extra_info == "extra"
    assert loaded_other.type() == loaded_other_fitted.type()

# ##########################
# Dynamic BN
# ##########################

@pytest.fixture
def dyn_gaussian_bytes():
    gaussian = DynamicGaussianNetwork(["a", "b", "c", "d"], 2)
    gaussian.static_bn().add_arc("a_t_2", "d_t_1")
    gaussian.transition_bn().add_arc("c_t_2", "b_t_0")
    return pickle.dumps(gaussian)

@pytest.fixture
def dyn_spbn_bytes():
    spbn = DynamicSemiparametricBN(["a", "b", "c", "d"], 2)
    spbn.static_bn().add_arc("a_t_2", "d_t_1")
    spbn.transition_bn().add_arc("c_t_2", "b_t_0")
    spbn.transition_bn().set_node_type("b_t_0", CKDEType())
    return pickle.dumps(spbn)

@pytest.fixture
def dyn_kde_bytes():
    kde = DynamicKDENetwork(["a", "b", "c", "d"], 2)
    kde.static_bn().add_arc("a_t_2", "d_t_1")
    kde.transition_bn().add_arc("c_t_2", "b_t_0")
    return pickle.dumps(kde)

@pytest.fixture
def dyn_discrete_bytes():
    discrete = DynamicDiscreteBN(["a", "b", "c", "d"], 2)
    discrete.static_bn().add_arc("a_t_2", "d_t_1")
    discrete.transition_bn().add_arc("c_t_2", "b_t_0")
    return pickle.dumps(discrete)

@pytest.fixture
def dyn_genericbn_bytes():
    gen = DynamicBayesianNetwork(MyRestrictedGaussianNetworkType(), ["a", "b", "c", "d"], 2)
    gen.static_bn().add_arc("a_t_2", "d_t_1")
    gen.transition_bn().add_arc("a_t_2", "b_t_0")
    return pickle.dumps(gen)

class DynamicNewBN(DynamicBayesianNetwork):
    def __init__(self, variables, markovian_order):
        DynamicBayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables, markovian_order)

class DynamicOtherBN(DynamicBayesianNetwork):
    def __init__(self, variables, markovian_order, static_bn=None, transition_bn=None):
        if static_bn is None or transition_bn is None:
            DynamicBayesianNetwork.__init__(self, NonHomogeneousType(), variables, markovian_order)
        else:
            DynamicBayesianNetwork.__init__(self, variables, markovian_order, static_bn, transition_bn)
        self.extra_info = "extra"

    def __getstate_extra__(self):
        return self.extra_info

    def __setstate_extra__(self, t):
        self.extra_info = t

@pytest.fixture
def dyn_newbn_bytes():
    new = DynamicNewBN(["a", "b", "c", "d"], 2)
    new.static_bn().add_arc("a_t_2", "d_t_1")
    new.transition_bn().add_arc("a_t_2", "b_t_0")
    return pickle.dumps(new)

@pytest.fixture
def dyn_otherbn_bytes():
    other = DynamicOtherBN(["a", "b", "c", "d"], 2)
    other.static_bn().add_arc("a_t_2", "d_t_1")
    other.static_bn().set_node_type("c_t_1", DiscreteFactorType())
    other.static_bn().set_node_type("d_t_1", CKDEType())

    other.transition_bn().add_arc("a_t_2", "b_t_0")
    other.transition_bn().set_node_type("d_t_0", CKDEType())
    return pickle.dumps(other)

def test_serialization_dbn_model(dyn_gaussian_bytes, dyn_spbn_bytes, dyn_kde_bytes, dyn_discrete_bytes,
                                dyn_genericbn_bytes, dyn_newbn_bytes, dyn_otherbn_bytes):
    loaded_g = pickle.loads(dyn_gaussian_bytes)
    assert set(loaded_g.variables()) == set(["a", "b", "c", "d"])
    assert loaded_g.static_bn().arcs() == [("a_t_2", "d_t_1")]
    assert loaded_g.transition_bn().arcs() == [("c_t_2", "b_t_0")]
    assert loaded_g.type() == GaussianNetworkType()

    loaded_s = pickle.loads(dyn_spbn_bytes)
    assert set(loaded_s.variables()) == set(["a", "b", "c", "d"])
    assert loaded_s.static_bn().arcs() == [("a_t_2", "d_t_1")]
    assert loaded_s.transition_bn().arcs() == [("c_t_2", "b_t_0")]
    assert loaded_s.type() == SemiparametricBNType()
    node_types = {v + "_t_0": LinearGaussianCPDType() for v in loaded_s.variables()}
    node_types["b_t_0"] = CKDEType()
    assert loaded_s.transition_bn().node_types() == node_types

    loaded_k = pickle.loads(dyn_kde_bytes)
    assert set(loaded_k.variables()) == set(["a", "b", "c", "d"])
    assert loaded_k.static_bn().arcs() == [("a_t_2", "d_t_1")]
    assert loaded_k.transition_bn().arcs() == [("c_t_2", "b_t_0")]
    assert loaded_k.type() == KDENetworkType()

    loaded_d = pickle.loads(dyn_discrete_bytes)
    assert set(loaded_d.variables()) == set(["a", "b", "c", "d"])
    assert loaded_d.static_bn().arcs() == [("a_t_2", "d_t_1")]
    assert loaded_d.transition_bn().arcs() == [("c_t_2", "b_t_0")]
    assert loaded_d.type() == DiscreteBNType()

    loaded_gen = pickle.loads(dyn_genericbn_bytes)
    assert set(loaded_gen.variables()) == set(["a", "b", "c", "d"])
    assert loaded_gen.static_bn().arcs() == [("a_t_2", "d_t_1")]
    assert loaded_gen.transition_bn().arcs() == [("a_t_2", "b_t_0")]
    assert loaded_gen.type() == MyRestrictedGaussianNetworkType()

    loaded_nn = pickle.loads(dyn_newbn_bytes)
    assert set(loaded_nn.variables()) == set(["a", "b", "c", "d"])
    assert loaded_nn.static_bn().arcs() == [("a_t_2", "d_t_1")]
    assert loaded_nn.transition_bn().arcs() == [("a_t_2", "b_t_0")]
    assert loaded_nn.type() == MyRestrictedGaussianNetworkType()

    loaded_other = pickle.loads(dyn_otherbn_bytes)
    assert set(loaded_other.variables()) == set(["a", "b", "c", "d"])
    assert loaded_other.static_bn().arcs() == [("a_t_2", "d_t_1")]
    assert loaded_other.transition_bn().arcs() == [("a_t_2", "b_t_0")]
    assert loaded_other.type() == NonHomogeneousType()
    assert loaded_other.extra_info == "extra"

    assert loaded_other.static_bn().node_type("c_t_1") == DiscreteFactorType()
    assert loaded_other.static_bn().node_type("d_t_1") == CKDEType()
    assert loaded_other.transition_bn().node_type("d_t_0") == CKDEType()

@pytest.fixture
def dyn_gaussian_partial_fit_bytes():
    gaussian = DynamicGaussianNetwork(["a", "b", "c", "d"], 2)
    gaussian.static_bn().add_arc("a_t_2", "d_t_1")
    gaussian.transition_bn().add_arc("c_t_2", "b_t_0")
    lg = LinearGaussianCPD("d_t_1", ["a_t_2"], [1, 2], 2)
    gaussian.static_bn().add_cpds([lg])
    lg = LinearGaussianCPD("b_t_0", ["c_t_2"], [3, 4], 5)
    gaussian.transition_bn().add_cpds([lg])
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)

@pytest.fixture
def dyn_gaussian_fit_bytes():
    gaussian = DynamicGaussianNetwork(["a", "b", "c", "d"], 2)
    gaussian.static_bn().add_arc("a_t_2", "d_t_1")
    gaussian.transition_bn().add_arc("c_t_2", "b_t_0")
    df = util_test.generate_normal_data_indep(1000)
    gaussian.fit(df)
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)

@pytest.fixture
def dyn_other_partial_fit_bytes():
    variables = ["a", "b", "c", "d"]
    static_nodes = [v + "_t_" + str(m) for v in variables for m in range(1, 3)]
    transition_nodes = [v + "_t_0" for v in variables]

    other_static = OtherBN(static_nodes, [("a_t_2", "d_t_1")], [("b_t_1", DiscreteFactorType()),
                                                                ("c_t_1", CKDEType()),
                                                                ("d_t_1", LinearGaussianCPDType())])
    lg = LinearGaussianCPD("d_t_1", ["a_t_2"], [1, 2], 2)
    other_static.add_cpds([lg])

    other_transition = ConditionalOtherBN(transition_nodes,
                                          static_nodes,
                                          [("a_t_2", "d_t_0")],
                                          [("b_t_0", DiscreteFactorType()),
                                            ("c_t_0", CKDEType()),
                                            ("d_t_0", LinearGaussianCPDType())])
    lg = LinearGaussianCPD("d_t_0", ["a_t_2"], [3, 4], 1.5)
    other_transition.add_cpds([lg])

    assert other_static.type() == other_transition.type()

    dyn_other = DynamicOtherBN(variables, 2, other_static, other_transition)
    dyn_other.include_cpd = True
    return pickle.dumps(dyn_other)

@pytest.fixture
def dyn_other_fit_bytes():
    variables = ["a", "b", "c", "d"]
    static_nodes = [v + "_t_" + str(m) for v in variables for m in range(1, 3)]
    transition_nodes = [v + "_t_0" for v in variables]

    other_static = OtherBN(static_nodes, [("a_t_2", "d_t_1")], [("b_t_2", DiscreteFactorType()),
                                                                ("b_t_1", DiscreteFactorType()),
                                                                ("c_t_1", CKDEType()),
                                                                ("d_t_1", LinearGaussianCPDType())])
    lg = LinearGaussianCPD("d_t_1", ["a_t_2"], [1, 2], 2)
    other_static.add_cpds([lg])

    other_transition = ConditionalOtherBN(transition_nodes,
                                          static_nodes,
                                          [("a_t_2", "d_t_0")],
                                          [("b_t_0", DiscreteFactorType()),
                                            ("c_t_0", CKDEType()),
                                            ("d_t_0", LinearGaussianCPDType())])
    lg = LinearGaussianCPD("d_t_0", ["a_t_2"], [3, 4], 1.5)
    other_transition.add_cpds([lg])

    assert other_static.type() == other_transition.type()

    dyn_other = DynamicOtherBN(variables, 2, other_static, other_transition)
    df_continuous = util_test.generate_normal_data_indep(1000)
    df_discrete = util_test.generate_discrete_data_dependent(1000)
    df = df_continuous
    df["b"] = df_discrete["B"]
    dyn_other.fit(df)
    dyn_other.include_cpd = True
    return pickle.dumps(dyn_other)

def test_serialization_fitted_dbn(dyn_gaussian_partial_fit_bytes, dyn_gaussian_fit_bytes,
                                    dyn_other_partial_fit_bytes, dyn_other_fit_bytes):
    # ####################
    # Gaussian partial fit
    # ####################
    loaded_partial = pickle.loads(dyn_gaussian_partial_fit_bytes)
    assert not loaded_partial.fitted
    assert not loaded_partial.static_bn().fitted
    assert not loaded_partial.transition_bn().fitted
    cpd = loaded_partial.static_bn().cpd("d_t_1")
    assert cpd.variable == "d_t_1"
    assert cpd.evidence == ["a_t_2"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    cpd = loaded_partial.transition_bn().cpd("b_t_0")
    assert cpd.variable == "b_t_0"
    assert cpd.evidence == ["c_t_2"]
    assert list(cpd.beta) == [3, 4]
    assert cpd.variance == 5

    # ####################
    # Gaussian fit
    # ####################
    loaded_fitted = pickle.loads(dyn_gaussian_fit_bytes)
    assert loaded_fitted.fitted
    assert loaded_fitted.static_bn().fitted
    assert loaded_fitted.transition_bn().fitted

    # ####################
    # Other partial fit
    # ####################
    loaded_partial = pickle.loads(dyn_other_partial_fit_bytes)
    assert not loaded_partial.fitted
    assert not loaded_partial.static_bn().fitted
    assert not loaded_partial.transition_bn().fitted
    assert loaded_partial.static_bn().node_type("b_t_1") == DiscreteFactorType()
    assert loaded_partial.static_bn().node_type("c_t_1") == CKDEType()
    assert loaded_partial.static_bn().node_type("d_t_1") == LinearGaussianCPDType()

    assert loaded_partial.transition_bn().node_type("b_t_0") == DiscreteFactorType()
    assert loaded_partial.transition_bn().node_type("c_t_0") == CKDEType()
    assert loaded_partial.transition_bn().node_type("d_t_0") == LinearGaussianCPDType()

    cpd = loaded_partial.static_bn().cpd("d_t_1")
    assert cpd.variable == "d_t_1"
    assert cpd.evidence == ["a_t_2"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    cpd = loaded_partial.transition_bn().cpd("d_t_0")
    assert cpd.variable == "d_t_0"
    assert cpd.evidence == ["a_t_2"]
    assert list(cpd.beta) == [3, 4]
    assert cpd.variance == 1.5

    # ####################
    # Other fit
    # ####################
    loaded_fitted = pickle.loads(dyn_other_fit_bytes)
    assert loaded_fitted.fitted
    assert loaded_fitted.static_bn().fitted
    assert loaded_fitted.transition_bn().fitted
    assert loaded_partial.static_bn().node_type("b_t_1") == DiscreteFactorType()
    assert loaded_partial.static_bn().node_type("c_t_1") == CKDEType()
    assert loaded_partial.static_bn().node_type("d_t_1") == LinearGaussianCPDType()

    assert loaded_partial.transition_bn().node_type("b_t_0") == DiscreteFactorType()
    assert loaded_partial.transition_bn().node_type("c_t_0") == CKDEType()
    assert loaded_partial.transition_bn().node_type("d_t_0") == LinearGaussianCPDType()

    cpd = loaded_partial.static_bn().cpd("d_t_1")
    assert cpd.variable == "d_t_1"
    assert cpd.evidence == ["a_t_2"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    cpd = loaded_partial.transition_bn().cpd("d_t_0")
    assert cpd.variable == "d_t_0"
    assert cpd.evidence == ["a_t_2"]
    assert list(cpd.beta) == [3, 4]
    assert cpd.variance == 1.5
