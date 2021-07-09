import pytest
import pybnesian as pbn
import pickle
import itertools

@pytest.fixture
def gaussian_type_bytes():
    g = pbn.GaussianNetworkType()
    return pickle.dumps(g)

@pytest.fixture
def spbn_type_bytes():
    s = pbn.SemiparametricBNType()
    return pickle.dumps(s)

@pytest.fixture
def kde_type_bytes():
    k = pbn.KDENetworkType()
    return pickle.dumps(k)

@pytest.fixture
def discrete_type_bytes():
    d = pbn.DiscreteBNType()
    return pickle.dumps(d)

class NewBNType(pbn.BayesianNetworkType):
    def __init__(self):
        pbn.BayesianNetworkType.__init__(self)

    def __str__(self):
        return "NewType"

@pytest.fixture
def new_type_bytes():
    nn = NewBNType()
    return pickle.dumps(nn)

class OtherBNType(pbn.BayesianNetworkType):
    def __init__(self):
        pbn.BayesianNetworkType.__init__(self)
        self.some_useful_info = "info"

    def __str__(self):
        return "OtherType"

    def __getstate_extra__(self):
        return self.some_useful_info
    
    def __setstate_extra__(self, extra):
        self.some_useful_info = extra

@pytest.fixture
def other_type_bytes():
    o = OtherBNType()
    return pickle.dumps(o)


def test_serialization_bn_type(gaussian_type_bytes, spbn_type_bytes, kde_type_bytes,
                               discrete_type_bytes, new_type_bytes, other_type_bytes):
    loaded_g = pickle.loads(gaussian_type_bytes)
    new_g = pbn.GaussianNetworkType()
    assert loaded_g == new_g

    loaded_s = pickle.loads(spbn_type_bytes)
    new_s = pbn.SemiparametricBNType()
    assert loaded_s == new_s

    loaded_k = pickle.loads(kde_type_bytes)
    new_k = pbn.KDENetworkType()
    assert loaded_k == new_k

    loaded_d = pickle.loads(discrete_type_bytes)
    new_d = pbn.DiscreteBNType()
    assert loaded_d == new_d

    loaded_nn = pickle.loads(new_type_bytes)
    new_nn = NewBNType()
    assert loaded_nn == new_nn

    loaded_o = pickle.loads(other_type_bytes)
    new_o = OtherBNType()
    assert loaded_o == new_o
    assert loaded_o.some_useful_info == "info"

    m = [loaded_g, loaded_s, loaded_k, loaded_d, loaded_nn, loaded_o]

    for t in itertools.combinations(m, 2):
        assert t[0] != t[1]