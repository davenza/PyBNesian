import pytest
from pybnesian.models import *
import pickle
import itertools

@pytest.fixture
def gaussian_type_bytes():
    g = GaussianNetworkType()
    return pickle.dumps(g)

@pytest.fixture
def spbn_type_bytes():
    s = SemiparametricBNType()
    return pickle.dumps(s)

@pytest.fixture
def kde_type_bytes():
    k = KDENetworkType()
    return pickle.dumps(k)

@pytest.fixture
def discrete_type_bytes():
    d = DiscreteBNType()
    return pickle.dumps(d)

class NewBNType(BayesianNetworkType):
    def __init__(self):
        BayesianNetworkType.__init__(self)

    def ToString(self):
        return "NewType"

@pytest.fixture
def new_type_bytes():
    nn = NewBNType()
    return pickle.dumps(nn)

class OtherBNType(BayesianNetworkType):
    def __init__(self):
        BayesianNetworkType.__init__(self)
        self.some_useful_info = "info"

    def ToString(self):
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
    new_g = GaussianNetworkType()
    assert loaded_g == new_g

    loaded_s = pickle.loads(spbn_type_bytes)
    new_s = SemiparametricBNType()
    assert loaded_s == new_s

    loaded_k = pickle.loads(kde_type_bytes)
    new_k = KDENetworkType()
    assert loaded_k == new_k

    loaded_d = pickle.loads(discrete_type_bytes)
    new_d = DiscreteBNType()
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