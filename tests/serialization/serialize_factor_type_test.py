import pytest
import pybnesian as pbn
from pybnesian import FactorType
import pickle

@pytest.fixture
def lg_type_bytes():
    lg = pbn.LinearGaussianCPDType()
    return pickle.dumps(lg)

@pytest.fixture
def ckde_type_bytes():
    ckde = pbn.CKDEType()
    return pickle.dumps(ckde)

@pytest.fixture
def discrete_type_bytes():
    discrete = pbn.DiscreteFactorType()
    return pickle.dumps(discrete)

class NewType(FactorType):
    def __init__(self):
        FactorType.__init__(self)

class OtherType(FactorType):
    def __init__(self):
        FactorType.__init__(self)

@pytest.fixture
def new_type_bytes():
    n = NewType()
    return pickle.dumps(n)

@pytest.fixture
def other_type_bytes():
    o = OtherType()
    return pickle.dumps(o)

def test_serialization_factor_type(lg_type_bytes, ckde_type_bytes, discrete_type_bytes, new_type_bytes, other_type_bytes):
    loaded_lg = pickle.loads(lg_type_bytes)
    new_lg = pbn.LinearGaussianCPDType()
    assert new_lg == loaded_lg

    loaded_ckde = pickle.loads(ckde_type_bytes)
    new_ckde = pbn.CKDEType()
    assert loaded_ckde == new_ckde

    loaded_discrete = pickle.loads(discrete_type_bytes)
    new_discrete = pbn.DiscreteFactorType()
    assert loaded_discrete == new_discrete

    loaded_new = pickle.loads(new_type_bytes)
    new_new = NewType()
    assert loaded_new == new_new

    loaded_other = pickle.loads(other_type_bytes)
    new_other = OtherType()
    assert loaded_other == new_other

    assert new_lg != new_ckde
    assert new_lg != new_discrete
    assert new_lg != new_new
    assert new_lg != new_other
    assert new_ckde != new_discrete
    assert new_ckde != new_new
    assert new_ckde != new_other
    assert new_discrete != new_new
    assert new_discrete != new_other
    assert new_new != new_other