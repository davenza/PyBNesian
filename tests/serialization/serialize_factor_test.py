import numpy as np
import pandas as pd
import pytest
import pybnesian as pbn
from pybnesian import FactorType, Factor, LinearGaussianCPD, CKDE, DiscreteFactor
import pickle

@pytest.fixture
def lg_bytes():
    lg = LinearGaussianCPD("c", ["a", "b"])
    return pickle.dumps(lg)

@pytest.fixture
def ckde_bytes():
    ckde = CKDE("c", ["a", "b"])
    return pickle.dumps(ckde)

@pytest.fixture
def discrete_bytes():
    discrete = DiscreteFactor("c", ["a", "b"])
    return pickle.dumps(discrete)

class NewType(FactorType):
    def __init__(self, factor_class):
        FactorType.__init__(self)
        self.factor_class = factor_class

    def new_factor(self, model, variable, evidence):
        return self.factor_class(variable, evidence)

    def __str__(self):
        return "NewType"

class NewFactor(Factor):
    def __init__(self, variable, evidence):
        Factor.__init__(self, variable, evidence)
        self._fitted = False
        self.some_fit_data = None
    
    def fit(self, df):
        self.some_fit_data = "fitted"
        self._fitted = True
    
    def fitted(self):
        return self._fitted

    def type(self):
        return NewType(NewFactor)

    def __str__(self):
        return "NewFactor"

    def __getstate_extra__(self):
        d = {'fitted': self._fitted,  'some_fit_data': self.some_fit_data}
        return d

    def __setstate_extra__(self, d):
        self._fitted = d['fitted']
        self.some_fit_data = d['some_fit_data']

class NewFactorBis(Factor):
    def __init__(self, variable, evidence):
        Factor.__init__(self, variable, evidence)
        self._fitted = False
        self.some_fit_data = None

    def fit(self, df):
        self.some_fit_data = "fitted"
        self._fitted = True
    
    def fitted(self):
        return self._fitted

    def type(self):
        return NewType(NewFactorBis)

    def __str__(self):
        return "NewFactor"

    def __getstate__(self):
        d = {'variable': self.variable(),
             'evidence': self.evidence(),
             'fitted': self._fitted,
             'some_fit_data': self.some_fit_data}
        return d

    def __setstate__(self, d):
        Factor.__init__(self, d['variable'], d['evidence'])
        self._fitted = d['fitted']
        self.some_fit_data = d['some_fit_data']

@pytest.fixture
def new_bytes():
    n = NewFactor("c", ["a", "b"])
    return pickle.dumps(n)

@pytest.fixture
def newbis_bytes():
    n = NewFactorBis("c", ["a", "b"])
    return pickle.dumps(n)

def test_serialization_unfitted_factor(lg_bytes, ckde_bytes, discrete_bytes, new_bytes, newbis_bytes):
    loaded_lg = pickle.loads(lg_bytes)
    assert loaded_lg.variable() == "c"
    assert set(loaded_lg.evidence()) == set(["a", "b"])
    assert not loaded_lg.fitted()
    assert loaded_lg.type() == pbn.LinearGaussianCPDType()

    loaded_ckde = pickle.loads(ckde_bytes)
    assert loaded_ckde.variable() == "c"
    assert set(loaded_ckde.evidence()) == set(["a", "b"])
    assert not loaded_ckde.fitted()
    assert loaded_ckde.type() == pbn.CKDEType()

    loaded_discrete = pickle.loads(discrete_bytes)
    assert loaded_discrete.variable() == "c"
    assert set(loaded_discrete.evidence()) == set(["a", "b"])
    assert not loaded_discrete.fitted()
    assert loaded_discrete.type() == pbn.DiscreteFactorType()

    loaded_new = pickle.loads(new_bytes)
    assert loaded_new.variable() == "c"
    assert set(loaded_new.evidence()) == set(["a", "b"])
    assert not loaded_new.fitted()
    assert type(loaded_new.type()) == NewType
    nn = NewFactor("a", [])
    assert loaded_new.type() == nn.type()

    from pybnesian import GaussianNetwork
    dummy_network = GaussianNetwork(["a", "b", "c", "d"])
    assert type(loaded_new.type().new_factor(dummy_network, "a", [])) == NewFactor

    loaded_newbis = pickle.loads(newbis_bytes)
    assert loaded_newbis.variable() == "c"
    assert set(loaded_newbis.evidence()) == set(["a", "b"])
    assert not loaded_newbis.fitted()
    assert type(loaded_newbis.type()) == NewType
    nnbis = NewFactorBis("a", [])
    assert loaded_newbis.type() == nnbis.type()
    assert type(loaded_newbis.type().new_factor(dummy_network, "a", [])) == NewFactorBis

    assert loaded_lg.type() != loaded_ckde.type()
    assert loaded_lg.type() != loaded_discrete.type()
    assert loaded_lg.type() != loaded_new.type()
    assert loaded_ckde.type() != loaded_discrete.type()
    assert loaded_ckde.type() != loaded_new.type()
    assert loaded_discrete.type() != loaded_new.type()
    assert loaded_newbis.type() == loaded_new.type()

@pytest.fixture
def lg_fitted_bytes():
    lg = LinearGaussianCPD("c", ["a", "b"], [1, 2, 3], 0.5)
    return pickle.dumps(lg)

@pytest.fixture
def ckde_fitted_bytes():
    np.random.seed(1)
    data = pd.DataFrame({'a': np.random.rand(10), 'b': np.random.rand(10), 'c': np.random.rand(10)}).astype(float)
    ckde = CKDE("c", ["a", "b"])
    ckde.fit(data)
    return pickle.dumps(ckde)

@pytest.fixture
def discrete_fitted_bytes():
    discrete = DiscreteFactor("c", ["a", "b"])

    data = pd.DataFrame({'a': ["a1", "a2", "a1", "a2", "a2", "a2", "a2", "a2"], 
                         'b': ["b1", "b1", "b1", "b1", "b1", "b2", "b1", "b2"],
                         'c': ["c1", "c1", "c1", "c1", "c2", "c2", "c2", "c2"]}, dtype="category")
    discrete.fit(data)
    return pickle.dumps(discrete)

@pytest.fixture
def new_fitted_bytes():
    n = NewFactor("c", ["a", "b"])
    n.fit(None)
    return pickle.dumps(n)

@pytest.fixture
def newbis_fitted_bytes():
    n = NewFactorBis("c", ["a", "b"])
    n.fit(None)
    return pickle.dumps(n)

def test_serialization_fitted_factor(lg_fitted_bytes, ckde_fitted_bytes, discrete_fitted_bytes, new_fitted_bytes,
                                     newbis_fitted_bytes):
    loaded_lg = pickle.loads(lg_fitted_bytes)
    assert loaded_lg.variable() == "c"
    assert set(loaded_lg.evidence()) == set(["a", "b"])
    assert loaded_lg.fitted()
    assert list(loaded_lg.beta) == [1, 2, 3]
    assert loaded_lg.variance == 0.5

    loaded_ckde = pickle.loads(ckde_fitted_bytes)
    assert loaded_ckde.variable() == "c"
    assert set(loaded_ckde.evidence()) == set(["a", "b"])
    assert loaded_ckde.fitted()
    assert loaded_ckde.type() == pbn.CKDEType()
    assert loaded_ckde.num_instances() == 10
    tr = loaded_ckde.kde_joint().dataset().to_pandas()
    np.random.seed(1)
    assert np.all(tr['a'] == np.random.rand(10))
    assert np.all(tr['b'] == np.random.rand(10))
    assert np.all(tr['c'] == np.random.rand(10))

    loaded_discrete = pickle.loads(discrete_fitted_bytes)
    assert loaded_discrete.variable() == "c"
    assert set(loaded_discrete.evidence()) == set(["a", "b"])
    assert loaded_discrete.fitted()
    assert loaded_discrete.type() == pbn.DiscreteFactorType()

    test = pd.DataFrame({'a': ["a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2"], 
                         'b': ["b1", "b1", "b2", "b2", "b1", "b1", "b2", "b2"],
                         'c': ["c1", "c1", "c1", "c1", "c2", "c2", "c2", "c2"]}, dtype="category")
    ll = loaded_discrete.logl(test)
    assert list(np.exp(ll)) == [1, 0.5, 0.5, 0, 0, 0.5, 0.5, 1]

    loaded_new = pickle.loads(new_fitted_bytes)
    assert loaded_new.variable() == "c"
    assert set(loaded_new.evidence()) == set(["a", "b"])
    assert loaded_new.fitted()
    assert type(loaded_new.type()) == NewType
    nn = NewFactor("a", [])
    assert loaded_new.type() == nn.type()
    assert loaded_new.some_fit_data == "fitted"

    loaded_newbis = pickle.loads(newbis_fitted_bytes)
    assert loaded_newbis.variable() == "c"
    assert set(loaded_newbis.evidence()) == set(["a", "b"])
    assert loaded_newbis.fitted()
    assert type(loaded_newbis.type()) == NewType
    nn = NewFactorBis("a", [])
    assert loaded_newbis.type() == nn.type()
    assert loaded_newbis.some_fit_data == "fitted"
    assert type(loaded_newbis.type()) == type(loaded_new.type())

    assert loaded_lg.type() != loaded_ckde.type()
    assert loaded_lg.type() != loaded_discrete.type()
    assert loaded_lg.type() != loaded_new.type()
    assert loaded_ckde.type() != loaded_discrete.type()
    assert loaded_ckde.type() != loaded_new.type()
    assert loaded_discrete.type() != loaded_new.type()