import pytest
import numpy as np
import pandas as pd
import pyarrow as pa
from pybnesian.factors.discrete import DiscreteCPD
import util_test

df = util_test.generate_discrete_data_dependent(10000)

def test_data_type():
    a = DiscreteCPD("A", [])
    with pytest.raises(ValueError) as ex:
        a.data_type()
    "DiscreteCPD factor not fitted." in str(ex.value)
    
    categories = np.asarray(["a1", "a2"])
    a_values = pd.Categorical(categories[np.random.randint(len(categories), size=100)], categories=categories, ordered=False)
    df = pd.DataFrame({'A': a_values})
    a.fit(df)
    assert a.data_type() == pa.dictionary(pa.int8(), pa.string())

    categories = np.asarray(["a" + str(i) for i in range(1, 129)])
    a_values = pd.Categorical(categories[np.random.randint(len(categories), size=100)], categories=categories, ordered=False)
    df = pd.DataFrame({'A': a_values})
    a.fit(df)
    assert a.data_type() == pa.dictionary(pa.int8(), pa.string())

    categories = np.asarray(["a" + str(i) for i in range(1, 130)])
    a_values = pd.Categorical(categories[np.random.randint(len(categories), size=100)], categories=categories, ordered=False)
    df = pd.DataFrame({'A': a_values})
    a.fit(df)
    assert a.data_type() == pa.dictionary(pa.int16(), pa.string())

def test_fit():
    a = DiscreteCPD('C', [])
    a.fit(df)
