import numpy as np
import pandas as pd
import pyarrow as pa
import pybnesian as pbn
import pytest
from helpers.data import DATA_SIZE, generate_discrete_data

df = generate_discrete_data(DATA_SIZE)


def test_data_type():
    a = pbn.DiscreteFactor("a", [])
    with pytest.raises(ValueError) as ex:
        a.data_type()
    assert "DiscreteFactor factor not fitted." in str(ex.value)

    categories = np.asarray(["A1", "A2"])
    a_values = pd.Categorical(
        categories[np.random.randint(len(categories), size=100)],
        categories=categories,
        ordered=False,
    )
    df = pd.DataFrame({"a": a_values})
    a.fit(df)
    assert a.data_type() == pa.dictionary(pa.int8(), pa.string())

    categories = np.asarray(["a" + str(i) for i in range(1, 129)])
    a_values = pd.Categorical(
        categories[np.random.randint(len(categories), size=100)],
        categories=categories,
        ordered=False,
    )
    df = pd.DataFrame({"a": a_values})
    a.fit(df)
    assert a.data_type() == pa.dictionary(pa.int8(), pa.string())

    categories = np.asarray(["a" + str(i) for i in range(1, 130)])
    a_values = pd.Categorical(
        categories[np.random.randint(len(categories), size=100)],
        categories=categories,
        ordered=False,
    )
    df = pd.DataFrame({"a": a_values})
    a.fit(df)
    assert a.data_type() == pa.dictionary(pa.int16(), pa.string())


def test_fit():
    # a = DiscreteFactor('C', ['A', 'B'])
    a = pbn.DiscreteFactor("c", [])
    a.fit(df)
