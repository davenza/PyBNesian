import pyarrow as pa
from pgm_dataset.factors.discrete import DiscreteFactor
import util_test

df = util_test.generate_discrete_data(10000)

def test_fit():
    a = DiscreteFactor('A', [])
    a.fit(df)