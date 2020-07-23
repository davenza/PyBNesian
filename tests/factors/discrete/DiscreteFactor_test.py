import pyarrow as pa
from pgm_dataset.factors.discrete import DiscreteFactor
import util_test

df = util_test.generate_discrete_data_dependent(10000)

def test_fit():
    a = DiscreteFactor('D', ['C'])
    a.fit(df)
    a.print_prob()
    pass