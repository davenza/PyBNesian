import pyarrow as pa
from pyarrow import csv
import numpy as np
import pandas as pd
from pgm_dataset import estimate
# from pgm_dataset import benchmark_sort_vec, benchmark_partial_sort_vec, benchmark_sort_set, benchmark_sort_priority, benchmark_sort_heap

np.random.seed(0)

SIZE = 10000

a_array = np.random.normal(3, 0.5, size=SIZE)
b_array = 2.5 + 1.65*a_array + np.random.normal(0, 2, size=SIZE)
c_array = -4.2 - 1.2*a_array + 3.2*b_array + np.random.normal(0, 0.75, size=SIZE)
d_array = 1.5 - 0.9*a_array + 5.6*b_array + 0.3 * c_array + np.random.normal(0, 0.5, size=SIZE)


df = pd.DataFrame({
                    'a': a_array,
                    'b': b_array,
                    'c': c_array,
                    'd': d_array
                    })



pa_df = pa.RecordBatch.from_pandas(df)

spambase = pd.read_csv('spambase.csv')
spambase = spambase.astype(np.float64)
spambase = spambase.drop("class", axis=1)

# estimate(pa_df, "bic", [], [("a", "b"), ("b", "c"), ("c", "d")], 5, 10e-4)
estimate(spambase, "bic", [], [], 0, 10e-4)

# nodes = 500
# iterations = 100
# sampling = 20

# # benchmark_sort_heap(nodes, iterations, sampling)
# benchmark_sort_priority(nodes, iterations, sampling)
# benchmark_sort_set(nodes, iterations, sampling)
# benchmark_sort_vec(nodes, iterations, sampling)
# benchmark_partial_sort_vec(nodes, iterations, sampling)

