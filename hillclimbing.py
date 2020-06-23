import pyarrow as pa
from pyarrow import csv
import numpy as np
import pandas as pd
from pgm_dataset.learning.algorithms import hc
# from pgm_dataset import benchmark_sort_vec, benchmark_partial_sort_vec, benchmark_sort_set, benchmark_sort_priority, benchmark_sort_heap
import time
from pgmpy.estimators import GaussianValidationLikelihood, CachedHillClimbing
from pgmpy.estimators.callbacks import DrawModel, SaveModel

import os


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

# tic = time.time()
# hc(pa_df, "gbn", "predic-l", ["arcs"], [], [], [], 0, 0, 0, 0, "matrix")
# toc = time.time()
# print("Time: " + str((toc-tic)*1000) + "ms")

# gv = GaussianValidationLikelihood(df, k=10, seed=0)
# cb_draw = DrawModel('pgmpy_models/')
# cb_save = DrawModel('pgmpy_models/')
# ghc = CachedHillClimbing(df, scoring_method=gv)
# gbn = ghc.estimate(callbacks=[cb_draw, cb_save], patience=0)

spambase = pd.read_csv('spambase.csv')
spambase = spambase.astype(np.float64)
spambase = spambase.drop("class", axis=1)

tic = time.time()
hc(spambase, "gbn", "predic-l", ["arcs"], [], [], [], 0, 0, 0, 5, "matrix")
toc = time.time()
print("Time: " + str((toc-tic)*1000) + "ms")


# gv = GaussianValidationLikelihood(spambase, k=10, seed=0)
# cb_draw = DrawModel('pgmpy_models/')
# cb_save = DrawModel('pgmpy_models/')
# ghc = CachedHillClimbing(spambase, scoring_method=gv)
# gbn = ghc.estimate(callbacks=[cb_draw, cb_save], patience=0)

# nodes = 500
# iterations = 100
# sampling = 20

# # benchmark_sort_heap(nodes, iterations, sampling)
# benchmark_sort_priority(nodes, iterations, sampling)
# benchmark_sort_set(nodes, iterations, sampling)
# benchmark_sort_vec(nodes, iterations, sampling)
# benchmark_partial_sort_vec(nodes, iterations, sampling)

