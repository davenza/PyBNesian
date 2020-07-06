import pyarrow as pa
from pyarrow import csv
import numpy as np
import pandas as pd
from pgm_dataset.learning.algorithms import hc
from pgm_dataset.dataset import CrossValidation, HoldOut
# from pgm_dataset import benchmark_sort_vec, benchmark_partial_sort_vec, benchmark_sort_set, benchmark_sort_priority, benchmark_sort_heap
import time
from pgmpy.estimators import GaussianValidationLikelihood, CachedHillClimbing, ValidationSPBNStrict, HybridCachedHillClimbingStrict
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

# cv = CrossValidation(df, seed=1)

# for (train_df, test_df), (train_indices, test_indices) in zip(cv, cv.indices()):
#     nptrain = np.asarray(train_indices)
#     nptest = np.asarray(test_indices)
#     print(np.intersect1d(nptrain, nptest))
#     combination = np.hstack((nptrain, nptest))
#     print(np.all(np.sort(combination) == np.arange(SIZE)))

#     print(np.all(train_df.to_pandas().to_numpy() == df.iloc[train_indices,:].to_numpy()))
#     print(np.all(test_df.to_pandas().to_numpy() == df.iloc[test_indices,:].to_numpy()))



# holdout = HoldOut(df, seed = 0)
# print(holdout.training_data().to_pandas())
# print(holdout.test_data().to_pandas())


# for(training_df, test_df) in cv.indices():
#     print(np.asarray(training_df))
#     print(np.asarray(test_df))
    # print(training_df)
    # print(hex(training_df.column(0).buffers()[1].address))

# tic = time.time()
# hc(pa_df, "gbn", "predic-l", ["arcs"], [("d", "b")], [], [], 0, 0, 0, 0, "matrix")
# toc = time.time()
# print("Time: " + str((toc-tic)*1000) + "ms")


# holdout = HoldOut(df, test_ratio=0.2, seed=0)
# cv = CrossValidation(holdout.training_data(), k=10, seed=0)

# indices = list(cv.indices())

# gv = GaussianValidationLikelihood(df, k=10, seed=0)
# gv.validation_data = holdout.test_data().to_pandas()
# gv.data = holdout.training_data().to_pandas()
# gv.fold_indices = indices


# cb_draw = DrawModel('pgmpy_models/')
# cb_save = DrawModel('pgmpy_models/')
# ghc = CachedHillClimbing(df, scoring_method=gv)
# gbn = ghc.estimate(callbacks=[cb_draw, cb_save], patience=0)




spambase = pd.read_csv('spambase.csv')
spambase = spambase.astype(np.float64)
spambase = spambase.drop("class", axis=1)

tic = time.time()
hc(spambase, "gbn", "bic", ["arcs"], [], [], [], 0, 0, 0, 5, "matrix")
toc = time.time()
print("Time: " + str((toc-tic)*1000) + "ms")


# holdout = HoldOut(spambase, test_ratio=0.2, seed=0)
# cv = CrossValidation(holdout.training_data(), k=10, seed=0)

# indices = list(cv.indices())

# val = ValidationSPBNStrict(df, k=10, seed=0)
# val.validation_data = holdout.test_data().to_pandas()
# val.data = holdout.training_data().to_pandas()
# val.fold_indices = indices


# cb_draw = DrawModel('pgmpy_models/')
# cb_save = DrawModel('pgmpy_models/')
# ghc = HybridCachedHillClimbingStrict(spambase, scoring_method=val)
# tic = time.time()
# gbn = ghc.estimate(callbacks=[cb_draw, cb_save], patience=0)
# toc = time.time()
# print("Time: " + str((toc-tic)*1000) + "ms")