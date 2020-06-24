import pyarrow as pa
from pgm_dataset.factors.continuous import LinearGaussianCPD
import numpy as np
import pandas as pd

from pgmpy.estimators.MLE import MaximumLikelihoodEstimator
from pgmpy.factors.continuous import LinearGaussianCPD as LG_pgmpy

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

train_df = df.iloc[:7200, :]
test_df = df.iloc[7200:8000, :]

a_cpd = LinearGaussianCPD("a", [])
a_cpd.fit(df)
print(a_cpd.slogpdf(test_df))


a_pgmpy = MaximumLikelihoodEstimator.gaussian_estimate_with_parents("a", [], train_df)
print("pgmpy: beta: " + str(a_pgmpy.beta) + ", variance: " + str(a_pgmpy.variance))
print(a_pgmpy.logpdf_dataset(test_df).sum())