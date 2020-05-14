import numpy as np
import pyarrow as pa
import pgm_dataset

from pgm_dataset import LinearGaussianCPD

import pandas as pd

SIZE = 20000
NA_SIZE = 0
# df = pd.DataFrame({'a': [0.1, np.nan, np.nan], 'b': [-23, np.nan, 4]})

# df = pd.DataFrame({'a': np.random.normal(size=10), 'b': np.random.normal(size=10)})
# df = pd.DataFrame({
#                     'a': pd.Series(np.random.randint(0, 20, size=SIZE), dtype='float'),
#                     'b': pd.Series(np.random.randint(0, 5, size=SIZE), dtype='Int32')
#                     })


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



a_nan_indices = np.random.randint(0,SIZE, size=NA_SIZE)
b_nan_indices = np.random.randint(0,SIZE, size=NA_SIZE)
c_nan_indices = np.random.randint(0,SIZE, size=NA_SIZE)
d_nan_indices = np.random.randint(0,SIZE, size=NA_SIZE)

df.loc[a_nan_indices,'a'] = np.nan
df.loc[b_nan_indices,'b'] = np.nan
df.loc[c_nan_indices,'c'] = np.nan
df.loc[d_nan_indices,'d'] = np.nan


pa_df = pa.RecordBatch.from_pandas(df)

df_non_nan = df[["a"]].dropna()
print("Python beta " + str(df_non_nan["a"].mean()))
print("Python var " + str(df_non_nan["a"].var()))
cpd = LinearGaussianCPD("a", [])
cpd.fit(pa_df)

print()

df_non_nan = df[["a", "b"]].dropna()
linregress_data = np.column_stack((np.ones(df_non_nan.shape[0]), df_non_nan[["a"]]))
beta, res,_,_ = np.linalg.lstsq(linregress_data, df_non_nan.loc[:,'b'].values, rcond=None)
print("Python solution: " + str(beta))
print("Python var: " + str(res / (df_non_nan.shape[0]-2)))
cpd = LinearGaussianCPD("b", ["a"])
cpd.fit(pa_df)

print()

df_non_nan = df[["a", "b", "c"]].dropna()
linregress_data = np.column_stack((np.ones(df_non_nan.shape[0]), df_non_nan[["a", "b"]]))
beta, res,_,_ = np.linalg.lstsq(linregress_data, df_non_nan.loc[:,'c'].values, rcond=None)
print("Python solution: " + str(beta))
print("Python var: " + str(res / (df_non_nan.shape[0]-3)))
cpd = LinearGaussianCPD("c", ["a", "b"])
cpd.fit(pa_df)


print()

df_non_nan = df[["a", "b", "c", "d"]].dropna()
linregress_data = np.column_stack((np.ones(df_non_nan.shape[0]), df_non_nan[["a", "b", "c"]]))
beta, res,_,_ = np.linalg.lstsq(linregress_data, df_non_nan.loc[:,'d'].values, rcond=None)
print("Python solution: " + str(beta))
print("Python var: " + str(res / (df_non_nan.shape[0]-4)))
cpd = LinearGaussianCPD("d", ["a", "b", "c"])
cpd.fit(pa_df)