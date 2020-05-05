import numpy as np
import pyarrow as pa
import pgm_dataset

from pgm_dataset import LinearGaussianCPD

import pandas as pd

SIZE = 100000
NA_SIZE = 0
# df = pd.DataFrame({'a': [0.1, np.nan, np.nan], 'b': [-23, np.nan, 4]})

# df = pd.DataFrame({'a': np.random.normal(size=10), 'b': np.random.normal(size=10)})
# df = pd.DataFrame({
#                     'a': pd.Series(np.random.randint(0, 20, size=SIZE), dtype='float'),
#                     'b': pd.Series(np.random.randint(0, 5, size=SIZE), dtype='Int32')
#                     })


a_array = np.random.normal(3, 0.5, size=SIZE)
b_array = 2.5 + 1.65*a_array + np.random.normal(0, 2, size=SIZE)


df = pd.DataFrame({
                    'a': a_array,
                    'b': b_array
                    })



a_nan_indices = np.random.randint(0,SIZE, size=NA_SIZE)
b_nan_indices = np.random.randint(0,SIZE, size=NA_SIZE)
# print(np.unique(a_nan_indices).size)
# print(np.sort(a_nan_indices))
# print(b_nan_indices)

df.loc[a_nan_indices,'a'] = np.nan
df.loc[b_nan_indices,'b'] = np.nan

print("Python mean: " + str(df.loc[:,'a'].mean()))
print("Python var: " + str(df.loc[:,'a'].var()))

# df.loc[:,'b'] = df.loc[:,'b'].astype('float')
print(df.dtypes)
print(df)

# for i in range(0, SIZE, 64):
#     print(df.loc[:,'a'].iloc[i:(i+64)].values)


pa_df = pa.RecordBatch.from_pandas(df)

# cpd = LinearGaussianCPD("a", [])
# cpd.fit(df)

cpd = LinearGaussianCPD("b", ["a"])
cpd.fit(df)