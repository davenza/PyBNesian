import pyarrow as pa
from pgm_dataset.factors.continuous import KDE
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, multivariate_normal
from scipy.special import logsumexp

np.random.seed(1)

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



k = gaussian_kde(df.to_numpy().T)

covariance = k.covariance
cholesky = np.linalg.cholesky(covariance)

spk = gaussian_kde(df.iloc[:8000, :].loc[:, "a"].to_numpy().T)
print(spk.logpdf(df.iloc[8000:, :].loc[:, "a"].to_numpy().T))

k = KDE(["a"])
k.fit(df.iloc[:8000])
print(k.logpdf(df.iloc[8000:]))