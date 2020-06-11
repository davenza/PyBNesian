import pyarrow as pa
from pgm_dataset.factors.continuous import KDE
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from scipy.special import logsumexp

np.random.seed(1)

SIZE = 100000

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


train = df.iloc[:10].loc[:, 'a']
test = df.iloc[10:].loc[:, 'a']
print("train data: " + str(train))
print("test data: " + str(test))
bandwidth = train.shape[0]**(-2/(5))*np.var(train, ddof=1)


k = gaussian_kde(train)

# print("bandwidth: "  + str(bandwidth))
# print("bandwidth scipy: " + str(k.covariance))
# print("cholesky scipy: " + str(np.sqrt(k.covariance)))

# l = norm(test[0], np.sqrt(k.covariance)).logpdf(train)
# print("invididual logpdf: " + str(l - np.log(train.shape[0])))

cpd = KDE(['a'])
cpd.fit(df.iloc[:10])

print("Ground truth: " + str(k.logpdf(test[:10])))
print("My implementation " + str(cpd.logpdf(df.iloc[10:])[:10]))