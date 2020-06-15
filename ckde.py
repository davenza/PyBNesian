import pyarrow as pa
from pgm_dataset.factors.continuous import KDE, CKDE
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, multivariate_normal
from scipy.special import logsumexp
import time

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


TRAIN_POINT = 90000
OFFSET_SHOW = 6481
SHOW_INSTANCES = 20


# start_time = time.time()
# lp = spk.logpdf(df.iloc[(TRAIN_POINT + OFFSET_SHOW):(TRAIN_POINT + OFFSET_SHOW + SHOW_INSTANCES), :].loc[:, ["a", "b"]].to_numpy().T)
# end_time = time.time()
# print("Python time: " + str(end_time - start_time))
# print(lp)

# k_joint = KDE(["a", "b"])
# k.fit(df.iloc[:TRAIN_POINT])
# start_time = time.time()
# lc = k.logpdf(df.iloc[TRAIN_POINT:])
# end_time = time.time()
# print("c++ time: " + str(end_time - start_time))
# # print(lc[OFFSET_SHOW:(OFFSET_SHOW + SHOW_INSTANCES)])
# print(lc.sum())

print(df)

spk_joint = gaussian_kde(df.iloc[:TRAIN_POINT, :].loc[:, ["a", "b"]].to_numpy().T)
spk_marg = gaussian_kde(df.iloc[:TRAIN_POINT, :].loc[:, ["b"]].to_numpy().T, bw_method=spk_joint.covariance_factor())

# print(spk_joint.covariance)
# print(spk_marg.covariance)

start_time = time.time()
lp = spk_joint.logpdf(df.iloc[(TRAIN_POINT + OFFSET_SHOW):(TRAIN_POINT + OFFSET_SHOW + SHOW_INSTANCES), :].loc[:, ["a", "b"]].to_numpy().T) -\
     spk_marg.logpdf(df.iloc[(TRAIN_POINT + OFFSET_SHOW):(TRAIN_POINT + OFFSET_SHOW + SHOW_INSTANCES), :].loc[:, ["b"]].to_numpy().T)
end_time = time.time()
print("Python time: " + str(end_time - start_time))
print(lp[:SHOW_INSTANCES])

# print()
# print("Joint component:")
# print(spk_joint.logpdf(df.iloc[(TRAIN_POINT + OFFSET_SHOW):(TRAIN_POINT + OFFSET_SHOW + SHOW_INSTANCES), :].loc[:, ["a", "b"]].to_numpy().T))
# print("Marg component:")
# print(spk_marg.logpdf(df.iloc[(TRAIN_POINT + OFFSET_SHOW):(TRAIN_POINT + OFFSET_SHOW + SHOW_INSTANCES), :].loc[:, ["b"]].to_numpy().T))

kde = KDE(["a"])
kde.fit(df.iloc[:TRAIN_POINT])
start_time = time.time()
lc = kde.logpdf(df.iloc[TRAIN_POINT:])
end_time = time.time()
print("KDE c++ time: " + str(end_time - start_time))
print(lc[(OFFSET_SHOW):(OFFSET_SHOW + SHOW_INSTANCES)])


ckde = CKDE("a", [])
ckde.fit(df.iloc[:TRAIN_POINT])
start_time = time.time()
lc = ckde.logpdf(df.iloc[TRAIN_POINT:])
end_time = time.time()
print("c++ time: " + str(end_time - start_time))
print(lc[(OFFSET_SHOW):(OFFSET_SHOW + SHOW_INSTANCES)])