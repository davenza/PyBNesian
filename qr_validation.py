import numpy as np
import pandas as pd




spambase = pd.read_csv('spambase.csv')
spambase = spambase.drop('class', axis=1)
spambase = spambase.astype(np.float64)


X = np.column_stack((np.ones(spambase.shape[0]), spambase[["capital_run_length_average", "capital_run_length_longest"]]))
y = spambase["capital_run_length_total"].to_numpy()

qf, rf = np.linalg.qr(X)

y1 = y[:1000]
y2 = y[1000:2000]
y3 = y[2000:]

y12 = y[:2000]
y13 = np.concatenate((y[:1000], y[2000:]))
y23 = y[1000:]

qp1, rp1 = np.linalg.qr(X[:1000])
qp2, rp2 = np.linalg.qr(X[1000:2000])
qp3, rp3 = np.linalg.qr(X[2000:])

qp12, rp12 = np.linalg.qr(X[:2000])
qp13, rp13 = np.linalg.qr(np.vstack((X[:1000], X[2000:])))
qp23, rp23 = np.linalg.qr(X[1000:])


w = np.linalg.inv(rf).dot(qf.T).dot(y)
w1 = np.linalg.inv(rp1).dot(qp1.T).dot(y1)
w2 = np.linalg.inv(rp2).dot(qp2.T).dot(y2)
w3 = np.linalg.inv(rp3).dot(qp3.T).dot(y3)
w12 = np.linalg.inv(rp12).dot(qp12.T).dot(y12)
w13 = np.linalg.inv(rp13).dot(qp13.T).dot(y13)
w23 = np.linalg.inv(rp23).dot(qp23.T).dot(y23)

z = np.vstack((rp1, rp2, rp3))

q, r = np.linalg.qr(z)

q1 = q[:X.shape[1]]
q2 = q[X.shape[1]:2*X.shape[1]]
q3 = q[2*X.shape[1]:]
q12 = q[:2*X.shape[1]]
q13 = np.vstack((q[:X.shape[1]], q[2*X.shape[1]:]))
q23 = q[X.shape[1]:]

a1 = np.linalg.inv(r).dot(q1.T).dot(q1).dot(r)
a2 = np.linalg.inv(r).dot(q2.T).dot(q2).dot(r)
a3 = np.linalg.inv(r).dot(q3.T).dot(q3).dot(r)

print("Truth w: " + str(w))
print("Estimated w: " + str(a1.dot(w1) + a2.dot(w2) + a3.dot(w3)))
print()
print("Truth w12: " + str(w12))
print("Estimated w12: " + str(a1.dot(w1) + a2.dot(w2)))
print()
print("Truth w13: " + str(w13))
print("Estimated w13: " + str(a1.dot(w1) + a3.dot(w3)))
print()
print("Truth w23: " + str(w23))
print("Estimated w23: " + str(a2.dot(w2) + a3.dot(w3)))
