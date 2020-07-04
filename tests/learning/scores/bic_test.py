import numpy as np
from scipy.stats import norm
import pyarrow as pa
from pgm_dataset.models import GaussianNetwork
from pgm_dataset.learning.scores import BIC
from pgm_dataset.factors.continuous import LinearGaussianCPD

import util_test

SIZE = 10000

df = util_test.generate_normal_data(SIZE)


def numpy_local_score(variable_data, evidence_data):
    N = variable_data.shape[0]
    d = evidence_data.shape[1]
    linregress_data = np.column_stack((np.ones(N), evidence_data.to_numpy()))
    (beta, res, _, _) = np.linalg.lstsq(linregress_data, variable_data.to_numpy(), rcond=None)
    var = res / (N - d - 1)

    means = beta[0] + np.sum(beta[1:]*evidence_data, axis=1)
    loglik = norm.logpdf(variable_data, means, np.sqrt(var))

    return loglik.sum() - np.log(N) * 0.5 * (d + 2)

def local_score_tester(data, bic, gbn, variable, evidence):
    if isinstance(variable, str):
        node_data = data.loc[:, [variable] + evidence].dropna()
        variable_data = node_data.loc[:, variable]
        evidence_data = node_data.loc[:, evidence]
    else:
        node_data = data.iloc[:, [variable] + evidence].dropna()
        variable_data = node_data.iloc[:, 0]
        evidence_data = node_data.iloc[:, 1:]

    assert np.isclose(bic.local_score(gbn, variable, evidence), numpy_local_score(variable_data, evidence_data))

def test_local_score():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'], [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
    bic = BIC(df)
    
    local_score_tester(df, bic, gbn, 'a', [])
    local_score_tester(df, bic, gbn, 'b', ['a'])
    local_score_tester(df, bic, gbn, 'c', ['a', 'b'])
    local_score_tester(df, bic, gbn, 'd', ['a', 'b', 'c'])

    local_score_tester(df, bic, gbn, 0, [])
    local_score_tester(df, bic, gbn, 1, [0])
    local_score_tester(df, bic, gbn, 2, [0, 1])
    local_score_tester(df, bic, gbn, 3, [0, 1, 2])

def test_local_score_null():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'], [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    np.random.seed(0)
    a_null = np.random.randint(0, SIZE, size=100)
    b_null = np.random.randint(0, SIZE, size=100)
    c_null = np.random.randint(0, SIZE, size=100)
    d_null = np.random.randint(0, SIZE, size=100)

    df_null = df.copy()
    df_null.loc[df_null.index[a_null], 'a'] = np.nan
    df_null.loc[df_null.index[b_null], 'b'] = np.nan
    df_null.loc[df_null.index[c_null], 'c'] = np.nan
    df_null.loc[df_null.index[d_null], 'd'] = np.nan
    
    bic = BIC(df_null)
    
    local_score_tester(df_null, bic, gbn, 'a', [])
    local_score_tester(df_null, bic, gbn, 'b', ['a'])
    local_score_tester(df_null, bic, gbn, 'c', ['a', 'b'])
    local_score_tester(df_null, bic, gbn, 'd', ['a', 'b', 'c'])

    local_score_tester(df_null, bic, gbn, 0, [])
    local_score_tester(df_null, bic, gbn, 1, [0])
    local_score_tester(df_null, bic, gbn, 2, [0, 1])
    local_score_tester(df_null, bic, gbn, 3, [0, 1, 2])


def test_score():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
    bic = BIC(df)
    
    assert bic.score(gbn) == (bic.local_score(gbn, 'a', []) + 
                              bic.local_score(gbn, 'b', ['a']) + 
                              bic.local_score(gbn, 'c', ['a', 'b']) +
                              bic.local_score(gbn, 'd', ['a', 'b', 'c']))


