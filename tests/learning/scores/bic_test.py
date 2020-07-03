import numpy as np
from scipy.stats import norm
import pyarrow as pa
from pgm_dataset.models import GaussianNetwork
from pgm_dataset.learning.scores import BIC
from pgm_dataset.factors.continuous import LinearGaussianCPD

import util_test

SIZE = 10000

df = util_test.generate_normal_data(SIZE)


def numpy_local_score(data, variable, evidence):
    node_data = data[[variable] + evidence].dropna()
    N = node_data.shape[0]
    linregress_data = np.column_stack((np.ones(N), node_data[evidence]))
    (beta, res, _, _) = np.linalg.lstsq(linregress_data, node_data[variable], rcond=None)
    var = res / (N - len(evidence) - 1)

    means = beta[0] + np.sum(beta[1:]*node_data.loc[:,evidence], axis=1)
    loglik = norm.logpdf(node_data.loc[:,variable], means, np.sqrt(var))

    return loglik.sum() - np.log(N) * 0.5 * (len(evidence) + 2)

def local_score_tester(data, bic, gbn, variable, evidence):
    assert np.isclose(bic.local_score(gbn, variable, evidence), numpy_local_score(data, variable, evidence))

def test_local_score():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
    bic = BIC(df)
    
    local_score_tester(df, bic, gbn, 'a', [])
    local_score_tester(df, bic, gbn, 'b', ['a'])
    local_score_tester(df, bic, gbn, 'c', ['a', 'b'])
    local_score_tester(df, bic, gbn, 'd', ['a', 'b', 'c'])

def test_local_score_null():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

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

def test_score():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
    bic = BIC(df)
    
    assert bic.score(gbn) == (bic.local_score(gbn, 'a', []) + 
                              bic.local_score(gbn, 'b', ['a']) + 
                              bic.local_score(gbn, 'c', ['a', 'b']) +
                              bic.local_score(gbn, 'd', ['a', 'b', 'c']))


