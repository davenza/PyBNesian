import numpy as np
from scipy.stats import norm
from pybnesian.models import GaussianNetwork
from pybnesian.learning.scores import BIC
from pybnesian.factors.continuous import LinearGaussianCPD

import util_test

SIZE = 10000

df = util_test.generate_normal_data(SIZE)

def numpy_local_score(data, variable, evidence):
    if isinstance(variable, str):
        node_data = data.loc[:, [variable] + evidence].dropna()
        variable_data = node_data.loc[:, variable]
        evidence_data = node_data.loc[:, evidence]
    else:
        node_data = data.iloc[:, [variable] + evidence].dropna()
        variable_data = node_data.iloc[:, 0]
        evidence_data = node_data.iloc[:, 1:]

    N = variable_data.shape[0]
    d = evidence_data.shape[1]
    linregress_data = np.column_stack((np.ones(N), evidence_data.to_numpy()))
    (beta, res, _, _) = np.linalg.lstsq(linregress_data, variable_data.to_numpy(), rcond=None)
    var = res / (N - d - 1)

    means = beta[0] + np.sum(beta[1:]*evidence_data, axis=1)
    loglik = norm.logpdf(variable_data, means, np.sqrt(var))

    return loglik.sum() - np.log(N) * 0.5 * (d + 2)

def test_local_score():
    gbn = GaussianNetwork(['a', 'b', 'c', 'd'], [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
    bic = BIC(df)
    
    assert np.isclose(bic.local_score(gbn, 'a', []), numpy_local_score(df, 'a', []))
    assert np.isclose(bic.local_score(gbn, 'b', ['a']), numpy_local_score(df, 'b', ['a']))
    assert np.isclose(bic.local_score(gbn, 'c', ['a', 'b']), numpy_local_score(df, 'c', ['a', 'b']))
    assert np.isclose(bic.local_score(gbn, 'd', ['a', 'b', 'c']), numpy_local_score(df, 'd', ['a', 'b', 'c']))
    assert np.isclose(bic.local_score(gbn, 'd', ['a', 'b', 'c']), numpy_local_score(df, 'd', ['b', 'c', 'a']))

    assert np.isclose(bic.local_score(gbn, 0, []), numpy_local_score(df, 0, []))
    assert np.isclose(bic.local_score(gbn, 1, [0]), numpy_local_score(df, 1, [0]))
    assert np.isclose(bic.local_score(gbn, 2, [0, 1]), numpy_local_score(df, 2, [0, 1]))
    assert np.isclose(bic.local_score(gbn, 3, [0, 1, 2]), numpy_local_score(df, 3, [0, 1, 2]))
    assert np.isclose(bic.local_score(gbn, 3, [0, 1, 2]), numpy_local_score(df, 3, [1, 2, 0]))

    assert bic.local_score(gbn, 'a') == bic.local_score(gbn, 'a', gbn.parents('a'))
    assert bic.local_score(gbn, 'b') == bic.local_score(gbn, 'b', gbn.parents('b'))
    assert bic.local_score(gbn, 'c') == bic.local_score(gbn, 'c', gbn.parents('c'))
    assert bic.local_score(gbn, 'd') == bic.local_score(gbn, 'd', gbn.parents('d'))

    assert bic.local_score(gbn, 0) == bic.local_score(gbn, 0, gbn.parent_indices(0))
    assert bic.local_score(gbn, 1) == bic.local_score(gbn, 1, gbn.parent_indices(1))
    assert bic.local_score(gbn, 2) == bic.local_score(gbn, 2, gbn.parent_indices(2))
    assert bic.local_score(gbn, 3) == bic.local_score(gbn, 3, gbn.parent_indices(3))

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
    
    assert np.isclose(bic.local_score(gbn, 'a', []), numpy_local_score(df_null, 'a', []))
    assert np.isclose(bic.local_score(gbn, 'b', ['a']), numpy_local_score(df_null, 'b', ['a']))
    assert np.isclose(bic.local_score(gbn, 'c', ['a', 'b']), numpy_local_score(df_null, 'c', ['a', 'b']))
    assert np.isclose(bic.local_score(gbn, 'd', ['a', 'b', 'c']), numpy_local_score(df_null, 'd', ['a', 'b', 'c']))
    assert np.isclose(bic.local_score(gbn, 'd', ['a', 'b', 'c']), numpy_local_score(df_null, 'd', ['b', 'c', 'a']))

    assert np.isclose(bic.local_score(gbn, 0, []), numpy_local_score(df_null, 0, []))
    assert np.isclose(bic.local_score(gbn, 1, [0]), numpy_local_score(df_null, 1, [0]))
    assert np.isclose(bic.local_score(gbn, 2, [0, 1]), numpy_local_score(df_null, 2, [0, 1]))
    assert np.isclose(bic.local_score(gbn, 3, [0, 1, 2]), numpy_local_score(df_null, 3, [0, 1, 2]))
    assert np.isclose(bic.local_score(gbn, 3, [0, 1, 2]), numpy_local_score(df_null, 3, [1, 2, 0]))

    assert bic.local_score(gbn, 'a') == bic.local_score(gbn, 'a', gbn.parents('a'))
    assert bic.local_score(gbn, 'b') == bic.local_score(gbn, 'b', gbn.parents('b'))
    assert bic.local_score(gbn, 'c') == bic.local_score(gbn, 'c', gbn.parents('c'))
    assert bic.local_score(gbn, 'd') == bic.local_score(gbn, 'd', gbn.parents('d'))

    assert bic.local_score(gbn, 0) == bic.local_score(gbn, 0, gbn.parent_indices(0))
    assert bic.local_score(gbn, 1) == bic.local_score(gbn, 1, gbn.parent_indices(1))
    assert bic.local_score(gbn, 2) == bic.local_score(gbn, 2, gbn.parent_indices(2))
    assert bic.local_score(gbn, 3) == bic.local_score(gbn, 3, gbn.parent_indices(3))


def test_score():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
    bic = BIC(df)
    
    assert np.isclose(bic.score(gbn), (bic.local_score(gbn, 'a', []) + 
                              bic.local_score(gbn, 'b', ['a']) + 
                              bic.local_score(gbn, 'c', ['a', 'b']) +
                              bic.local_score(gbn, 'd', ['a', 'b', 'c'])))


