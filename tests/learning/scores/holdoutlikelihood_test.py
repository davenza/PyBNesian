import pytest
import numpy as np
from scipy.stats import gaussian_kde, norm
import pybnesian as pbn
import util_test

SIZE = 1000
df = util_test.generate_normal_data(SIZE)
seed = 0

def numpy_local_score(node_type, training_data, test_data, variable, evidence):
    if isinstance(variable, str):
        node_data = training_data.loc[:, [variable] + evidence].dropna()
        variable_data = node_data.loc[:, variable]
        evidence_data = node_data.loc[:, evidence]
        test_node_data = test_data.loc[:, [variable] + evidence].dropna()
        test_variable_data = test_node_data.loc[:, variable]
        test_evidence_data = test_node_data.loc[:, evidence]
    else:
        node_data = training_data.iloc[:, [variable] + evidence].dropna()
        variable_data = node_data.iloc[:, 0]
        evidence_data = node_data.iloc[:, 1:]
        test_node_data = test_data.iloc[:, [variable] + evidence].dropna()
        test_variable_data = test_node_data.iloc[:, 0]
        test_evidence_data = test_node_data.iloc[:, 1:]

    if node_type == pbn.LinearGaussianCPDType():
        N = variable_data.shape[0]
        d = evidence_data.shape[1]
        linregress_data = np.column_stack((np.ones(N), evidence_data.to_numpy()))
        (beta, res, _, _) = np.linalg.lstsq(linregress_data, variable_data.to_numpy(), rcond=None)
        var = res / (N - d - 1)

        means = beta[0] + np.sum(beta[1:]*test_evidence_data, axis=1)
        return norm.logpdf(test_variable_data, means, np.sqrt(var)).sum()
    elif node_type == pbn.CKDEType():
        k_joint = gaussian_kde(node_data.to_numpy().T,
                    bw_method=lambda s : np.power(4 / (s.d + 2), 1 / (s.d + 4)) * s.scotts_factor())
        if evidence:
            k_marg = gaussian_kde(evidence_data.to_numpy().T, bw_method=k_joint.covariance_factor())
            return np.sum(k_joint.logpdf(test_node_data.to_numpy().T) - k_marg.logpdf(test_evidence_data.to_numpy().T))
        else:
            return np.sum(k_joint.logpdf(test_node_data.to_numpy().T))

def test_holdout_create():
    s = pbn.HoldoutLikelihood(df)
    assert s.training_data().num_rows == 0.8 * SIZE
    assert s.test_data().num_rows == 0.2 * SIZE

    s = pbn.HoldoutLikelihood(df, 0.5)
    assert s.training_data().num_rows == 0.5 * SIZE
    assert s.test_data().num_rows == 0.5 * SIZE
    
    s = pbn.HoldoutLikelihood(df, 0.2, 0)
    s2 = pbn.HoldoutLikelihood(df, 0.2, 0)

    assert s.training_data().equals(s2.training_data())
    assert s.test_data().equals(s2.test_data())

    with pytest.raises(ValueError) as ex:
        s = pbn.HoldoutLikelihood(df, 10, 0)
    assert "test_ratio must be a number"  in str(ex.value)

    with pytest.raises(ValueError) as ex:
        s = pbn.HoldoutLikelihood(df, 0, 0)
    assert "test_ratio must be a number"  in str(ex.value)


def test_holdout_local_score_gbn():
    gbn = pbn.GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
    hl = pbn.HoldoutLikelihood(df, 0.2, seed)

    assert np.isclose(hl.local_score(gbn, 'a', []), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'a', []))
    assert np.isclose(hl.local_score(gbn, 'b', ['a']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'b', ['a']))
    assert np.isclose(hl.local_score(gbn, 'c', ['a', 'b']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'c', ['a', 'b']))
    assert np.isclose(hl.local_score(gbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'd', ['a', 'b', 'c']))
    assert np.isclose(hl.local_score(gbn, 'd', ['a', 'b', 'c']), 
                      hl.local_score(gbn, 'd', ['b', 'c', 'a']))

    assert hl.local_score(gbn, 'a') == hl.local_score(gbn, 'a', gbn.parents('a'))
    assert hl.local_score(gbn, 'b') == hl.local_score(gbn, 'b', gbn.parents('b'))
    assert hl.local_score(gbn, 'c') == hl.local_score(gbn, 'c', gbn.parents('c'))
    assert hl.local_score(gbn, 'd') == hl.local_score(gbn, 'd', gbn.parents('d'))

def test_holdout_local_score_gbn_null():
    gbn = pbn.GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
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

    hl = pbn.HoldoutLikelihood(df_null, 0.2, seed)

    assert np.isclose(hl.local_score(gbn, 'a', []), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'a', []))
    assert np.isclose(hl.local_score(gbn, 'b', ['a']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'b', ['a']))
    assert np.isclose(hl.local_score(gbn, 'c', ['a', 'b']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'c', ['a', 'b']))
    assert np.isclose(hl.local_score(gbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'd', ['a', 'b', 'c']))
    assert np.isclose(hl.local_score(gbn, 'd', ['a', 'b', 'c']), 
                      hl.local_score(gbn, 'd', ['b', 'c', 'a']))

    assert hl.local_score(gbn, 'a') == hl.local_score(gbn, 'a', gbn.parents('a'))
    assert hl.local_score(gbn, 'b') == hl.local_score(gbn, 'b', gbn.parents('b'))
    assert hl.local_score(gbn, 'c') == hl.local_score(gbn, 'c', gbn.parents('c'))
    assert hl.local_score(gbn, 'd') == hl.local_score(gbn, 'd', gbn.parents('d'))

def test_holdout_local_score_spbn():
    spbn = pbn.SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')],
                                [('a', pbn.CKDEType()), ('c', pbn.CKDEType())])
    
    hl = pbn.HoldoutLikelihood(df, 0.2, seed)

    assert np.isclose(hl.local_score(spbn, 'a', []), 
                      numpy_local_score(pbn.CKDEType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'a', []))
    assert np.isclose(hl.local_score(spbn, 'b', ['a']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'b', ['a']))
    assert np.isclose(hl.local_score(spbn, 'c', ['a', 'b']), 
                      numpy_local_score(pbn.CKDEType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'c', ['a', 'b']))
    assert np.isclose(hl.local_score(spbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'd', ['a', 'b', 'c']))
    assert np.isclose(hl.local_score(spbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'd', ['b', 'c', 'a']))

    assert hl.local_score(spbn, 'a') == hl.local_score(spbn, 'a', spbn.parents('a'))
    assert hl.local_score(spbn, 'b') == hl.local_score(spbn, 'b', spbn.parents('b'))
    assert hl.local_score(spbn, 'c') == hl.local_score(spbn, 'c', spbn.parents('c'))
    assert hl.local_score(spbn, 'd') == hl.local_score(spbn, 'd', spbn.parents('d'))

def test_holdout_local_score_null_spbn():
    spbn = pbn.SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')],
                                [('a', pbn.CKDEType()), ('c', pbn.CKDEType())])
    
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

    hl = pbn.HoldoutLikelihood(df_null, 0.2, seed)

    assert np.isclose(hl.local_score(spbn, 'a', []), 
                      numpy_local_score(pbn.CKDEType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'a', []))
    assert np.isclose(hl.local_score(spbn, 'b', ['a']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'b', ['a']))
    assert np.isclose(hl.local_score(spbn, 'c', ['a', 'b']), 
                      numpy_local_score(pbn.CKDEType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'c', ['a', 'b']))
    assert np.isclose(hl.local_score(spbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'd', ['a', 'b', 'c']))
    assert np.isclose(hl.local_score(spbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(pbn.LinearGaussianCPDType(), hl.training_data().to_pandas(), hl.test_data().to_pandas(), 'd', ['b', 'c', 'a']))

    assert hl.local_score(spbn, 'a') == hl.local_score(spbn, 'a', spbn.parents('a'))
    assert hl.local_score(spbn, 'b') == hl.local_score(spbn, 'b', spbn.parents('b'))
    assert hl.local_score(spbn, 'c') == hl.local_score(spbn, 'c', spbn.parents('c'))
    assert hl.local_score(spbn, 'd') == hl.local_score(spbn, 'd', spbn.parents('d'))

def test_holdout_score():
    gbn = pbn.GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    hl = pbn.HoldoutLikelihood(df, 0.2, 0)

    assert np.isclose(hl.score(gbn), (
                            hl.local_score(gbn, 'a', []) +
                            hl.local_score(gbn, 'b', ['a']) +
                            hl.local_score(gbn, 'c', ['a', 'b']) +
                            hl.local_score(gbn, 'd', ['a', 'b', 'c'])))

    spbn = pbn.SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')], 
                                [('a', pbn.CKDEType()), ('c', pbn.CKDEType())])

    assert np.isclose(hl.score(spbn), (
                            hl.local_score(spbn, 'a') +
                            hl.local_score(spbn, 'b') +
                            hl.local_score(spbn, 'c') +
                            hl.local_score(spbn, 'd')))