import pytest
import numpy as np
from scipy.stats import norm, gaussian_kde
from pybnesian.dataset import CrossValidation
from pybnesian.factors.continuous import LinearGaussianCPDType, CKDEType
from pybnesian.models import GaussianNetwork, SemiparametricBN
from pybnesian.learning.scores import CVLikelihood
import util_test

SIZE = 1000
df = util_test.generate_normal_data(SIZE)

seed = 0

def numpy_local_score(node_type, data, variable, evidence):
    cv = CrossValidation(data, 10, seed)
    loglik = 0
    for train_df, test_df in cv:
        if isinstance(variable, str):
            node_data = train_df.to_pandas().loc[:, [variable] + evidence].dropna()
            variable_data = node_data.loc[:, variable]
            evidence_data = node_data.loc[:, evidence]
            test_node_data = test_df.to_pandas().loc[:, [variable] + evidence].dropna()
            test_variable_data = test_node_data.loc[:, variable]
            test_evidence_data = test_node_data.loc[:, evidence]
        else:
            node_data = train_df.to_pandas().iloc[:, [variable] + evidence].dropna()
            variable_data = node_data.iloc[:, 0]
            evidence_data = node_data.iloc[:, 1:]
            test_node_data = test_df.to_pandas().iloc[:, [variable] + evidence].dropna()
            test_variable_data = test_node_data.iloc[:, 0]
            test_evidence_data = test_node_data.iloc[:, 1:]

        if node_type == LinearGaussianCPDType():
            N = variable_data.shape[0]
            d = evidence_data.shape[1]
            linregress_data = np.column_stack((np.ones(N), evidence_data.to_numpy()))
            (beta, res, _, _) = np.linalg.lstsq(linregress_data, variable_data.to_numpy(), rcond=None)
            var = res / (N - d - 1)

            means = beta[0] + np.sum(beta[1:]*test_evidence_data, axis=1)
            loglik += norm.logpdf(test_variable_data, means, np.sqrt(var)).sum()
        elif node_type == CKDEType():
            k_joint = gaussian_kde(node_data.to_numpy().T)
            if evidence:
                k_marg = gaussian_kde(evidence_data.to_numpy().T, bw_method=k_joint.covariance_factor())
                loglik += np.sum(k_joint.logpdf(test_node_data.to_numpy().T) - k_marg.logpdf(test_evidence_data.to_numpy().T))
            else:
                loglik += np.sum(k_joint.logpdf(test_node_data.to_numpy().T))

    return loglik

def test_cvl_create():
    s = CVLikelihood(df)
    assert len(list(s.cv)) == 10
    s = CVLikelihood(df, 5)
    assert len(list(s.cv)) == 5
    
    s = CVLikelihood(df, 10, 0)
    assert len(list(s.cv)) == 10
    s2 = CVLikelihood(df, 10, 0)
    assert len(list(s2.cv)) == 10

    for (train_cv, test_cv), (train_cv2, test_cv2) in zip(s.cv, s2.cv):
        assert train_cv.equals(train_cv2), "Train CV DataFrames with the same seed are not equal."
        assert test_cv.equals(test_cv2), "Test CV DataFrames with the same seed are not equal."

    with pytest.raises(ValueError) as ex:
        s = CVLikelihood(df, SIZE+1)
    assert "Cannot split" in str(ex.value)

def test_cvl_local_score_gbn():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])
    
    cvl = CVLikelihood(df, 10, seed)
    
    assert np.isclose(cvl.local_score(gbn, 'a', []),
                      numpy_local_score(LinearGaussianCPDType(), df, 'a', []))
    assert np.isclose(cvl.local_score(gbn, 'b', ['a']),
                      numpy_local_score(LinearGaussianCPDType(), df, 'b', ['a']))
    assert np.isclose(cvl.local_score(gbn, 'c', ['a', 'b']),
                      numpy_local_score(LinearGaussianCPDType(), df, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(gbn, 'd', ['a', 'b', 'c']),
                      numpy_local_score(LinearGaussianCPDType(), df, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(gbn, 'd', ['a', 'b', 'c']),
                      cvl.local_score(gbn, 'd', ['b', 'c', 'a']))

    assert cvl.local_score(gbn, 'a') == cvl.local_score(gbn, 'a', gbn.parents('a'))
    assert cvl.local_score(gbn, 'b') == cvl.local_score(gbn, 'b', gbn.parents('b'))
    assert cvl.local_score(gbn, 'c') == cvl.local_score(gbn, 'c', gbn.parents('c'))
    assert cvl.local_score(gbn, 'd') == cvl.local_score(gbn, 'd', gbn.parents('d'))

def test_cvl_local_score_gbn_null():
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

    cvl = CVLikelihood(df_null, 10, seed)

    assert np.isclose(cvl.local_score(gbn, 'a', []),
                      numpy_local_score(LinearGaussianCPDType(), df_null, 'a', []))
    assert np.isclose(cvl.local_score(gbn, 'b', ['a']),
                      numpy_local_score(LinearGaussianCPDType(), df_null, 'b', ['a']))
    assert np.isclose(cvl.local_score(gbn, 'c', ['a', 'b']),
                      numpy_local_score(LinearGaussianCPDType(), df_null, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(gbn, 'd', ['a', 'b', 'c']),
                      numpy_local_score(LinearGaussianCPDType(), df_null, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(gbn, 'd', ['a', 'b', 'c']),
                      cvl.local_score(gbn, 'd', ['b', 'c', 'a']))

    assert cvl.local_score(gbn, 'a') == cvl.local_score(gbn, 'a', gbn.parents('a'))
    assert cvl.local_score(gbn, 'b') == cvl.local_score(gbn, 'b', gbn.parents('b'))
    assert cvl.local_score(gbn, 'c') == cvl.local_score(gbn, 'c', gbn.parents('c'))
    assert cvl.local_score(gbn, 'd') == cvl.local_score(gbn, 'd', gbn.parents('d'))

def test_cvl_local_score_spbn():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')],
                            [('a', CKDEType()), ('c', CKDEType())])
    
    cvl = CVLikelihood(df, 10, seed)

    assert np.isclose(cvl.local_score(spbn, 'a', []),
                      numpy_local_score(CKDEType(), df, 'a', []))
    assert np.isclose(cvl.local_score(spbn, 'b', ['a']),
                      numpy_local_score(LinearGaussianCPDType(), df, 'b', ['a']))
    assert np.isclose(cvl.local_score(spbn, 'c', ['a', 'b']),
                      numpy_local_score(CKDEType(), df, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(spbn, 'd', ['a', 'b', 'c']),
                      numpy_local_score(LinearGaussianCPDType(), df, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(spbn, 'd', ['a', 'b', 'c']),
                      numpy_local_score(LinearGaussianCPDType(), df, 'd', ['b', 'c', 'a']))

    assert cvl.local_score(spbn, 'a') == cvl.local_score(spbn, 'a', spbn.parents('a'))
    assert cvl.local_score(spbn, 'b') == cvl.local_score(spbn, 'b', spbn.parents('b'))
    assert cvl.local_score(spbn, 'c') == cvl.local_score(spbn, 'c', spbn.parents('c'))
    assert cvl.local_score(spbn, 'd') == cvl.local_score(spbn, 'd', spbn.parents('d'))

    assert np.isclose(cvl.local_score(spbn, LinearGaussianCPDType(), 'a', []), 
                      numpy_local_score(LinearGaussianCPDType(), df, 'a', []))
    assert np.isclose(cvl.local_score(spbn, CKDEType(), 'b', ['a']),
                      numpy_local_score(CKDEType(), df, 'b', ['a']))
    assert np.isclose(cvl.local_score(spbn, LinearGaussianCPDType(), 'c', ['a', 'b']),
                      numpy_local_score(LinearGaussianCPDType(), df, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(spbn, CKDEType(), 'd', ['a', 'b', 'c']),
                      numpy_local_score(CKDEType(), df, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(spbn, CKDEType(), 'd', ['a', 'b', 'c']),
                      numpy_local_score(CKDEType(), df, 'd', ['b', 'c', 'a']))


def test_cvl_local_score_null_spbn():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')],
                            [('a', CKDEType()), ('c', CKDEType())])
    
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

    cvl = CVLikelihood(df_null, 10, seed)

    assert np.isclose(cvl.local_score(spbn, 'a', []), 
                      numpy_local_score(CKDEType(), df_null, 'a', []))
    assert np.isclose(cvl.local_score(spbn, 'b', ['a']), 
                      numpy_local_score(LinearGaussianCPDType(), df_null, 'b', ['a']))
    assert np.isclose(cvl.local_score(spbn, 'c', ['a', 'b']), 
                      numpy_local_score(CKDEType(), df_null, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(spbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(LinearGaussianCPDType(), df_null, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(spbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(LinearGaussianCPDType(), df_null, 'd', ['b', 'c', 'a']))

    assert cvl.local_score(spbn, 'a') == cvl.local_score(spbn, 'a', spbn.parents('a'))
    assert cvl.local_score(spbn, 'b') == cvl.local_score(spbn, 'b', spbn.parents('b'))
    assert cvl.local_score(spbn, 'c') == cvl.local_score(spbn, 'c', spbn.parents('c'))
    assert cvl.local_score(spbn, 'd') == cvl.local_score(spbn, 'd', spbn.parents('d'))

    assert np.isclose(cvl.local_score(spbn, LinearGaussianCPDType(), 'a', []), 
                      numpy_local_score(LinearGaussianCPDType(), df_null, 'a', []))
    assert np.isclose(cvl.local_score(spbn, CKDEType(), 'b', ['a']),
                      numpy_local_score(CKDEType(), df_null, 'b', ['a']))
    assert np.isclose(cvl.local_score(spbn, LinearGaussianCPDType(), 'c', ['a', 'b']),
                      numpy_local_score(LinearGaussianCPDType(), df_null, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(spbn, CKDEType(), 'd', ['a', 'b', 'c']),
                      numpy_local_score(CKDEType(), df_null, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(spbn, CKDEType(), 'd', ['a', 'b', 'c']),
                      numpy_local_score(CKDEType(), df_null, 'd', ['b', 'c', 'a']))

def test_cvl_score():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    cv = CVLikelihood(df, 10, 0)

    assert np.isclose(cv.score(gbn), (
                            cv.local_score(gbn, 'a', []) +
                            cv.local_score(gbn, 'b', ['a']) +
                            cv.local_score(gbn, 'c', ['a', 'b']) +
                            cv.local_score(gbn, 'd', ['a', 'b', 'c'])))

    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')], 
                            [('a', CKDEType()), ('c', CKDEType())])

    assert np.isclose(cv.score(spbn), (
                            cv.local_score(spbn, 'a') +
                            cv.local_score(spbn, 'b') +
                            cv.local_score(spbn, 'c') +
                            cv.local_score(spbn, 'd')))