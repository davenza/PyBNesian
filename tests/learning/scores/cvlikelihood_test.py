import pytest
import numpy as np
from scipy.stats import norm, gaussian_kde
from pybnesian.dataset import CrossValidation
from pybnesian.factors import FactorType
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

        if node_type == FactorType.LinearGaussianCPD:
            N = variable_data.shape[0]
            d = evidence_data.shape[1]
            linregress_data = np.column_stack((np.ones(N), evidence_data.to_numpy()))
            (beta, res, _, _) = np.linalg.lstsq(linregress_data, variable_data.to_numpy(), rcond=None)
            var = res / (N - d - 1)

            means = beta[0] + np.sum(beta[1:]*test_evidence_data, axis=1)
            loglik += norm.logpdf(test_variable_data, means, np.sqrt(var)).sum()
        elif node_type == FactorType.CKDE:
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
    
    from pybnesian.learning.scores import ScoreSPBN
    from pybnesian.learning.scores import Score
    cvl.local_score(gbn, 'a', [])
    
    assert np.isclose(cvl.local_score(gbn, 'a', []), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 'a', []))
    assert np.isclose(cvl.local_score(gbn, 'b', ['a']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 'b', ['a']))
    assert np.isclose(cvl.local_score(gbn, 'c', ['a', 'b']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(gbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(gbn, 'd', ['a', 'b', 'c']), 
                      cvl.local_score(gbn, 'd', ['b', 'c', 'a']))

    assert np.isclose(cvl.local_score(gbn, 0, []), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 0, []))
    assert np.isclose(cvl.local_score(gbn, 1, [0]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 1, [0]))
    assert np.isclose(cvl.local_score(gbn, 2, [0, 1]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 2, [0, 1]))
    assert np.isclose(cvl.local_score(gbn, 3, [0, 1, 2]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 3, [0, 1, 2]))
    assert np.isclose(cvl.local_score(gbn, 3, [0, 1, 2]), 
                      cvl.local_score(gbn, 3, [1, 2, 0]))

    assert cvl.local_score(gbn, 'a') == cvl.local_score(gbn, 'a', gbn.parents('a'))
    assert cvl.local_score(gbn, 'b') == cvl.local_score(gbn, 'b', gbn.parents('b'))
    assert cvl.local_score(gbn, 'c') == cvl.local_score(gbn, 'c', gbn.parents('c'))
    assert cvl.local_score(gbn, 'd') == cvl.local_score(gbn, 'd', gbn.parents('d'))

    assert cvl.local_score(gbn, 0) == cvl.local_score(gbn, 0, gbn.parent_indices(0))
    assert cvl.local_score(gbn, 1) == cvl.local_score(gbn, 1, gbn.parent_indices(1))
    assert cvl.local_score(gbn, 2) == cvl.local_score(gbn, 2, gbn.parent_indices(2))
    assert cvl.local_score(gbn, 3) == cvl.local_score(gbn, 3, gbn.parent_indices(3))

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
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 'a', []))
    assert np.isclose(cvl.local_score(gbn, 'b', ['a']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 'b', ['a']))
    assert np.isclose(cvl.local_score(gbn, 'c', ['a', 'b']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(gbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(gbn, 'd', ['a', 'b', 'c']), 
                      cvl.local_score(gbn, 'd', ['b', 'c', 'a']))

    assert np.isclose(cvl.local_score(gbn, 0, []), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 0, []))
    assert np.isclose(cvl.local_score(gbn, 1, [0]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 1, [0]))
    assert np.isclose(cvl.local_score(gbn, 2, [0, 1]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 2, [0, 1]))
    assert np.isclose(cvl.local_score(gbn, 3, [0, 1, 2]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 3, [0, 1, 2]))
    assert np.isclose(cvl.local_score(gbn, 3, [0, 1, 2]), 
                      cvl.local_score(gbn, 3, [1, 2, 0]))

    assert cvl.local_score(gbn, 'a') == cvl.local_score(gbn, 'a', gbn.parents('a'))
    assert cvl.local_score(gbn, 'b') == cvl.local_score(gbn, 'b', gbn.parents('b'))
    assert cvl.local_score(gbn, 'c') == cvl.local_score(gbn, 'c', gbn.parents('c'))
    assert cvl.local_score(gbn, 'd') == cvl.local_score(gbn, 'd', gbn.parents('d'))

    assert cvl.local_score(gbn, 0) == cvl.local_score(gbn, 0, gbn.parent_indices(0))
    assert cvl.local_score(gbn, 1) == cvl.local_score(gbn, 1, gbn.parent_indices(1))
    assert cvl.local_score(gbn, 2) == cvl.local_score(gbn, 2, gbn.parent_indices(2))
    assert cvl.local_score(gbn, 3) == cvl.local_score(gbn, 3, gbn.parent_indices(3))


def test_cvl_local_score_spbn():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')],
                            [('a', FactorType.CKDE), ('c', FactorType.CKDE)])
    
    cvl = CVLikelihood(df, 10, seed)

    assert np.isclose(cvl.local_score(spbn, 'a', []), 
                      numpy_local_score(FactorType.CKDE, df, 'a', []))
    assert np.isclose(cvl.local_score(spbn, 'b', ['a']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 'b', ['a']))
    assert np.isclose(cvl.local_score(spbn, 'c', ['a', 'b']), 
                      numpy_local_score(FactorType.CKDE, df, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(spbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(spbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 'd', ['b', 'c', 'a']))

    assert np.isclose(cvl.local_score(spbn, 0, []), 
                      numpy_local_score(FactorType.CKDE, df, 0, []))
    assert np.isclose(cvl.local_score(spbn, 1, [0]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 1, [0]))
    assert np.isclose(cvl.local_score(spbn, 2, [0, 1]), 
                      numpy_local_score(FactorType.CKDE, df, 2, [0, 1]))
    assert np.isclose(cvl.local_score(spbn, 3, [0, 1, 2]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 3, [0, 1, 2]))
    assert np.isclose(cvl.local_score(spbn, 3, [0, 1, 2]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 3, [1, 2, 0]))

    assert cvl.local_score(spbn, 'a') == cvl.local_score(spbn, 'a', spbn.parents('a'))
    assert cvl.local_score(spbn, 'b') == cvl.local_score(spbn, 'b', spbn.parents('b'))
    assert cvl.local_score(spbn, 'c') == cvl.local_score(spbn, 'c', spbn.parents('c'))
    assert cvl.local_score(spbn, 'd') == cvl.local_score(spbn, 'd', spbn.parents('d'))

    assert cvl.local_score(spbn, 0) == cvl.local_score(spbn, 0, spbn.parent_indices(0))
    assert cvl.local_score(spbn, 1) == cvl.local_score(spbn, 1, spbn.parent_indices(1))
    assert cvl.local_score(spbn, 2) == cvl.local_score(spbn, 2, spbn.parent_indices(2))
    assert cvl.local_score(spbn, 3) == cvl.local_score(spbn, 3, spbn.parent_indices(3))

    assert np.isclose(cvl.local_score(FactorType.LinearGaussianCPD, 'a', []), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 'a', []))
    assert np.isclose(cvl.local_score(FactorType.CKDE, 'b', ['a']),
                      numpy_local_score(FactorType.CKDE, df, 'b', ['a']))
    assert np.isclose(cvl.local_score(FactorType.LinearGaussianCPD, 'c', ['a', 'b']),
                      numpy_local_score(FactorType.LinearGaussianCPD, df, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(FactorType.CKDE, 'd', ['a', 'b', 'c']),
                      numpy_local_score(FactorType.CKDE, df, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(FactorType.CKDE, 'd', ['a', 'b', 'c']),
                      numpy_local_score(FactorType.CKDE, df, 'd', ['b', 'c', 'a']))


def test_cvl_local_score_null_spbn():
    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')],
                            [('a', FactorType.CKDE), ('c', FactorType.CKDE)])
    
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
                      numpy_local_score(FactorType.CKDE, df_null, 'a', []))
    assert np.isclose(cvl.local_score(spbn, 'b', ['a']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 'b', ['a']))
    assert np.isclose(cvl.local_score(spbn, 'c', ['a', 'b']), 
                      numpy_local_score(FactorType.CKDE, df_null, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(spbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(spbn, 'd', ['a', 'b', 'c']), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 'd', ['b', 'c', 'a']))

    assert np.isclose(cvl.local_score(spbn, 0, []), 
                      numpy_local_score(FactorType.CKDE, df_null, 0, []))
    assert np.isclose(cvl.local_score(spbn, 1, [0]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 1, [0]))
    assert np.isclose(cvl.local_score(spbn, 2, [0, 1]), 
                      numpy_local_score(FactorType.CKDE, df_null, 2, [0, 1]))
    assert np.isclose(cvl.local_score(spbn, 3, [0, 1, 2]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 3, [0, 1, 2]))
    assert np.isclose(cvl.local_score(spbn, 3, [0, 1, 2]), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 3, [1, 2, 0]))

    assert cvl.local_score(spbn, 'a') == cvl.local_score(spbn, 'a', spbn.parents('a'))
    assert cvl.local_score(spbn, 'b') == cvl.local_score(spbn, 'b', spbn.parents('b'))
    assert cvl.local_score(spbn, 'c') == cvl.local_score(spbn, 'c', spbn.parents('c'))
    assert cvl.local_score(spbn, 'd') == cvl.local_score(spbn, 'd', spbn.parents('d'))

    assert cvl.local_score(spbn, 0) == cvl.local_score(spbn, 0, spbn.parent_indices(0))
    assert cvl.local_score(spbn, 1) == cvl.local_score(spbn, 1, spbn.parent_indices(1))
    assert cvl.local_score(spbn, 2) == cvl.local_score(spbn, 2, spbn.parent_indices(2))
    assert cvl.local_score(spbn, 3) == cvl.local_score(spbn, 3, spbn.parent_indices(3))

    assert np.isclose(cvl.local_score(FactorType.LinearGaussianCPD, 'a', []), 
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 'a', []))
    assert np.isclose(cvl.local_score(FactorType.CKDE, 'b', ['a']),
                      numpy_local_score(FactorType.CKDE, df_null, 'b', ['a']))
    assert np.isclose(cvl.local_score(FactorType.LinearGaussianCPD, 'c', ['a', 'b']),
                      numpy_local_score(FactorType.LinearGaussianCPD, df_null, 'c', ['a', 'b']))
    assert np.isclose(cvl.local_score(FactorType.CKDE, 'd', ['a', 'b', 'c']),
                      numpy_local_score(FactorType.CKDE, df_null, 'd', ['a', 'b', 'c']))
    assert np.isclose(cvl.local_score(FactorType.CKDE, 'd', ['a', 'b', 'c']),
                      numpy_local_score(FactorType.CKDE, df_null, 'd', ['b', 'c', 'a']))

def test_cvl_score():
    gbn = GaussianNetwork([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

    cv = CVLikelihood(df, 10, 0)

    assert np.isclose(cv.score(gbn), (
                            cv.local_score(gbn, 'a', []) +
                            cv.local_score(gbn, 'b', ['a']) +
                            cv.local_score(gbn, 'c', ['a', 'b']) +
                            cv.local_score(gbn, 'd', ['a', 'b', 'c'])))

    spbn = SemiparametricBN([('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')], 
                            [('a', FactorType.CKDE), ('c', FactorType.CKDE)])

    assert np.isclose(cv.score(spbn), (
                            cv.local_score(spbn, 'a') +
                            cv.local_score(spbn, 'b') +
                            cv.local_score(spbn, 'c') +
                            cv.local_score(spbn, 'd')))