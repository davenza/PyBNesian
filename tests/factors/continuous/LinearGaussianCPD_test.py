import numpy as np
import pandas as pd
import pyarrow as pa
import pybnesian as pbn

import pytest
from scipy.stats import norm

import util_test

SIZE = 10000

df = util_test.generate_normal_data(SIZE)

def test_lg_variable():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = pbn.LinearGaussianCPD(variable, evidence)
        assert cpd.variable() == variable

def test_lg_evidence():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = pbn.LinearGaussianCPD(variable, evidence)
        assert cpd.evidence() == evidence


def fit_numpy(_df, variable, evidence):
    df_na = _df.loc[:, [variable] + evidence].dropna()
    linregress_data = np.column_stack((np.ones(df_na.shape[0]), df_na.loc[:, evidence]))
    (beta, res, _, _) = np.linalg.lstsq(linregress_data, df_na.loc[:, variable], rcond=None)
    
    return beta, res / (df_na.count()[variable] - len(evidence) - 1)

def test_lg_data_type():
    cpd = pbn.LinearGaussianCPD("a", [])
    assert cpd.data_type() == pa.float64()

def test_lg_fit():
    for variable, evidence in [("a", []), ("b", ["a"]), ("c", ["a", "b"]), ("d", ["a", "b", "c"])]:
        cpd = pbn.LinearGaussianCPD(variable, evidence)
        assert not cpd.fitted()
        cpd.fit(df)
        assert cpd.fitted()

        npbeta, npvar = fit_numpy(df, variable, evidence)
        
        assert np.all(np.isclose(npbeta, cpd.beta)), "Wrong beta vector."
        assert np.all(np.isclose(npvar, cpd.variance)), "Wrong variance."

def test_lg_fit_null():
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

    for variable, evidence in [("a", []), ("b", ["a"]), ("c", ["a", "b"]), ("d", ["a", "b", "c"])]:
        cpd = pbn.LinearGaussianCPD(variable, evidence)
        assert not cpd.fitted()
        cpd.fit(df_null)
        assert cpd.fitted()

        npbeta, npvar = fit_numpy(df_null, variable, evidence)
        
        assert np.all(np.isclose(npbeta, cpd.beta)), "Wrong beta vector."
        assert np.all(np.isclose(npvar, cpd.variance)), "Wrong variance."

def numpy_logpdf(test_df, variable, evidence, beta, variance):
    npdata = test_df.loc[:, evidence].to_numpy()
    means = beta[0] + np.sum(beta[1:]*npdata, axis=1)

    result = np.empty((test_df.shape[0],))

    isnan_vec = np.full((test_df.shape[0],), False, dtype=bool)
    isnan_vec[np.isnan(means)] = True
    isnan_vec[np.isnan(test_df.loc[:, variable].to_numpy())] = True

    result[isnan_vec] = np.nan
    result[~isnan_vec] = norm.logpdf(test_df.loc[:, variable].to_numpy()[~isnan_vec], means[~isnan_vec], np.sqrt(variance))
    return result

def numpy_cdf(test_df, variable, evidence, beta, variance):
    npdata = test_df.loc[:, evidence].to_numpy()
    means = beta[0] + np.sum(beta[1:]*npdata, axis=1)

    result = np.empty((test_df.shape[0],))

    isnan_vec = np.full((test_df.shape[0],), False, dtype=bool)
    isnan_vec[np.isnan(means)] = True
    isnan_vec[np.isnan(test_df.loc[:, variable].to_numpy())] = True

    result[isnan_vec] = np.nan
    result[~isnan_vec] = norm.cdf(test_df.loc[:, variable].to_numpy()[~isnan_vec], means[~isnan_vec], np.sqrt(variance))
    return result

def test_lg_logl():
    test_df = util_test.generate_normal_data(5000)

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = pbn.LinearGaussianCPD(variable, evidence)
        cpd.fit(df)

        beta = cpd.beta
        variance = cpd.variance

        assert np.all(np.isclose(cpd.logl(test_df), numpy_logpdf(test_df, variable, evidence, beta, variance))),\
                     "Wrong logl for LinearGaussianCPD(" + str(variable) + " | " + str(evidence) + ")"

    
    cpd = pbn.LinearGaussianCPD('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.LinearGaussianCPD('d', ['c', 'a', 'b'])
    cpd2.fit(df)

    assert np.all(np.isclose(cpd.logl(test_df), cpd2.logl(test_df))), "The order of the evidence changes the logl() result."

def test_lg_logl_null():
    test_df = util_test.generate_normal_data(5000)

    np.random.seed(0)
    a_null = np.random.randint(0, 5000, size=100)
    b_null = np.random.randint(0, 5000, size=100)
    c_null = np.random.randint(0, 5000, size=100)
    d_null = np.random.randint(0, 5000, size=100)

    df_null = test_df.copy()
    df_null.loc[df_null.index[a_null], 'a'] = np.nan
    df_null.loc[df_null.index[b_null], 'b'] = np.nan
    df_null.loc[df_null.index[c_null], 'c'] = np.nan
    df_null.loc[df_null.index[d_null], 'd'] = np.nan

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = pbn.LinearGaussianCPD(variable, evidence)
        cpd.fit(df)

        beta = cpd.beta
        variance = cpd.variance

        assert np.all(np.isclose(
                        cpd.logl(test_df), 
                        numpy_logpdf(test_df, variable, evidence, beta, variance), equal_nan=True)),\
                     "Wrong logl for LinearGaussianCPD(" + str(variable) + " | " + str(evidence) + ") with null values."

    cpd = pbn.LinearGaussianCPD('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.LinearGaussianCPD('d', ['c', 'a', 'b'])
    cpd2.fit(df)

    assert np.all(np.isclose(
                        cpd.logl(test_df), 
                        cpd2.logl(test_df), equal_nan=True)),\
                     "The order of the evidence changes the logl() result."

def test_lg_slogl():
    test_df = util_test.generate_normal_data(5000)

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = pbn.LinearGaussianCPD(variable, evidence)
        cpd.fit(df)

        beta = cpd.beta
        variance = cpd.variance

        assert np.all(np.isclose(cpd.slogl(test_df), np.sum(numpy_logpdf(test_df, variable, evidence, beta, variance)))),\
                     "Wrong slogl for LinearGaussianCPD(" + str(variable) + " | " + str(evidence) + ")"

    cpd = pbn.LinearGaussianCPD('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.LinearGaussianCPD('d', ['c', 'a', 'b'])
    cpd2.fit(df)

    assert np.all(np.isclose(cpd.slogl(test_df), cpd2.slogl(test_df))), "The order of the evidence changes the slogl() result."

def test_lg_slogl_null():
    test_df = util_test.generate_normal_data(5000)

    np.random.seed(0)
    a_null = np.random.randint(0, 5000, size=100)
    b_null = np.random.randint(0, 5000, size=100)
    c_null = np.random.randint(0, 5000, size=100)
    d_null = np.random.randint(0, 5000, size=100)

    df_null = test_df.copy()
    df_null.loc[df_null.index[a_null], 'a'] = np.nan
    df_null.loc[df_null.index[b_null], 'b'] = np.nan
    df_null.loc[df_null.index[c_null], 'c'] = np.nan
    df_null.loc[df_null.index[d_null], 'd'] = np.nan

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = pbn.LinearGaussianCPD(variable, evidence)
        cpd.fit(df)

        beta = cpd.beta
        variance = cpd.variance

        assert np.all(np.isclose(cpd.slogl(df_null), np.nansum(numpy_logpdf(df_null, variable, evidence, beta, variance)))),\
                     "Wrong slogl for LinearGaussianCPD(" + str(variable) + " | " + str(evidence) + ") with null values."

    cpd = pbn.LinearGaussianCPD('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.LinearGaussianCPD('d', ['c', 'a', 'b'])
    cpd2.fit(df)

    assert np.all(np.isclose(cpd.slogl(df_null), cpd2.slogl(df_null))), "The order of the evidence changes the slogl() result."

def test_lg_cdf():
    test_df = util_test.generate_normal_data(5000)

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = pbn.LinearGaussianCPD(variable, evidence)
        cpd.fit(df)

        beta = cpd.beta
        variance = cpd.variance

        assert np.all(np.isclose(cpd.cdf(test_df), numpy_cdf(test_df, variable, evidence, beta, variance))),\
                     "Wrong cdf for LinearGaussianCPD(" + str(variable) + " | " + str(evidence) + ")"

    
    cpd = pbn.LinearGaussianCPD('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.LinearGaussianCPD('d', ['c', 'a', 'b'])
    cpd2.fit(df)

    assert np.all(np.isclose(cpd.cdf(test_df), cpd2.cdf(test_df))), "The order of the evidence changes the cdf() result."

def test_lg_cdf_null():
    test_df = util_test.generate_normal_data(5000)

    np.random.seed(0)
    a_null = np.random.randint(0, 5000, size=100)
    b_null = np.random.randint(0, 5000, size=100)
    c_null = np.random.randint(0, 5000, size=100)
    d_null = np.random.randint(0, 5000, size=100)

    df_null = test_df.copy()
    df_null.loc[df_null.index[a_null], 'a'] = np.nan
    df_null.loc[df_null.index[b_null], 'b'] = np.nan
    df_null.loc[df_null.index[c_null], 'c'] = np.nan
    df_null.loc[df_null.index[d_null], 'd'] = np.nan

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = pbn.LinearGaussianCPD(variable, evidence)
        cpd.fit(df)

        beta = cpd.beta
        variance = cpd.variance

        assert np.all(np.isclose(
                        cpd.cdf(df_null), 
                        numpy_cdf(df_null, variable, evidence, beta, variance), equal_nan=True)),\
                     "Wrong cdf for LinearGaussianCPD(" + str(variable) + " | " + str(evidence) + ") with null values."

    cpd = pbn.LinearGaussianCPD('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.LinearGaussianCPD('d', ['c', 'a', 'b'])
    cpd2.fit(df)

    assert np.all(np.isclose(
                        cpd.cdf(df_null), 
                        cpd2.cdf(df_null), equal_nan=True)),\
                     "The order of the evidence changes the cdf() result."

def test_lg_sample():
    SAMPLE_SIZE = 1000

    cpd = pbn.LinearGaussianCPD('a', [])
    cpd.fit(df)
    
    sampled = cpd.sample(SAMPLE_SIZE, None, 0)

    assert sampled.type == pa.float64()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE
        
    cpd = pbn.LinearGaussianCPD('b', ['a'])
    cpd.fit(df)

    sampling_df = pd.DataFrame({'a': np.full((SAMPLE_SIZE,), 3.0)})
    sampled = cpd.sample(SAMPLE_SIZE, sampling_df, 0)

    assert sampled.type == pa.float64()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE
    
    cpd = pbn.LinearGaussianCPD('c', ['a', 'b'])
    cpd.fit(df)

    sampling_df = pd.DataFrame({'a': np.full((SAMPLE_SIZE,), 3.0),
                                'b': np.full((SAMPLE_SIZE,), 7.45)})
    sampled = cpd.sample(SAMPLE_SIZE, sampling_df, 0)

    assert sampled.type == pa.float64()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE