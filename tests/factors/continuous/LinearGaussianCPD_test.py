import numpy as np
import pandas as pd
import pyarrow as pa
from pgm_dataset.factors.continuous import LinearGaussianCPD

import pytest
from scipy.stats import norm

import util_test

SIZE = 10000

df = util_test.generate_normal_data(SIZE)


def test_variable():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = LinearGaussianCPD(variable, evidence)
        assert cpd.variable == variable

def test_evidence():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = LinearGaussianCPD(variable, evidence)
        assert cpd.evidence == evidence


def fit_numpy(df, variable, evidence):
    df = df.loc[:, [variable] + evidence].dropna()
    linregress_data = np.column_stack((np.ones(df.shape[0]), df.loc[:, evidence]))
    (beta, res, _, _) = np.linalg.lstsq(linregress_data, df.loc[:, variable], rcond=None)
    
    return beta, res / (df.shape[0] - len(evidence) - 1)

def test_fit():
    for variable, evidence in [("a", []), ("b", ["a"]), ("c", ["a", "b"]), ("d", ["a", "b", "c"])]:
        cpd = LinearGaussianCPD(variable, evidence)
        cpd.fit(df)

        npbeta, npvar = fit_numpy(df, variable, evidence)
        
        assert np.all(np.isclose(npbeta, cpd.beta)), "Wrong beta vector."
        assert np.all(np.isclose(npvar, cpd.variance)), "Wrong variance."


def test_fit_null():
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
        cpd = LinearGaussianCPD(variable, evidence)
        cpd.fit(df)

        npbeta, npvar = fit_numpy(df, variable, evidence)
        
        assert np.all(np.isclose(npbeta, cpd.beta)), "Wrong beta vector."
        assert np.all(np.isclose(npvar, cpd.variance)), "Wrong variance."

def numpy_logpdf(test_df, variable, evidence, beta, variance):
    npdata = test_df.loc[:, evidence].to_numpy()
    means = beta[0] + np.sum(beta[1:]*npdata, axis=1)

    result = np.empty((test_df.shape[0],))

    isnan_vec = np.full((test_df.shape[0],), False, dtype=np.bool)
    isnan_vec[np.isnan(means)] = True
    isnan_vec[np.isnan(test_df.loc[:, variable].to_numpy())] = True

    result[isnan_vec] = np.nan
    result[~isnan_vec] = norm.logpdf(test_df.loc[:, variable].to_numpy()[~isnan_vec], means[~isnan_vec], np.sqrt(variance))
    return result

def test_logpdf():
    test_df = util_test.generate_normal_data(5000)

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = LinearGaussianCPD(variable, evidence)
        cpd.fit(test_df)

        beta = cpd.beta
        variance = cpd.variance

        assert np.all(np.isclose(cpd.logpdf(test_df), numpy_logpdf(test_df, variable, evidence, beta, variance))),\
                     "Wrong logpdf for LinearGaussianCPD(" + str(variable) + " | " + str(evidence) + ")"

    
    cpd = LinearGaussianCPD('d', ['a', 'b', 'c'])
    cpd.fit(test_df)
    cpd2 = LinearGaussianCPD('d', ['c', 'a', 'b'])
    cpd2.fit(test_df)

    assert np.all(np.isclose(cpd.logpdf(test_df), cpd2.logpdf(test_df))), "The order of the evidence changes the logpdf() result."

def test_logpdf_null():
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
        cpd = LinearGaussianCPD(variable, evidence)
        cpd.fit(test_df)

        beta = cpd.beta
        variance = cpd.variance

        assert np.all(np.isnan(cpd.logpdf(df_null)) == np.isnan(numpy_logpdf(df_null, variable, evidence, beta, variance))),\
                                                            "Wrong positions for nan values in LinearGaussianCPD::logpdf()."

        non_nan = ~np.isnan(cpd.logpdf(df_null))
        assert np.all(np.isclose(
                        cpd.logpdf(test_df)[non_nan], 
                        numpy_logpdf(test_df, variable, evidence, beta, variance)[non_nan])),\
                     "Wrong logpdf for LinearGaussianCPD(" + str(variable) + " | " + str(evidence) + ") with null values."

    cpd = LinearGaussianCPD('d', ['a', 'b', 'c'])
    cpd.fit(test_df)
    cpd2 = LinearGaussianCPD('d', ['c', 'a', 'b'])
    cpd2.fit(test_df)

    assert np.all(np.isnan(cpd.logpdf(df_null)) == np.isnan(cpd2.logpdf(df_null))),\
                                                         "The order of the evidence changes the logpdf() result."


    non_nan = ~np.isnan(cpd.logpdf(df_null))
    assert np.all(np.isclose(
                        cpd.logpdf(test_df)[non_nan], 
                        cpd2.logpdf(test_df)[non_nan])),\
                     "The order of the evidence changes the logpdf() result."

def test_slogpdf():
    test_df = util_test.generate_normal_data(5000)

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = LinearGaussianCPD(variable, evidence)
        cpd.fit(test_df)

        beta = cpd.beta
        variance = cpd.variance

        assert np.all(np.isclose(cpd.slogpdf(test_df), np.sum(numpy_logpdf(test_df, variable, evidence, beta, variance)))),\
                     "Wrong slogpdf for LinearGaussianCPD(" + str(variable) + " | " + str(evidence) + ")"

    cpd = LinearGaussianCPD('d', ['a', 'b', 'c'])
    cpd.fit(test_df)
    cpd2 = LinearGaussianCPD('d', ['c', 'a', 'b'])
    cpd2.fit(test_df)

    assert np.all(np.isclose(cpd.slogpdf(test_df), cpd2.slogpdf(test_df))), "The order of the evidence changes the slogpdf() result."

def test_slogpdf_null():
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
        cpd = LinearGaussianCPD(variable, evidence)
        cpd.fit(test_df)

        beta = cpd.beta
        variance = cpd.variance

        assert np.all(np.isclose(cpd.slogpdf(test_df), np.nansum(numpy_logpdf(test_df, variable, evidence, beta, variance)))),\
                     "Wrong slogpdf for LinearGaussianCPD(" + str(variable) + " | " + str(evidence) + ") with null values."

    cpd = LinearGaussianCPD('d', ['a', 'b', 'c'])
    cpd.fit(test_df)
    cpd2 = LinearGaussianCPD('d', ['c', 'a', 'b'])
    cpd2.fit(test_df)

    assert np.all(np.isclose(cpd.slogpdf(test_df), cpd2.slogpdf(test_df))), "The order of the evidence changes the slogpdf() result."