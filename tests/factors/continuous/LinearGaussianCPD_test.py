import numpy as np
import pandas as pd
import pyarrow as pa
from pgm_dataset.factors.continuous import LinearGaussianCPD

import util_test

SIZE = 10000

df = util_test.generate_normal_data(SIZE)

def fit_numpy(df, variable, evidence):
    df = df.dropna()
    linregress_data = np.column_stack((np.ones(df.shape[0]), df.loc[:, evidence]))
    (beta, res, _, _) = np.linalg.lstsq(linregress_data, df.loc[:, variable], rcond=None)
    
    return beta, res / (df.shape[0] - len(evidence) - 1)

def test_fit():
    
    a_cpd = LinearGaussianCPD("a", [])
    a_cpd.fit(df)

    npbeta, npvar = fit_numpy(df, "a", [])

    assert np.all(np.isclose(npbeta, a_cpd.beta)), "Wrong beta vector."
    assert np.all(np.isclose(npvar, a_cpd.variance)), "Wrong variance."

    b_cpd = LinearGaussianCPD("b", ["a"])
    b_cpd.fit(df)

    npbeta, npvar = fit_numpy(df, "b", ["a"])

    assert np.all(np.isclose(npbeta, b_cpd.beta)), "Wrong beta vector."
    assert np.all(np.isclose(npvar, b_cpd.variance)), "Wrong variance."

    c_cpd = LinearGaussianCPD("c", ["a", "b"])
    c_cpd.fit(df)

    npbeta, npvar = fit_numpy(df, "c", ["a", "b"])

    assert np.all(np.isclose(npbeta, c_cpd.beta)), "Wrong beta vector."
    assert np.all(np.isclose(npvar, c_cpd.variance)), "Wrong variance."


    d_cpd = LinearGaussianCPD("d", ["a", "b", "c"])
    d_cpd.fit(df)

    npbeta, npvar = fit_numpy(df, "d", ["a", "b", "c"])

    assert np.all(np.isclose(npbeta, d_cpd.beta)), "Wrong beta vector."
    assert np.all(np.isclose(npvar, d_cpd.variance)), "Wrong variance."

def test_fit_null():
    np.random.seed(0)
    a_null = np.random.randint(0, SIZE, size=100)
    b_null = np.random.randint(0, SIZE, size=100)
    c_null = np.random.randint(0, SIZE, size=100)
    d_null = np.random.randint(0, SIZE, size=100)

    df_null = df
    df_null.loc[df_null.index[a_null], 'a'] = np.nan
    df_null.loc[df_null.index[b_null], 'b'] = np.nan
    df_null.loc[df_null.index[c_null], 'c'] = np.nan
    df_null.loc[df_null.index[d_null], 'd'] = np.nan

    a_cpd = LinearGaussianCPD("a", [])
    a_cpd.fit(df_null)

    npbeta, npvar = fit_numpy(df_null, "a", [])

    assert np.all(np.isclose(npbeta, a_cpd.beta)), "Wrong beta vector."
    assert np.all(np.isclose(npvar, a_cpd.variance)), "Wrong variance."

    b_cpd = LinearGaussianCPD("b", ["a"])
    b_cpd.fit(df_null)

    npbeta, npvar = fit_numpy(df_null, "b", ["a"])

    assert np.all(np.isclose(npbeta, b_cpd.beta)), "Wrong beta vector."
    assert np.all(np.isclose(npvar, b_cpd.variance)), "Wrong variance."

    c_cpd = LinearGaussianCPD("c", ["a", "b"])
    c_cpd.fit(df_null)

    npbeta, npvar = fit_numpy(df_null, "c", ["a", "b"])

    assert np.all(np.isclose(npbeta, c_cpd.beta)), "Wrong beta vector."
    assert np.all(np.isclose(npvar, c_cpd.variance)), "Wrong variance."


    d_cpd = LinearGaussianCPD("d", ["a", "b", "c"])
    d_cpd.fit(df_null)

    npbeta, npvar = fit_numpy(df_null, "d", ["a", "b", "c"])

    assert np.all(np.isclose(npbeta, d_cpd.beta)), "Wrong beta vector."
    assert np.all(np.isclose(npvar, d_cpd.variance)), "Wrong variance."