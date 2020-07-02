import numpy as np
import pyarrow as pa
from pgm_dataset.factors.continuous import KDE
from scipy.stats import gaussian_kde

import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)

def test_kde_variables():
    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        cpd = KDE(variables)
        assert cpd.variables == variables

def test_kde_bandwidth():

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        for instances in [50, 1000, 10000]:
            npdata = df.loc[:, variables].to_numpy()
            scipy_kde = gaussian_kde(npdata[:instances, :].T)

            cpd = KDE(variables)
            cpd.fit(df.iloc[:instances])

            assert np.all(np.isclose(cpd.bandwidth, scipy_kde.covariance)), "Wrong bandwidth computed with Scott's rule."

    cpd = KDE(['a'])
    cpd.fit(df)
    cpd.bandwidth = [[1]]

    assert cpd.bandwidth == np.asarray([[1]]), "Could not change bandwidth."

def test_fit():
    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        for instances in [50, 1000, 10000]:
            cpd = KDE(variables)
            assert not cpd.fitted
            cpd.fit(df.iloc[:instances,:])
            assert cpd.fitted

            npdata = df.loc[:, variables].to_numpy()
            scipy_kde = gaussian_kde(npdata[:instances, :].T)

            assert scipy_kde.n == cpd.n, "Wrong number of training instances."
            assert scipy_kde.d == cpd.d, "Wrong number of training variables."

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

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        for instances in [50, 1000, 10000]:
            cpd = KDE(variables)
            assert not cpd.fitted
            cpd.fit(df.iloc[:instances,:])
            assert cpd.fitted

            npdata = df.loc[:, variables].to_numpy()
            npdata_instances = npdata[:instances,:]

            nan_rows = np.any(np.isnan(npdata_instances), axis=1)
            npdata_no_null = npdata_instances[~nan_rows,:]
            scipy_kde = gaussian_kde(npdata_no_null.T)

            assert scipy_kde.n == cpd.n, "Wrong number of training instances with null values."
            assert scipy_kde.d == cpd.d, "Wrong number of training variables with null values."
            assert np.all(np.isclose(scipy_kde.covariance, cpd.bandwidth)), "Wrong bandwidth with null values."

def test_logpdf():
    test_df = util_test.generate_normal_data(50)

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        cpd = KDE(variables)
        cpd.fit(df)

        npdata = df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata.T)

        test_npdata = test_df.loc[:, variables].to_numpy()
        assert np.all(np.isclose(cpd.logpdf(test_df), scipy_kde.logpdf(test_npdata.T)))

    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)

    assert np.all(np.isclose(cpd.logpdf(test_df), cpd2.logpdf(test_df))), "Order of evidence changes logpdf() result."

def test_logpdf_null():
    test_df = util_test.generate_normal_data(50)

    np.random.seed(0)
    a_null = np.random.randint(0, 50, size=10)
    b_null = np.random.randint(0, 50, size=10)
    c_null = np.random.randint(0, 50, size=10)
    d_null = np.random.randint(0, 50, size=10)

    df_null = test_df.copy()
    df_null.loc[df_null.index[a_null], 'a'] = np.nan
    df_null.loc[df_null.index[b_null], 'b'] = np.nan
    df_null.loc[df_null.index[c_null], 'c'] = np.nan
    df_null.loc[df_null.index[d_null], 'd'] = np.nan

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        cpd = KDE(variables)
        cpd.fit(df)

        npdata = df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata.T)

        test_npdata = test_df.loc[:, variables].to_numpy()

        assert np.all(np.isnan(cpd.logpdf(test_df)) == np.isnan(scipy_kde.logpdf(test_npdata.T)))
        nan_indices = np.any(np.isnan(test_npdata), axis=1)
        assert np.all(np.isclose(cpd.logpdf(test_df)[~nan_indices], scipy_kde.logpdf(test_npdata.T)[~nan_indices]))


    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)

    assert np.all(np.isnan(cpd.logpdf(test_df)) == np.isnan(cpd2.logpdf(test_df))), "Order of evidence changes the position of nan values."

    test_npdata = test_df.loc[:, variables].to_numpy()
    nan_indices = np.any(np.isnan(test_npdata), axis=1)
    assert np.all(np.isclose(cpd.logpdf(test_df)[~nan_indices], cpd2.logpdf(test_df)[~nan_indices])), "Order of evidence changes logpdf() result."

def test_slogpdf():
    test_df = util_test.generate_normal_data(50)

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        cpd = KDE(variables)
        cpd.fit(df)

        npdata = df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata.T)

        test_npdata = test_df.loc[:, variables].to_numpy()
        assert np.all(np.isclose(cpd.slogpdf(test_df), scipy_kde.logpdf(test_npdata.T).sum()))

    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)

    assert np.all(np.isclose(cpd.slogpdf(test_df), cpd2.slogpdf(test_df))), "Order of evidence changes slogpdf() result."

def test_slogpdf_null():
    test_df = util_test.generate_normal_data(50)

    np.random.seed(0)
    a_null = np.random.randint(0, 50, size=10)
    b_null = np.random.randint(0, 50, size=10)
    c_null = np.random.randint(0, 50, size=10)
    d_null = np.random.randint(0, 50, size=10)

    df_null = test_df.copy()
    df_null.loc[df_null.index[a_null], 'a'] = np.nan
    df_null.loc[df_null.index[b_null], 'b'] = np.nan
    df_null.loc[df_null.index[c_null], 'c'] = np.nan
    df_null.loc[df_null.index[d_null], 'd'] = np.nan

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        cpd = KDE(variables)
        cpd.fit(df)

        npdata = df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata.T)

        test_npdata = test_df.loc[:, variables].to_numpy()

        assert np.all(np.isclose(cpd.slogpdf(test_df), np.nansum(scipy_kde.logpdf(test_npdata.T))))


    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)

    assert np.all(np.isclose(cpd.slogpdf(test_df), cpd2.slogpdf(test_df))), "Order of evidence changes slogpdf() result."
