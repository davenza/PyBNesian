import pytest
import numpy as np
from pybnesian.factors.continuous import KDE
from scipy.stats import gaussian_kde

import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE, seed=0)
df_float = df.astype('float32')

def test_check_type():
    cpd = KDE(['a'])
    cpd.fit(df)
    with pytest.raises(ValueError) as ex:
        cpd.logl(df_float)
    assert "Data type of training and test datasets is different." in str(ex.value)
    with pytest.raises(ValueError) as ex:
        cpd.slogl(df_float)
    assert "Data type of training and test datasets is different." in str(ex.value)

    cpd.fit(df_float)
    with pytest.raises(ValueError) as ex:
        cpd.logl(df)
    assert "Data type of training and test datasets is different." in str(ex.value)
    with pytest.raises(ValueError) as ex:
        cpd.slogl(df)
    assert "Data type of training and test datasets is different." in str(ex.value)

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

            cpd.fit(df_float.iloc[:instances])
            assert np.all(np.isclose(cpd.bandwidth, scipy_kde.covariance)), "Wrong bandwidth computed with Scott's rule."

    cpd = KDE(['a'])
    cpd.fit(df)
    cpd.bandwidth = [[1]]
    assert cpd.bandwidth == np.asarray([[1]]), "Could not change bandwidth."

    cpd.fit(df_float)
    cpd.bandwidth = [[1]]
    assert cpd.bandwidth == np.asarray([[1]]), "Could not change bandwidth."

def test_kde_fit():
    def _test_kde_fit_iter(variables, _df, instances):
        cpd = KDE(variables)
        assert not cpd.fitted
        cpd.fit(_df.iloc[:instances,:])
        assert cpd.fitted

        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata[:instances, :].T)

        assert scipy_kde.n == cpd.N, "Wrong number of training instances."
        assert scipy_kde.d == cpd.d, "Wrong number of training variables."

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        for instances in [50, 1000, 10000]:
            _test_kde_fit_iter(variables, df, instances)
            _test_kde_fit_iter(variables, df_float, instances)

def test_kde_fit_null():
    def _test_kde_fit_null_iter(variables, _df, instances):
        cpd = KDE(variables)
        assert not cpd.fitted
        cpd.fit(_df.iloc[:instances,:])
        assert cpd.fitted

        npdata = _df.loc[:, variables].to_numpy()
        npdata_instances = npdata[:instances,:]

        nan_rows = np.any(np.isnan(npdata_instances), axis=1)
        npdata_no_null = npdata_instances[~nan_rows,:]
        scipy_kde = gaussian_kde(npdata_no_null.T)

        assert scipy_kde.n == cpd.N, "Wrong number of training instances with null values."
        assert scipy_kde.d == cpd.d, "Wrong number of training variables with null values."
        assert np.all(np.isclose(scipy_kde.covariance, cpd.bandwidth)), "Wrong bandwidth with null values."

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

    df_null_float = df_float.copy()
    df_null_float.loc[df_null_float.index[a_null], 'a'] = np.nan
    df_null_float.loc[df_null_float.index[b_null], 'b'] = np.nan
    df_null_float.loc[df_null_float.index[c_null], 'c'] = np.nan
    df_null_float.loc[df_null_float.index[d_null], 'd'] = np.nan

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        for instances in [50, 1000, 10000]:
            _test_kde_fit_null_iter(variables, df_null, instances)
            _test_kde_fit_null_iter(variables, df_null_float, instances)

def test_kde_logl():
    def _test_kde_logl_iter(variables, _df, _test_df):
        cpd = KDE(variables)
        cpd.fit(_df)

        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata.T)

        test_npdata = _test_df.loc[:, variables].to_numpy()

        logl = cpd.logl(_test_df)
        scipy = scipy_kde.logpdf(test_npdata.T)

        if np.all(_df.dtypes == 'float32'):
            assert np.all(np.isclose(logl, scipy, atol=0.0005))
        else:
            assert np.all(np.isclose(logl, scipy))

    test_df = util_test.generate_normal_data(50, seed=1)
    test_df_float = test_df.astype('float32')

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        _test_kde_logl_iter(variables, df, test_df)
        _test_kde_logl_iter(variables, df_float, test_df_float)

    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.logl(test_df), cpd2.logl(test_df))), "Order of evidence changes logl() result."

    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.logl(test_df_float), cpd2.logl(test_df_float))), "Order of evidence changes logl() result."

def test_kde_logl_null():
    def _test_kde_logl_null_iter(variables, _df, _test_df):
        cpd = KDE(variables)
        cpd.fit(_df)

        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata.T)

        test_npdata = _test_df.loc[:, variables].to_numpy()

        logl = cpd.logl(_test_df)
        scipy = scipy_kde.logpdf(test_npdata.T)

        if npdata.dtype == "float32":
            assert np.all(np.isclose(logl, scipy, atol=0.0005, equal_nan=True))
        else:
            assert np.all(np.isclose(logl, scipy, equal_nan=True))

    TEST_SIZE = 50

    test_df = util_test.generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype('float32')

    np.random.seed(0)
    a_null = np.random.randint(0, TEST_SIZE, size=10)
    b_null = np.random.randint(0, TEST_SIZE, size=10)
    c_null = np.random.randint(0, TEST_SIZE, size=10)
    d_null = np.random.randint(0, TEST_SIZE, size=10)

    df_null = test_df.copy()
    df_null.loc[df_null.index[a_null], 'a'] = np.nan
    df_null.loc[df_null.index[b_null], 'b'] = np.nan
    df_null.loc[df_null.index[c_null], 'c'] = np.nan
    df_null.loc[df_null.index[d_null], 'd'] = np.nan

    df_null_float = test_df_float.copy()
    df_null_float.loc[df_null_float.index[a_null], 'a'] = np.nan
    df_null_float.loc[df_null_float.index[b_null], 'b'] = np.nan
    df_null_float.loc[df_null_float.index[c_null], 'c'] = np.nan
    df_null_float.loc[df_null_float.index[d_null], 'd'] = np.nan

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        _test_kde_logl_null_iter(variables, df, df_null)
        _test_kde_logl_null_iter(variables, df_float, df_null_float)


    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.logl(df_null), cpd2.logl(df_null), equal_nan=True)), "Order of evidence changes logl() result."

    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.logl(df_null_float), cpd2.logl(df_null_float), atol=0.0005, equal_nan=True)), "Order of evidence changes logl() result."

def test_kde_slogl():
    def _test_kde_slogl_iter(variables, _df, _test_df):
        cpd = KDE(variables)
        cpd.fit(_df)

        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata.T)

        test_npdata = _test_df.loc[:, variables].to_numpy()
        assert np.all(np.isclose(cpd.slogl(_test_df), scipy_kde.logpdf(test_npdata.T).sum()))


    test_df = util_test.generate_normal_data(50, seed=1)
    test_df_float = test_df.astype('float32')

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        _test_kde_slogl_iter(variables, df, test_df)
        _test_kde_slogl_iter(variables, df_float, test_df_float)

    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.slogl(test_df), cpd2.slogl(test_df))), "Order of evidence changes slogl() result."

    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.slogl(test_df_float), cpd2.slogl(test_df_float))), "Order of evidence changes slogl() result."


def test_kde_slogl_null():
    def _test_kde_slogl_null_iter(variables, _df, _test_df):
        cpd = KDE(variables)
        cpd.fit(_df)

        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata.T)

        test_npdata = _test_df.loc[:, variables].to_numpy()

        assert np.all(np.isclose(cpd.slogl(_test_df), np.nansum(scipy_kde.logpdf(test_npdata.T))))

    TEST_SIZE = 50

    test_df = util_test.generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype('float32')

    np.random.seed(0)
    a_null = np.random.randint(0, TEST_SIZE, size=10)
    b_null = np.random.randint(0, TEST_SIZE, size=10)
    c_null = np.random.randint(0, TEST_SIZE, size=10)
    d_null = np.random.randint(0, TEST_SIZE, size=10)

    df_null = test_df.copy()
    df_null.loc[df_null.index[a_null], 'a'] = np.nan
    df_null.loc[df_null.index[b_null], 'b'] = np.nan
    df_null.loc[df_null.index[c_null], 'c'] = np.nan
    df_null.loc[df_null.index[d_null], 'd'] = np.nan

    df_null_float = test_df_float.copy()
    df_null_float.loc[df_null_float.index[a_null], 'a'] = np.nan
    df_null_float.loc[df_null_float.index[b_null], 'b'] = np.nan
    df_null_float.loc[df_null_float.index[c_null], 'c'] = np.nan
    df_null_float.loc[df_null_float.index[d_null], 'd'] = np.nan

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        _test_kde_slogl_null_iter(variables, df, df_null)
        _test_kde_slogl_null_iter(variables, df_float, df_null_float)


    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.slogl(df_null), cpd2.slogl(df_null))), "Order of evidence changes slogl() result."

    cpd = KDE(['d', 'a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = KDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.slogl(df_null_float), cpd2.slogl(df_null_float))), "Order of evidence changes slogl() result."
