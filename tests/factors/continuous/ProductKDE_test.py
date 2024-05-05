import pytest
import numpy as np
import pyarrow as pa
import pybnesian as pbn
from pybnesian import BandwidthSelector
from scipy.stats import gaussian_kde
from functools import reduce

import util_test

SIZE = 500
df = util_test.generate_normal_data(SIZE, seed=0)
df_float = df.astype('float32')

def test_check_type():
    cpd = pbn.ProductKDE(['a'])
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

def test_productkde_variables():
    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        cpd = pbn.ProductKDE(variables)
        assert cpd.variables() == variables

def py_nr_bandwidth(df, variables):
    cov = df[variables].cov().to_numpy()
    delta = np.linalg.inv(np.diag(np.diag(cov))).dot(cov)
    delta_inv = np.linalg.inv(delta)
    N = df.shape[0]
    d = len(variables)

    k = 4*d*np.sqrt(np.linalg.det(delta))/ (2*(delta_inv.dot(delta_inv)).trace() + delta_inv.trace()**2)
    return np.power(k / N, 2 / (d + 4)) * np.diag(cov)

def py_scott_bandwidth(df, variables):
    var = df[variables].var().to_numpy()
    N = df.shape[0]
    d = len(variables)

    return np.power(N, -2 / (d + 4)) * var

def test_productkde_bandwidth():
    # for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
    for variables in [['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        for instances in [50, 150, 500]:
            cpd = pbn.ProductKDE(variables)
            cpd.fit(df.iloc[:instances])
            assert np.all(np.isclose(cpd.bandwidth, py_nr_bandwidth(df[:instances], variables))), "Wrong bandwidth computed with normal reference rule."

            cpd.fit(df_float.iloc[:instances])
            assert np.all(np.isclose(cpd.bandwidth, py_nr_bandwidth(df[:instances], variables), atol=0.0005)), "Wrong bandwidth computed with normal reference rule."

            cpd = pbn.ProductKDE(variables, pbn.ScottsBandwidth())
            cpd.fit(df.iloc[:instances])
            assert np.all(np.isclose(cpd.bandwidth, py_scott_bandwidth(df[:instances], variables))), "Wrong bandwidth computed with Scott's rule."

            cpd.fit(df_float.iloc[:instances])
            assert np.all(np.isclose(cpd.bandwidth, py_scott_bandwidth(df[:instances], variables), atol=0.0005)), "Wrong bandwidth computed with Scott's rule."


    cpd = pbn.ProductKDE(['a'])
    cpd.fit(df)
    cpd.bandwidth = [1]
    assert cpd.bandwidth == np.asarray([1]), "Could not change bandwidth."

    cpd.fit(df_float)
    cpd.bandwidth = [1]
    assert cpd.bandwidth == np.asarray([1]), "Could not change bandwidth."

class UnitaryBandwidth(BandwidthSelector):
    def __init__(self):
        BandwidthSelector.__init__(self)

    def diag_bandwidth(self, df, variables):
        return np.ones((len(variables),))
    
def test_productkde_new_bandwidth():
    kde = pbn.ProductKDE(["a"], UnitaryBandwidth())
    kde.fit(df)
    assert kde.bandwidth == np.ones((1,))
    
    kde.fit(df_float)
    assert kde.bandwidth == np.ones((1,))

    kde = pbn.ProductKDE(["a", "b", "c", "d"], UnitaryBandwidth())
    kde.fit(df)
    assert np.all(kde.bandwidth == np.ones((4,)))
    
    kde.fit(df_float)
    assert np.all(kde.bandwidth == np.ones((4,)))

def test_productkde_data_type():
    k = pbn.ProductKDE(["a"])

    with pytest.raises(ValueError) as ex:
        k.data_type()
    "KDE factor not fitted" in str(ex.value)

    k.fit(df)
    assert k.data_type() == pa.float64()
    k.fit(df_float)
    assert k.data_type() == pa.float32()

def test_productkde_fit():
    def _test_productkde_fit_iter(variables, _df, instances):
        cpd = pbn.ProductKDE(variables)
        assert not cpd.fitted()
        cpd.fit(_df.iloc[:instances,:])
        assert cpd.fitted()

        assert instances == cpd.num_instances(), "Wrong number of training instances."
        assert len(variables) == cpd.num_variables(), "Wrong number of training variables."
        if np.all(_df.dtypes == 'float32'):
            assert np.all(np.isclose(cpd.bandwidth, py_nr_bandwidth(_df.iloc[:instances], variables), atol=0.0005)), "Wrong bandwidth."
        else:
            assert np.all(np.isclose(cpd.bandwidth, py_nr_bandwidth(_df.iloc[:instances], variables))), "Wrong bandwidth."

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        for instances in [50, 150, 500]:
            _test_productkde_fit_iter(variables, df, instances)
            _test_productkde_fit_iter(variables, df_float, instances)

def test_productkde_fit_null():
    def _test_productkde_fit_null_iter(variables, _df, instances):
        cpd = pbn.ProductKDE(variables)
        assert not cpd.fitted()
        cpd.fit(_df.iloc[:instances,:])
        assert cpd.fitted()

        npdata = _df.loc[:, variables].to_numpy()
        npdata_instances = npdata[:instances,:]

        nan_rows = np.any(np.isnan(npdata_instances), axis=1)
        nonnan_indices = np.where(~nan_rows)[0]

        assert (~nan_rows).sum() == cpd.num_instances(), "Wrong number of training instances with null values."
        assert len(variables) == cpd.num_variables(), "Wrong number of training variables with null values."
        if np.all(_df.dtypes == 'float32'):
            assert np.all(np.isclose(cpd.bandwidth, py_nr_bandwidth(_df.iloc[nonnan_indices,:], variables), atol=0.0005)), "Wrong bandwidth with null values."
        else:
            assert np.all(np.isclose(cpd.bandwidth, py_nr_bandwidth(_df.iloc[nonnan_indices,:], variables))), "Wrong bandwidth with null values."

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
        for instances in [50, 150, 500]:
            _test_productkde_fit_null_iter(variables, df_null, instances)
            _test_productkde_fit_null_iter(variables, df_null_float, instances)


def factor_product_kernel(train_data):
    # Estimate bandwidth factor using Equation (3.4) of Chacon and Duong (2018)

    cov_data = np.atleast_2d(np.cov(train_data, rowvar=False, bias=False))
    delta = np.diag(np.reciprocal(np.diag(cov_data))).dot(cov_data)
    delta_inv = np.linalg.inv(delta)

    N = train_data.shape[0]
    d = train_data.shape[1]

    num_factor = 4 * d * np.sqrt(np.linalg.det(delta))
    denom_factor = (2 * np.trace(np.dot(delta_inv, delta_inv)) + np.trace(delta_inv)**2)

    k = num_factor / denom_factor

    return (k / N)**(1. / (d + 4.))


def test_productkde_logl():
    def _test_productkde_logl_iter(variables, _df, _test_df):
        cpd = pbn.ProductKDE(variables)
        cpd.fit(_df)

        logl = cpd.logl(_test_df)

        npdata = _df.loc[:, variables].to_numpy()

        factor = factor_product_kernel(npdata)

        final_scipy_kde = gaussian_kde(npdata.T, 
                                       bw_method=lambda gkde: factor * np.eye(npdata.shape[1], dtype=npdata.dtype))
        final_scipy_kde.cho_cov = np.linalg.cholesky(final_scipy_kde.covariance)
        final_scipy_kde.log_det = 2*np.log(np.diag(final_scipy_kde.cho_cov * np.sqrt(2*np.pi))).sum()

        test_npdata = _test_df.loc[:, variables].to_numpy()
        scipy = final_scipy_kde.logpdf(test_npdata.T)

        if np.all(_df.dtypes == 'float32'):
            assert np.all(np.isclose(logl, scipy, atol=5e-3))
        else:
            assert np.all(np.isclose(logl, scipy))

    test_df = util_test.generate_normal_data(50, seed=1)
    test_df_float = test_df.astype('float32')

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        _test_productkde_logl_iter(variables, df, test_df)
        _test_productkde_logl_iter(variables, df_float, test_df_float)

    cpd = pbn.ProductKDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.ProductKDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.logl(test_df), cpd2.logl(test_df))), "Order of evidence changes logl() result."

    cpd = pbn.ProductKDE(['d', 'a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = pbn.ProductKDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.logl(test_df_float), cpd2.logl(test_df_float), atol=0.0005)), "Order of evidence changes logl() result."

def test_productkde_logl_null():
    def _test_productkde_logl_null_iter(variables, _df, _test_df):
        cpd = pbn.ProductKDE(variables)
        cpd.fit(_df)

        logl = cpd.logl(_test_df)

        npdata = _df.loc[:, variables].to_numpy()

        factor = factor_product_kernel(npdata)
        final_scipy_kde = gaussian_kde(npdata.T, 
                                       bw_method=lambda gkde: factor * np.eye(npdata.shape[1], dtype=npdata.dtype))
        final_scipy_kde.cho_cov = np.linalg.cholesky(final_scipy_kde.covariance)
        final_scipy_kde.log_det = 2*np.log(np.diag(final_scipy_kde.cho_cov * np.sqrt(2*np.pi))).sum()

        test_npdata = _test_df.loc[:, variables].to_numpy()

        scipy = np.full((test_npdata.shape[0],), np.nan)
        nan_rows = np.any(np.isnan(test_npdata), axis=1)
        scipy[~nan_rows] = final_scipy_kde.logpdf(test_npdata[~nan_rows].T)

        if npdata.dtype == "float32":
            assert np.all(np.isclose(logl, scipy, atol=5e-3, equal_nan=True))
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
        _test_productkde_logl_null_iter(variables, df, df_null)
        _test_productkde_logl_null_iter(variables, df_float, df_null_float)

    cpd = pbn.ProductKDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.ProductKDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.logl(df_null), cpd2.logl(df_null), equal_nan=True)), "Order of evidence changes logl() result."

    cpd = pbn.ProductKDE(['d', 'a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = pbn.ProductKDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.logl(df_null_float), cpd2.logl(df_null_float), atol=0.0005, equal_nan=True)), "Order of evidence changes logl() result."

def test_productkde_slogl():
    def _test_productkde_slogl_iter(variables, _df, _test_df):
        cpd = pbn.ProductKDE(variables)
        cpd.fit(_df)

        npdata = _df.loc[:, variables].to_numpy()
        
        factor = factor_product_kernel(npdata)
        final_scipy_kde = gaussian_kde(npdata.T, 
                                       bw_method=lambda gkde: factor * np.eye(npdata.shape[1], dtype=npdata.dtype))
        final_scipy_kde.cho_cov = np.linalg.cholesky(final_scipy_kde.covariance)
        final_scipy_kde.log_det = 2*np.log(np.diag(final_scipy_kde.cho_cov * np.sqrt(2*np.pi))).sum()

        test_npdata = _test_df.loc[:, variables].to_numpy()

        if np.all(_df.dtypes == 'float32'):
            assert np.all(np.isclose(cpd.slogl(_test_df), final_scipy_kde.logpdf(test_npdata.T).sum(),
                                     atol=5e-3*test_npdata.shape[0]))
        else:
            assert np.all(np.isclose(cpd.slogl(_test_df), final_scipy_kde.logpdf(test_npdata.T).sum()))

    test_df = util_test.generate_normal_data(50, seed=1)
    test_df_float = test_df.astype('float32')

    for variables in [['a'], ['b', 'a'], ['c', 'a', 'b'], ['d', 'a', 'b', 'c']]:
        _test_productkde_slogl_iter(variables, df, test_df)
        _test_productkde_slogl_iter(variables, df_float, test_df_float)

    cpd = pbn.ProductKDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.ProductKDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.slogl(test_df), cpd2.slogl(test_df))), "Order of evidence changes slogl() result."

    cpd = pbn.ProductKDE(['d', 'a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = pbn.ProductKDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.slogl(test_df_float), cpd2.slogl(test_df_float), atol=0.0005)), "Order of evidence changes slogl() result."


def test_productkde_slogl_null():
    def _test_productkde_slogl_null_iter(variables, _df, _test_df):
        cpd = pbn.ProductKDE(variables)
        cpd.fit(_df)

        npdata = _df.loc[:, variables].to_numpy()
        
        factor = factor_product_kernel(npdata)
        final_scipy_kde = gaussian_kde(npdata.T, 
                                       bw_method=lambda gkde: factor * np.eye(npdata.shape[1], dtype=npdata.dtype))
        final_scipy_kde.cho_cov = np.linalg.cholesky(final_scipy_kde.covariance)
        final_scipy_kde.log_det = 2*np.log(np.diag(final_scipy_kde.cho_cov * np.sqrt(2*np.pi))).sum()

        test_npdata = _test_df.loc[:, variables].to_numpy()
        nan_rows = np.any(np.isnan(test_npdata), axis=1)

        if npdata.dtype == "float32":
            assert np.all(np.isclose(cpd.slogl(_test_df),
                                     np.sum(final_scipy_kde.logpdf(test_npdata[~nan_rows].T)),
                                     atol=5e-3*test_npdata.shape[0]))
        else:
            assert np.all(np.isclose(cpd.slogl(_test_df),
                                     np.sum(final_scipy_kde.logpdf(test_npdata[~nan_rows].T))))

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
        _test_productkde_slogl_null_iter(variables, df, df_null)
        _test_productkde_slogl_null_iter(variables, df_float, df_null_float)


    cpd = pbn.ProductKDE(['d', 'a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.ProductKDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.slogl(df_null), cpd2.slogl(df_null))), "Order of evidence changes slogl() result."

    cpd = pbn.ProductKDE(['d', 'a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = pbn.ProductKDE(['a', 'c', 'd', 'b'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.slogl(df_null_float), cpd2.slogl(df_null_float), atol=0.0005)), "Order of evidence changes slogl() result."
