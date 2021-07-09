import pytest
import numpy as np
import pyarrow as pa
import pandas as pd
import pybnesian as pbn
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.special import logsumexp

import util_test

SIZE = 10000
SMALL_SIZE = 10
TEST_SIZE = 50
df = util_test.generate_normal_data(SIZE, seed=0)
df_small = util_test.generate_normal_data(SMALL_SIZE, seed=0)
df_float = df.astype('float32')
df_small_float = df_small.astype('float32')

def test_variable():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = pbn.CKDE(variable, evidence)
        assert cpd.variable() == variable

def test_evidence():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = pbn.CKDE(variable, evidence)
        assert cpd.evidence() == evidence

def test_kde_data_type():
    k = pbn.CKDE("a", [])

    with pytest.raises(ValueError) as ex:
        k.data_type()
    "CKDE factor not fitted" in str(ex.value)

    k.fit(df)
    assert k.data_type() == pa.float64()
    k.fit(df_float)
    assert k.data_type() == pa.float32()

def test_ckde_kde_joint():
    def _test_ckde_kde_joint_iter(variable, evidence, _df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)
        kde_joint = cpd.kde_joint
        kde_joint().bandwidth = np.eye(len(evidence) + 1)
        assert np.all(cpd.kde_joint().bandwidth == np.eye(len(evidence) + 1)), "kde_joint do not return a reference to the KDE joint, but a copy."

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_kde_joint_iter(variable, evidence, df)
        _test_ckde_kde_joint_iter(variable, evidence, df_float)

def test_ckde_kde_marg():
    def _test_ckde_kde_marg_iter(variable, evidence, _df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)
        kde_marg = cpd.kde_marg

        if evidence:
            assert kde_marg().fitted()
            kde_marg().bandwidth = np.eye(len(evidence))
            assert np.all(cpd.kde_marg().bandwidth == np.eye(len(evidence))), "kde_marg do not return a reference to the KDE joint, but a copy."
        else:
            # kde_marg contains garbage if there is no evidence
            pass

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_kde_marg_iter(variable, evidence, df)
        _test_ckde_kde_marg_iter(variable, evidence, df_float)

def test_ckde_fit():
    def _test_ckde_fit(variables, _df, instances):
        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata[:instances, :].T,
                        bw_method=lambda s : np.power(4 / (s.d + 2), 1 / (s.d + 4)) * s.scotts_factor())

        cpd = pbn.CKDE(variable, evidence)
        assert not cpd.fitted()
        cpd.fit(_df.iloc[:instances])
        assert cpd.fitted()

        kde_joint = cpd.kde_joint
        assert np.all(np.isclose(kde_joint().bandwidth, scipy_kde.covariance))
        
        if evidence:
            kde_marg = cpd.kde_marg
            assert np.all(np.isclose(kde_marg().bandwidth, scipy_kde.covariance[1:,1:]))
        
        assert cpd.num_instances() == instances

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        variables = [variable] + evidence
        for instances in [50, 1000, 10000]:            
            _test_ckde_fit(variables, df, instances)
            _test_ckde_fit(variables, df_float, instances)

def test_ckde_fit_null():
    def _test_ckde_fit_null(variable, evidence, variables, _df, instances):
        cpd = pbn.CKDE(variable, evidence)
        assert not cpd.fitted()
        cpd.fit(_df.iloc[:instances])
        assert cpd.fitted()

        npdata = _df.loc[:, variables].to_numpy()
        npdata_instances = npdata[:instances,:]
        nan_rows = np.any(np.isnan(npdata_instances), axis=1)
        npdata_no_null = npdata_instances[~nan_rows,:]
        scipy_kde = gaussian_kde(npdata_no_null.T,
                        bw_method=lambda s : np.power(4 / (s.d + 2), 1 / (s.d + 4)) * s.scotts_factor())

        kde_joint = cpd.kde_joint
        assert np.all(np.isclose(kde_joint().bandwidth, scipy_kde.covariance))
        
        if evidence:
            kde_marg = cpd.kde_marg
            assert np.all(np.isclose(kde_marg().bandwidth, scipy_kde.covariance[1:,1:]))

        assert cpd.num_instances() == scipy_kde.n

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

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        variables = [variable] + evidence
        for instances in [50, 1000, 10000]:            
            _test_ckde_fit_null(variable, evidence, variables, df, instances)
            _test_ckde_fit_null(variable, evidence, variables, df_float, instances)

def train_scipy_ckde(data, variable, evidence):
    variables = [variable] + evidence
    npdata_joint = data.loc[:, variables].to_numpy()
    npdata_marg = data.loc[:, evidence].to_numpy()

    nan_rows = np.any(np.isnan(npdata_joint), axis=1)

    scipy_kde_joint = gaussian_kde(npdata_joint[~nan_rows,:].T,
                        bw_method=lambda s : np.power(4 / (s.d + 2), 1 / (s.d + 4)) * s.scotts_factor())
    if evidence:
        scipy_kde_marg = gaussian_kde(npdata_marg[~nan_rows,:].T, bw_method=scipy_kde_joint.covariance_factor())
    else:
        scipy_kde_marg = None

    return scipy_kde_joint, scipy_kde_marg

def scipy_ckde_logpdf(test_data, joint_kde, marg_kde, variable, evidence):
    variables = [variable] + evidence
    test_data_joint = test_data.loc[:, variables].to_numpy()
    test_data_marg = test_data.loc[:, evidence].to_numpy()

    nan_rows = np.any(np.isnan(test_data_joint), axis=1)

    if np.all(test_data.dtypes == "float32"):
        result = np.full(test_data.shape[0], np.nan, dtype=np.float32)
    else:
        result = np.full(test_data.shape[0], np.nan, dtype=np.float64)

    if evidence:
        result[~nan_rows] = joint_kde.logpdf(test_data_joint[~nan_rows,:].T) - marg_kde.logpdf(test_data_marg[~nan_rows,:].T)
    else:
        result[~nan_rows] = joint_kde.logpdf(test_data_joint[~nan_rows,:].T)

    return result

def scipy_ckde_cdf(test_data, joint_kde, marg_kde, variable, evidence):
    variables = [variable] + evidence
    test_data_joint = test_data.loc[:, variables].to_numpy()
    test_data_marg = test_data.loc[:, evidence].to_numpy()

    nan_rows = np.any(np.isnan(test_data_joint), axis=1)

    if np.all(test_data.dtypes == "float32"):
        result = np.full(test_data.shape[0], np.nan, dtype=np.float32)
    else:
        result = np.full(test_data.shape[0], np.nan, dtype=np.float64)

    total_w = np.empty((joint_kde.n, test_data_joint.shape[0]))
    conditional_mean = np.empty((joint_kde.n, test_data_joint.shape[0]))
    total_cdf = np.empty((joint_kde.n, test_data_joint.shape[0]))

    if evidence:
        bandwidth = joint_kde.covariance
        cond_var = bandwidth[0,0] - bandwidth[0, 1:].dot(np.linalg.inv(bandwidth[1:, 1:])).dot(bandwidth[1:, 0])
        for test_index in np.where(~np.any(np.isnan(test_data_joint), axis=1))[0]:
            w = mvn.logpdf(marg_kde.dataset.T, mean=test_data_marg[test_index,:], cov=marg_kde.covariance)
            w = np.exp(w)
            total_w[:, test_index] = w

            evidence_diff = test_data_marg[test_index,:] - joint_kde.dataset[1:,:].T
            cond_mean = joint_kde.dataset[0,:] + bandwidth[0,1:].dot(np.linalg.inv(bandwidth[1:,1:])).dot(evidence_diff.T)

            conditional_mean[:, test_index] = cond_mean
            total_cdf[:, test_index] = norm.cdf(test_data_joint[test_index,0], cond_mean, np.sqrt(cond_var))

            result[test_index] = np.dot(w, norm.cdf(test_data_joint[test_index,0], cond_mean, np.sqrt(cond_var)))

        result /= np.sum(total_w, axis=0)

    else:
        cdf = norm.cdf(test_data_joint[~nan_rows], joint_kde.dataset, np.sqrt(joint_kde.covariance[0,0]))
        result[~nan_rows] = np.sum((1 / joint_kde.n) * cdf, axis=1)

    return result

def test_ckde_logl():
    def _test_ckde_logl(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)
        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)

        logl = cpd.logl(_test_df)
        scipy = scipy_ckde_logpdf(_test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        if np.all(_df.dtypes == 'float32'):
            assert np.all(np.isclose(logl, scipy, atol=0.0005))
        else:
            assert np.all(np.isclose(logl, scipy))
    
    test_df = util_test.generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype('float32')

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_logl(variable, evidence, df, test_df)
        _test_ckde_logl(variable, evidence, df_small, test_df)
        _test_ckde_logl(variable, evidence, df_float, test_df_float)
        _test_ckde_logl(variable, evidence, df_small_float, test_df_float)

    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.logl(test_df), cpd2.logl(test_df))), "Order of evidence changes logl() result."

    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.logl(test_df_float), cpd2.logl(test_df_float), atol=0.0005)), "Order of evidence changes logl() result."

def test_ckde_logl_null():
    def _test_ckde_logl_null(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)

        logl = cpd.logl(_test_df)
        scipy = scipy_ckde_logpdf(_test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        if np.all(_test_df.dtypes == "float32"):
            assert np.all(np.isclose(logl, scipy, atol=0.0005, equal_nan=True))
        else:
            assert np.all(np.isclose(logl, scipy, equal_nan=True))

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

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_logl_null(variable, evidence, df, df_null)
        _test_ckde_logl_null(variable, evidence, df_small, df_null)
        _test_ckde_logl_null(variable, evidence, df_float, df_null_float)
        _test_ckde_logl_null(variable, evidence, df_small_float, df_null_float)

    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)

    ll = cpd.logl(df_null)
    ll2 = cpd2.logl(df_null)
    assert np.all(np.isclose(ll, ll2, equal_nan=True)), "Order of evidence changes the position of nan values."

    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df_float)

    ll = cpd.logl(df_null_float)
    ll2 = cpd2.logl(df_null_float)
    assert np.all(np.isclose(ll, ll2, equal_nan=True)), "Order of evidence changes the position of nan values."

def test_ckde_slogl():
    def _test_ckde_slogl(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)
        scipy_logl = scipy_ckde_logpdf(_test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        if np.all(_test_df.dtypes == "float32"):
            # Allow an error of 0.0005 for each training instance.
            assert np.isclose(cpd.slogl(_test_df), scipy_logl.sum(), atol=0.0005*_df.shape[0])
        else:
            assert np.isclose(cpd.slogl(_test_df), scipy_logl.sum())

    test_df = util_test.generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype('float32')

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_slogl(variable, evidence, df, test_df)
        _test_ckde_slogl(variable, evidence, df_small, test_df)
        _test_ckde_slogl(variable, evidence, df_float, test_df_float)
        _test_ckde_slogl(variable, evidence, df_small_float, test_df_float)

    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.slogl(test_df), cpd2.slogl(test_df))), "Order of evidence changes slogl() result."

    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.slogl(test_df_float), cpd2.slogl(test_df_float))), "Order of evidence changes slogl() result."

def test_ckde_slogl_null():
    def _test_ckde_slogl_null(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)
        scipy_logl = scipy_ckde_logpdf(_test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        if np.all(_test_df.dtypes == "float32"):
            # Allow an error of 0.0005 for each training instance.
            assert np.isclose(cpd.slogl(_test_df), np.nansum(scipy_logl), atol=0.0005*_df.shape[0])
        else:
            assert np.isclose(cpd.slogl(_test_df), np.nansum(scipy_logl))


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

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_slogl_null(variable, evidence, df, df_null)
        _test_ckde_slogl_null(variable, evidence, df_small, df_null)
        _test_ckde_slogl_null(variable, evidence, df_float, df_null_float)
        _test_ckde_slogl_null(variable, evidence, df_small_float, df_null_float)


    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.slogl(df_null), cpd2.slogl(df_null))), "Order of evidence changes slogl() result."

    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.slogl(df_null_float), cpd2.slogl(df_null_float))), "Order of evidence changes slogl() result."

def test_ckde_cdf():
    def _test_ckde_cdf(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)
        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)

        cdf = cpd.cdf(_test_df)
        scipy = scipy_ckde_cdf(_test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        if np.all(_df.dtypes == 'float32'):
            assert np.all(np.isclose(cdf, scipy, atol=0.0005))
        else:
            assert np.all(np.isclose(cdf, scipy))
    
    test_df = util_test.generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype('float32')

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_cdf(variable, evidence, df, test_df)
        _test_ckde_cdf(variable, evidence, df_small, test_df)
        _test_ckde_cdf(variable, evidence, df_float, test_df_float)
        _test_ckde_cdf(variable, evidence, df_small_float, test_df_float)

    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.cdf(test_df), cpd2.cdf(test_df))), "Order of evidence changes logl() result."

    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.cdf(test_df_float), cpd2.cdf(test_df_float), atol=0.0005)), "Order of evidence changes logl() result."

def test_ckde_cdf_null():
    def _test_ckde_cdf_null(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)

        cdf = cpd.cdf(_test_df)
        scipy = scipy_ckde_cdf(_test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        if np.all(_df.dtypes == 'float32'):
            assert np.all(np.isclose(cdf, scipy, atol=0.0005, equal_nan=True))
        else:
            assert np.all(np.isclose(cdf, scipy, equal_nan=True))


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

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_cdf_null(variable, evidence, df, df_null)
        _test_ckde_cdf_null(variable, evidence, df_small, df_null)
        _test_ckde_cdf_null(variable, evidence, df_float, df_null_float)
        _test_ckde_cdf_null(variable, evidence, df_small_float, df_null_float)


    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.cdf(df_null), cpd2.cdf(df_null), equal_nan=True)), "Order of evidence changes cdf() result."

    cpd = pbn.CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.cdf(df_null_float), 
                             cpd2.cdf(df_null_float), 
                             atol=0.0005, equal_nan=True)), "Order of evidence changes cdf() result."

def test_ckde_sample():
    SAMPLE_SIZE = 1000

    cpd = pbn.CKDE('a', [])
    cpd.fit(df)
    
    sampled = cpd.sample(SAMPLE_SIZE, None, 0)

    assert sampled.type == pa.float64()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE
        
    cpd = pbn.CKDE('b', ['a'])
    cpd.fit(df)

    sampling_df = pd.DataFrame({'a': np.full((SAMPLE_SIZE,), 3.0)})
    sampled = cpd.sample(SAMPLE_SIZE, sampling_df, 0)

    assert sampled.type == pa.float64()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE
    
    cpd = pbn.CKDE('c', ['a', 'b'])
    cpd.fit(df)

    sampling_df = pd.DataFrame({'a': np.full((SAMPLE_SIZE,), 3.0),
                                'b': np.full((SAMPLE_SIZE,), 7.45)})
    sampled = cpd.sample(SAMPLE_SIZE, sampling_df, 0)

    assert sampled.type == pa.float64()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE

    cpd = pbn.CKDE('a', [])
    cpd.fit(df_float)
    
    sampled = cpd.sample(SAMPLE_SIZE, None, 0)

    assert sampled.type == pa.float32()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE
        
    cpd = pbn.CKDE('b', ['a'])
    cpd.fit(df_float)

    sampling_df = pd.DataFrame({'a': np.full((SAMPLE_SIZE,), 3.0, dtype=np.float32)})
    sampled = cpd.sample(SAMPLE_SIZE, sampling_df, 0)

    assert sampled.type == pa.float32()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE
    
    cpd = pbn.CKDE('c', ['a', 'b'])
    cpd.fit(df_float)

    sampling_df = pd.DataFrame({'a': np.full((SAMPLE_SIZE,), 3.0, dtype=np.float32),
                                'b': np.full((SAMPLE_SIZE,), 7.45, dtype=np.float32)})
    sampled = cpd.sample(SAMPLE_SIZE, sampling_df, 0)

    assert sampled.type == pa.float32()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE