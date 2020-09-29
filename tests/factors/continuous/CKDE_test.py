import numpy as np
import pyarrow as pa
from pgm_dataset.factors.continuous import CKDE
from scipy.stats import gaussian_kde

import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE, seed=0)
df_float = df.astype('float32')

def test_variable():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = CKDE(variable, evidence)
        assert cpd.variable == variable

def test_evidence():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = CKDE(variable, evidence)
        assert cpd.evidence == evidence

def test_ckde_kde_joint():
    def _test_ckde_kde_joint_iter(variable, evidence, _df):
        cpd = CKDE(variable, evidence)
        cpd.fit(_df)
        kde_joint = cpd.kde_joint
        kde_joint.bandwidth = np.eye(len(evidence) + 1)
        assert np.all(cpd.kde_joint.bandwidth == np.eye(len(evidence) + 1)), "kde_joint do not return a reference to the KDE joint, but a copy."

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_kde_joint_iter(variable, evidence, df)
        _test_ckde_kde_joint_iter(variable, evidence, df_float)

def test_ckde_kde_marg():
    def _test_ckde_kde_marg_iter(variable, evidence, _df):
        cpd = CKDE(variable, evidence)
        cpd.fit(_df)
        kde_marg = cpd.kde_marg
        kde_marg.bandwidth = np.eye(len(evidence))
        assert np.all(cpd.kde_marg.bandwidth == np.eye(len(evidence))), "kde_marg do not return a reference to the KDE joint, but a copy."

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_kde_marg_iter(variable, evidence, df)
        _test_ckde_kde_marg_iter(variable, evidence, df_float)

def test_ckde_fit():
    def _test_ckde_fit(variables, _df, instances):
        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(npdata[:instances, :].T)

        cpd = CKDE(variable, evidence)
        assert not cpd.fitted
        cpd.fit(_df.iloc[:instances])
        assert cpd.fitted

        kde_joint = cpd.kde_joint
        kde_marg = cpd.kde_marg

        assert np.all(np.isclose(kde_joint.bandwidth, scipy_kde.covariance))
        assert np.all(np.isclose(kde_marg.bandwidth, scipy_kde.covariance[1:,1:]))
        assert cpd.n == instances

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        variables = [variable] + evidence
        for instances in [50, 1000, 10000]:            
            _test_ckde_fit(variables, df, instances)
            _test_ckde_fit(variables, df_float, instances)

def test_ckde_fit_null():
    def _test_ckde_fit_null(variable, evidence, variables, _df, instances):
        cpd = CKDE(variable, evidence)
        assert not cpd.fitted
        cpd.fit(_df.iloc[:instances])
        assert cpd.fitted

        npdata = _df.loc[:, variables].to_numpy()
        npdata_instances = npdata[:instances,:]
        nan_rows = np.any(np.isnan(npdata_instances), axis=1)
        npdata_no_null = npdata_instances[~nan_rows,:]
        scipy_kde = gaussian_kde(npdata_no_null.T)

        kde_joint = cpd.kde_joint
        kde_marg = cpd.kde_marg

        assert np.all(np.isclose(kde_joint.bandwidth, scipy_kde.covariance))
        assert np.all(np.isclose(kde_marg.bandwidth, scipy_kde.covariance[1:,1:]))
        assert cpd.n == scipy_kde.n

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

    scipy_kde_joint = gaussian_kde(npdata_joint[~nan_rows,:].T)
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

def test_ckde_logl():
    def _test_ckde_logl(variable, evidence, _df, _test_df):
        cpd = CKDE(variable, evidence)
        cpd.fit(_df)
        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)

        logl = cpd.logl(_test_df)
        scipy = scipy_ckde_logpdf(_test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        if np.all(_df.dtypes == 'float32'):
            assert np.all(np.isclose(logl, scipy, atol=0.0005))
        else:
            assert np.all(np.isclose(logl, scipy))
    
    test_df = util_test.generate_normal_data(50, seed=1)
    test_df_float = test_df.astype('float32')

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_logl(variable, evidence, df, test_df)
        _test_ckde_logl(variable, evidence, df_float, test_df_float)

    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.logl(test_df), cpd2.logl(test_df))), "Order of evidence changes logl() result."

    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.logl(test_df_float), cpd2.logl(test_df_float))), "Order of evidence changes logl() result."

def test_ckde_logl_null():
    def _test_ckde_logl_null(variable, evidence, _df, _test_df):
        cpd = CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)

        logl = cpd.logl(_test_df)
        scipy = scipy_ckde_logpdf(_test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        if np.all(_test_df.dtypes == "float32"):
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

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_logl_null(variable, evidence, df, df_null)
        _test_ckde_logl_null(variable, evidence, df_float, df_null_float)

    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)

    ll = cpd.logl(df_null)
    ll2 = cpd2.logl(df_null)
    assert np.all(np.isclose(ll, ll2, equal_nan=True)), "Order of evidence changes the position of nan values."

    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df_float)

    ll = cpd.logl(df_null_float)
    ll2 = cpd2.logl(df_null_float)
    assert np.all(np.isclose(ll, ll2, equal_nan=True)), "Order of evidence changes the position of nan values."


def test_ckde_slogl():
    def _test_ckde_slogl(variable, evidence, _df, _test_df):
        cpd = CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)
        scipy_logl = scipy_ckde_logpdf(_test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        assert np.all(np.isclose(cpd.slogl(_test_df), scipy_logl.sum()))

    test_df = util_test.generate_normal_data(50, seed=1)
    test_df_float = test_df.astype('float32')

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_slogl(variable, evidence, df, test_df)
        _test_ckde_slogl(variable, evidence, df_float, test_df_float)

    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.slogl(test_df), cpd2.slogl(test_df))), "Order of evidence changes slogl() result."

    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.slogl(test_df_float), cpd2.slogl(test_df_float))), "Order of evidence changes slogl() result."

def test_ckde_slogl_null():
    def _test_ckde_slogl_null(variable, evidence, _df, _test_df):
        cpd = CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)
        scipy_logl = scipy_ckde_logpdf(_test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        assert np.all(np.isclose(cpd.slogl(_test_df), np.nansum(scipy_logl)))

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

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        _test_ckde_slogl_null(variable, evidence, df, df_null)
        _test_ckde_slogl_null(variable, evidence, df_float, df_null_float)


    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)
    assert np.all(np.isclose(cpd.slogl(df_null), cpd2.slogl(df_null))), "Order of evidence changes slogl() result."

    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df_float)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df_float)
    assert np.all(np.isclose(cpd.slogl(df_null_float), cpd2.slogl(df_null_float))), "Order of evidence changes slogl() result."

def test_sample():
    SAMPLE_SIZE = 1000000

    small_train = util_test.generate_normal_data(100, seed=2)
    validation_df = util_test.generate_normal_data(SAMPLE_SIZE, seed=3)

    print("Training data:")
    print(small_train)
    print("Validation data:")
    print(validation_df)

    cpd = CKDE('a', ['b'])
    cpd.fit(small_train)
    print()
    sampled = cpd.sample(SAMPLE_SIZE, validation_df, 0)

    assert sampled is not None