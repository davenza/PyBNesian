import numpy as np
import pyarrow as pa
from pgm_dataset.factors.continuous import CKDE
from scipy.stats import gaussian_kde

import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)

def test_variable():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = CKDE(variable, evidence)
        assert cpd.variable == variable

def test_evidence():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = CKDE(variable, evidence)
        assert cpd.evidence == evidence

def test_kde_joint():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = CKDE(variable, evidence)
        cpd.fit(df)
            
        kde_joint = cpd.kde_joint

        kde_joint.bandwidth = np.eye(len(evidence) + 1)

        assert np.all(cpd.kde_joint.bandwidth == np.eye(len(evidence) + 1)), "kde_joint do not return a reference to the KDE joint, but a copy."

def test_kde_marg():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = CKDE(variable, evidence)
        cpd.fit(df)
            
        kde_marg = cpd.kde_marg

        kde_marg.bandwidth = np.eye(len(evidence))

        assert np.all(cpd.kde_marg.bandwidth == np.eye(len(evidence))), "kde_marg do not return a reference to the KDE joint, but a copy."

def test_fit():
    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        variables = [variable] + evidence
        for instances in [50, 1000, 10000]:            
            npdata = df.loc[:, variables].to_numpy()
            scipy_kde = gaussian_kde(npdata[:instances, :].T)

            cpd = CKDE(variable, evidence)
            cpd.fit(df.iloc[:instances])

            kde_joint = cpd.kde_joint
            kde_marg = cpd.kde_marg

            assert np.all(np.isclose(kde_joint.bandwidth, scipy_kde.covariance))
            assert np.all(np.isclose(kde_marg.bandwidth, scipy_kde.covariance[1:,1:]))
            assert cpd.n == instances

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

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        variables = [variable] + evidence
        for instances in [50, 1000, 10000]:            
            cpd = CKDE(variable, evidence)
            cpd.fit(df.iloc[:instances])

            npdata = df.loc[:, variables].to_numpy()
            npdata_instances = npdata[:instances,:]
            nan_rows = np.any(np.isnan(npdata_instances), axis=1)
            npdata_no_null = npdata_instances[~nan_rows,:]
            scipy_kde = gaussian_kde(npdata_no_null.T)

            kde_joint = cpd.kde_joint
            kde_marg = cpd.kde_marg

            assert np.all(np.isclose(kde_joint.bandwidth, scipy_kde.covariance))
            assert np.all(np.isclose(kde_marg.bandwidth, scipy_kde.covariance[1:,1:]))
            assert cpd.n == scipy_kde.n

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

    if evidence:
        return joint_kde.logpdf(test_data_joint[~nan_rows,:].T) - marg_kde.logpdf(test_data_marg[~nan_rows,:].T)
    else:
        return joint_kde.logpdf(test_data_joint[~nan_rows,:].T)

def test_logpdf():
    test_df = util_test.generate_normal_data(50)

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = CKDE(variable, evidence)
        cpd.fit(df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(df, variable, evidence)

        assert np.all(np.isclose(
                            cpd.logpdf(test_df), 
                            scipy_ckde_logpdf(test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)))

    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
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

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = CKDE(variable, evidence)
        cpd.fit(df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(df, variable, evidence)

        ll = cpd.logpdf(test_df)
        scipy_logpdf = scipy_ckde_logpdf(test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        assert np.all(np.isnan(ll) == np.isnan(scipy_logpdf))
        nan_indices = np.isnan(ll)
        assert np.all(np.isclose(ll[~nan_indices], scipy_logpdf[~nan_indices]))

    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)

    ll = cpd.logpdf(test_df)
    ll2 = cpd2.logpdf(test_df)
    assert np.all(np.isnan(ll) == np.isnan(ll2)), "Order of evidence changes the position of nan values."
    nan_indices = np.isnan(ll)
    assert np.all(np.isclose(ll[~nan_indices], ll2[~nan_indices])), "Order of evidence changes the position of nan values."


def test_slogpdf():
    test_df = util_test.generate_normal_data(50)

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = CKDE(variable, evidence)
        cpd.fit(df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(df, variable, evidence)
        scipy_logpdf = scipy_ckde_logpdf(test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        assert np.all(np.isclose(cpd.slogpdf(test_df), scipy_logpdf.sum()))

    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
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

    for variable, evidence in [('a', []), ('b', ['a']), ('c', ['a', 'b']), ('d', ['a', 'b', 'c'])]:
        cpd = CKDE(variable, evidence)
        cpd.fit(df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(df, variable, evidence)
        scipy_logpdf = scipy_ckde_logpdf(test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence)

        assert np.all(np.isclose(cpd.slogpdf(test_df), np.nansum(scipy_logpdf)))


    cpd = CKDE('d', ['a', 'b', 'c'])
    cpd.fit(df)
    cpd2 = CKDE('d', ['c', 'b', 'a'])
    cpd2.fit(df)


    assert np.all(np.isclose(cpd.slogpdf(test_df), cpd2.slogpdf(test_df))), "Order of evidence changes slogpdf() result."
