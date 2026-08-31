import numpy as np
import pandas as pd
import pyarrow as pa
import pybnesian as pbn
import pytest
from helpers.data import DATA_SIZE, generate_normal_data
from helpers.kde import (
    diagonal_kde_logpdf,
    diagonal_kernel_logpdf_matrix,
    normal_reference_bandwidth,
)
from scipy.special import logsumexp
from scipy.stats import norm

SMALL_SIZE = 10
TEST_SIZE = 50
df = generate_normal_data(DATA_SIZE, seed=0)
df_small = generate_normal_data(SMALL_SIZE, seed=0)
df_float = df.astype("float32")
df_small_float = df_small.astype("float32")


def test_variable():
    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        cpd = pbn.CKDE(variable, evidence)
        assert cpd.variable() == variable


def test_evidence():
    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        cpd = pbn.CKDE(variable, evidence)
        assert cpd.evidence() == evidence


def test_kde_data_type():
    k = pbn.CKDE("a", [])

    with pytest.raises(ValueError) as ex:
        k.data_type()
    assert "CKDE factor not fitted" in str(ex.value)

    k.fit(df)
    assert k.data_type() == pa.float64()
    k.fit(df_float)
    assert k.data_type() == pa.float32()


def test_ckde_kde_joint():
    def _test_ckde_kde_joint_iter(variable, evidence, _df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)
        kde_joint = cpd.kde_joint
        bandwidth = normal_reference_bandwidth(
            _df.loc[:, [variable] + evidence].dropna(), [variable] + evidence
        )
        kde_joint().bandwidth = np.diag(bandwidth)
        assert np.all(
            cpd.kde_joint().bandwidth == np.diag(bandwidth)
        ), "kde_joint do not return a reference to the KDE joint, but a copy."

    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        _test_ckde_kde_joint_iter(variable, evidence, df)
        _test_ckde_kde_joint_iter(variable, evidence, df_float)


def test_ckde_kde_marg():
    def _test_ckde_kde_marg_iter(variable, evidence, _df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)
        kde_marg = cpd.kde_marg

        if evidence:
            assert kde_marg().fitted()
            joint_bandwidth = normal_reference_bandwidth(
                _df.loc[:, [variable] + evidence].dropna(), [variable] + evidence
            )
            kde_marg().bandwidth = np.diag(joint_bandwidth[1:])
            assert np.all(
                cpd.kde_marg().bandwidth == np.diag(joint_bandwidth[1:])
            ), "kde_marg do not return a reference to the KDE joint, but a copy."
        else:
            # kde_marg contains garbage if there is no evidence
            pass

    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        _test_ckde_kde_marg_iter(variable, evidence, df)
        _test_ckde_kde_marg_iter(variable, evidence, df_float)


def test_ckde_fit():
    def _test_ckde_fit(variables, _df, instances):
        joint_bandwidth = normal_reference_bandwidth(_df.iloc[:instances], variables)

        cpd = pbn.CKDE(variable, evidence)
        assert not cpd.fitted()
        cpd.fit(_df.iloc[:instances])
        assert cpd.fitted()

        kde_joint = cpd.kde_joint
        assert np.all(np.isclose(kde_joint().bandwidth, np.diag(joint_bandwidth)))

        if evidence:
            kde_marg = cpd.kde_marg
            assert np.all(
                np.isclose(kde_marg().bandwidth, np.diag(joint_bandwidth[1:]))
            )

        assert cpd.num_instances() == instances

    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
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

        joint_data = _df.iloc[:instances].loc[:, variables].dropna()
        joint_bandwidth = normal_reference_bandwidth(joint_data, variables)

        kde_joint = cpd.kde_joint
        assert np.all(np.isclose(kde_joint().bandwidth, np.diag(joint_bandwidth)))

        if evidence:
            kde_marg = cpd.kde_marg
            assert np.all(
                np.isclose(kde_marg().bandwidth, np.diag(joint_bandwidth[1:]))
            )

        assert cpd.num_instances() == joint_data.shape[0]

    np.random.seed(0)
    a_null = np.random.randint(0, DATA_SIZE, size=100)
    b_null = np.random.randint(0, DATA_SIZE, size=100)
    c_null = np.random.randint(0, DATA_SIZE, size=100)
    d_null = np.random.randint(0, DATA_SIZE, size=100)

    df_null = df.copy()
    df_null.loc[df_null.index[a_null], "a"] = np.nan
    df_null.loc[df_null.index[b_null], "b"] = np.nan
    df_null.loc[df_null.index[c_null], "c"] = np.nan
    df_null.loc[df_null.index[d_null], "d"] = np.nan

    df_null_float = df_float.copy()
    df_null_float.loc[df_null_float.index[a_null], "a"] = np.nan
    df_null_float.loc[df_null_float.index[b_null], "b"] = np.nan
    df_null_float.loc[df_null_float.index[c_null], "c"] = np.nan
    df_null_float.loc[df_null_float.index[d_null], "d"] = np.nan

    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        variables = [variable] + evidence
        for instances in [50, 1000, 10000]:
            _test_ckde_fit_null(variable, evidence, variables, df, instances)
            _test_ckde_fit_null(variable, evidence, variables, df_float, instances)


def train_scipy_ckde(data, variable, evidence):
    variables = [variable] + evidence
    joint_data = data.loc[:, variables].dropna()
    joint_bandwidth = normal_reference_bandwidth(joint_data, variables)
    return joint_data, joint_bandwidth


def scipy_ckde_logpdf(test_data, joint_data, joint_bandwidth, variable, evidence):
    variables = [variable] + evidence
    test_data_joint = test_data.loc[:, variables].to_numpy()

    nan_rows = np.any(np.isnan(test_data_joint), axis=1)

    if np.all(test_data.dtypes == "float32"):
        result = np.full(test_data.shape[0], np.nan, dtype=np.float32)
    else:
        result = np.full(test_data.shape[0], np.nan, dtype=np.float64)

    if evidence:
        result[~nan_rows] = diagonal_kde_logpdf(
            test_data.loc[~nan_rows, variables], joint_data, variables, joint_bandwidth
        ) - diagonal_kde_logpdf(
            test_data.loc[~nan_rows, evidence],
            joint_data.loc[:, evidence],
            evidence,
            joint_bandwidth[1:],
        )
    else:
        result[~nan_rows] = diagonal_kde_logpdf(
            test_data.loc[~nan_rows, variables], joint_data, variables, joint_bandwidth
        )

    return result


def scipy_ckde_cdf(test_data, joint_data, joint_bandwidth, variable, evidence):
    variables = [variable] + evidence
    test_data_joint = test_data.loc[:, variables].to_numpy()

    nan_rows = np.any(np.isnan(test_data_joint), axis=1)

    if np.all(test_data.dtypes == "float32"):
        result = np.full(test_data.shape[0], np.nan, dtype=np.float32)
    else:
        result = np.full(test_data.shape[0], np.nan, dtype=np.float64)

    valid_rows = np.where(~nan_rows)[0]

    if evidence:
        log_weights = diagonal_kernel_logpdf_matrix(
            test_data.loc[valid_rows, evidence].to_numpy(),
            joint_data.loc[:, evidence].to_numpy(),
            joint_bandwidth[1:],
        )
        weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
        cdf = norm.cdf(
            test_data_joint[valid_rows, 0][:, None],
            joint_data.loc[:, variable].to_numpy()[None, :],
            np.sqrt(joint_bandwidth[0]),
        )
        result[valid_rows] = np.sum(weights * cdf, axis=1)
    else:
        cdf = norm.cdf(
            test_data_joint[valid_rows, 0][:, None],
            joint_data.loc[:, variable].to_numpy()[None, :],
            np.sqrt(joint_bandwidth[0]),
        )
        result[valid_rows] = np.mean(cdf, axis=1)

    return result


def test_ckde_logl():
    def _test_ckde_logl(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)
        scipy_kde_joint, scipy_kde_bandwidth = train_scipy_ckde(_df, variable, evidence)

        logl = cpd.logl(_test_df)
        scipy = scipy_ckde_logpdf(
            _test_df, scipy_kde_joint, scipy_kde_bandwidth, variable, evidence
        )

        if np.all(_df.dtypes == "float32"):
            assert np.all(np.isclose(logl, scipy, atol=0.0005))
        else:
            assert np.all(np.isclose(logl, scipy))

    test_df = generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype("float32")

    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        _test_ckde_logl(variable, evidence, df, test_df)
        _test_ckde_logl(variable, evidence, df_small, test_df)
        _test_ckde_logl(variable, evidence, df_float, test_df_float)
        _test_ckde_logl(variable, evidence, df_small_float, test_df_float)

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df)
    assert np.all(
        np.isclose(cpd.logl(test_df), cpd2.logl(test_df))
    ), "Order of evidence changes logl() result."

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df_float)
    assert np.all(
        np.isclose(cpd.logl(test_df_float), cpd2.logl(test_df_float), atol=0.0005)
    ), "Order of evidence changes logl() result."


def test_ckde_logl_null():
    def _test_ckde_logl_null(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_bandwidth = train_scipy_ckde(_df, variable, evidence)

        logl = cpd.logl(_test_df)
        scipy = scipy_ckde_logpdf(
            _test_df, scipy_kde_joint, scipy_kde_bandwidth, variable, evidence
        )

        if np.all(_test_df.dtypes == "float32"):
            assert np.all(np.isclose(logl, scipy, atol=0.0005, equal_nan=True))
        else:
            assert np.all(np.isclose(logl, scipy, equal_nan=True))

    test_df = generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype("float32")

    np.random.seed(0)
    a_null = np.random.randint(0, TEST_SIZE, size=10)
    b_null = np.random.randint(0, TEST_SIZE, size=10)
    c_null = np.random.randint(0, TEST_SIZE, size=10)
    d_null = np.random.randint(0, TEST_SIZE, size=10)

    df_null = test_df.copy()
    df_null.loc[df_null.index[a_null], "a"] = np.nan
    df_null.loc[df_null.index[b_null], "b"] = np.nan
    df_null.loc[df_null.index[c_null], "c"] = np.nan
    df_null.loc[df_null.index[d_null], "d"] = np.nan

    df_null_float = test_df_float.copy()
    df_null_float.loc[df_null_float.index[a_null], "a"] = np.nan
    df_null_float.loc[df_null_float.index[b_null], "b"] = np.nan
    df_null_float.loc[df_null_float.index[c_null], "c"] = np.nan
    df_null_float.loc[df_null_float.index[d_null], "d"] = np.nan

    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        _test_ckde_logl_null(variable, evidence, df, df_null)
        _test_ckde_logl_null(variable, evidence, df_small, df_null)
        _test_ckde_logl_null(variable, evidence, df_float, df_null_float)
        _test_ckde_logl_null(variable, evidence, df_small_float, df_null_float)

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df)

    ll = cpd.logl(df_null)
    ll2 = cpd2.logl(df_null)
    assert np.all(
        np.isclose(ll, ll2, equal_nan=True)
    ), "Order of evidence changes the position of nan values."

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df_float)

    ll = cpd.logl(df_null_float)
    ll2 = cpd2.logl(df_null_float)
    assert np.all(
        np.isclose(ll, ll2, equal_nan=True)
    ), "Order of evidence changes the position of nan values."


def test_ckde_slogl():
    def _test_ckde_slogl(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_bandwidth = train_scipy_ckde(_df, variable, evidence)
        scipy_logl = scipy_ckde_logpdf(
            _test_df, scipy_kde_joint, scipy_kde_bandwidth, variable, evidence
        )

        if np.all(_test_df.dtypes == "float32"):
            # Allow an error of 0.0005 for each training instance.
            assert np.isclose(
                cpd.slogl(_test_df), scipy_logl.sum(), atol=0.0005 * _df.shape[0]
            )
        else:
            assert np.isclose(cpd.slogl(_test_df), scipy_logl.sum())

    test_df = generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype("float32")

    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        _test_ckde_slogl(variable, evidence, df, test_df)
        _test_ckde_slogl(variable, evidence, df_small, test_df)
        _test_ckde_slogl(variable, evidence, df_float, test_df_float)
        _test_ckde_slogl(variable, evidence, df_small_float, test_df_float)

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df)
    assert np.all(
        np.isclose(cpd.slogl(test_df), cpd2.slogl(test_df))
    ), "Order of evidence changes slogl() result."

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df_float)
    assert np.all(
        np.isclose(cpd.slogl(test_df_float), cpd2.slogl(test_df_float))
    ), "Order of evidence changes slogl() result."


def test_ckde_slogl_null():
    def _test_ckde_slogl_null(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_bandwidth = train_scipy_ckde(_df, variable, evidence)
        scipy_logl = scipy_ckde_logpdf(
            _test_df, scipy_kde_joint, scipy_kde_bandwidth, variable, evidence
        )

        if np.all(_test_df.dtypes == "float32"):
            # Allow an error of 0.0005 for each training instance.
            assert np.isclose(
                cpd.slogl(_test_df), np.nansum(scipy_logl), atol=0.0005 * _df.shape[0]
            )
        else:
            assert np.isclose(cpd.slogl(_test_df), np.nansum(scipy_logl))

    test_df = generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype("float32")

    np.random.seed(0)
    a_null = np.random.randint(0, TEST_SIZE, size=10)
    b_null = np.random.randint(0, TEST_SIZE, size=10)
    c_null = np.random.randint(0, TEST_SIZE, size=10)
    d_null = np.random.randint(0, TEST_SIZE, size=10)

    df_null = test_df.copy()
    df_null.loc[df_null.index[a_null], "a"] = np.nan
    df_null.loc[df_null.index[b_null], "b"] = np.nan
    df_null.loc[df_null.index[c_null], "c"] = np.nan
    df_null.loc[df_null.index[d_null], "d"] = np.nan

    df_null_float = test_df_float.copy()
    df_null_float.loc[df_null_float.index[a_null], "a"] = np.nan
    df_null_float.loc[df_null_float.index[b_null], "b"] = np.nan
    df_null_float.loc[df_null_float.index[c_null], "c"] = np.nan
    df_null_float.loc[df_null_float.index[d_null], "d"] = np.nan

    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        _test_ckde_slogl_null(variable, evidence, df, df_null)
        _test_ckde_slogl_null(variable, evidence, df_small, df_null)
        _test_ckde_slogl_null(variable, evidence, df_float, df_null_float)
        _test_ckde_slogl_null(variable, evidence, df_small_float, df_null_float)

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df)
    assert np.all(
        np.isclose(cpd.slogl(df_null), cpd2.slogl(df_null))
    ), "Order of evidence changes slogl() result."

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df_float)
    assert np.all(
        np.isclose(cpd.slogl(df_null_float), cpd2.slogl(df_null_float))
    ), "Order of evidence changes slogl() result."


def test_ckde_cdf():
    def _test_ckde_cdf(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)
        scipy_kde_joint, scipy_kde_bandwidth = train_scipy_ckde(_df, variable, evidence)

        cdf = cpd.cdf(_test_df)
        scipy = scipy_ckde_cdf(
            _test_df, scipy_kde_joint, scipy_kde_bandwidth, variable, evidence
        )

        if np.all(_df.dtypes == "float32"):
            assert np.all(np.isclose(cdf, scipy, atol=0.0005))
        else:
            assert np.all(np.isclose(cdf, scipy))

    test_df = generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype("float32")

    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        _test_ckde_cdf(variable, evidence, df, test_df)
        _test_ckde_cdf(variable, evidence, df_small, test_df)
        _test_ckde_cdf(variable, evidence, df_float, test_df_float)
        _test_ckde_cdf(variable, evidence, df_small_float, test_df_float)

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df)
    assert np.all(
        np.isclose(cpd.cdf(test_df), cpd2.cdf(test_df))
    ), "Order of evidence changes logl() result."

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df_float)
    assert np.all(
        np.isclose(cpd.cdf(test_df_float), cpd2.cdf(test_df_float), atol=0.0005)
    ), "Order of evidence changes logl() result."


def test_ckde_cdf_null():
    def _test_ckde_cdf_null(variable, evidence, _df, _test_df):
        cpd = pbn.CKDE(variable, evidence)
        cpd.fit(_df)

        scipy_kde_joint, scipy_kde_marg = train_scipy_ckde(_df, variable, evidence)

        cdf = cpd.cdf(_test_df)
        scipy = scipy_ckde_cdf(
            _test_df, scipy_kde_joint, scipy_kde_marg, variable, evidence
        )

        if np.all(_df.dtypes == "float32"):
            assert np.all(np.isclose(cdf, scipy, atol=0.0005, equal_nan=True))
        else:
            assert np.all(np.isclose(cdf, scipy, equal_nan=True))

    test_df = generate_normal_data(TEST_SIZE, seed=1)
    test_df_float = test_df.astype("float32")

    np.random.seed(0)
    a_null = np.random.randint(0, TEST_SIZE, size=10)
    b_null = np.random.randint(0, TEST_SIZE, size=10)
    c_null = np.random.randint(0, TEST_SIZE, size=10)
    d_null = np.random.randint(0, TEST_SIZE, size=10)

    df_null = test_df.copy()
    df_null.loc[df_null.index[a_null], "a"] = np.nan
    df_null.loc[df_null.index[b_null], "b"] = np.nan
    df_null.loc[df_null.index[c_null], "c"] = np.nan
    df_null.loc[df_null.index[d_null], "d"] = np.nan

    df_null_float = test_df_float.copy()
    df_null_float.loc[df_null_float.index[a_null], "a"] = np.nan
    df_null_float.loc[df_null_float.index[b_null], "b"] = np.nan
    df_null_float.loc[df_null_float.index[c_null], "c"] = np.nan
    df_null_float.loc[df_null_float.index[d_null], "d"] = np.nan

    for variable, evidence in [
        ("a", []),
        ("b", ["a"]),
        ("c", ["a", "b"]),
        ("d", ["a", "b", "c"]),
    ]:
        _test_ckde_cdf_null(variable, evidence, df, df_null)
        _test_ckde_cdf_null(variable, evidence, df_small, df_null)
        _test_ckde_cdf_null(variable, evidence, df_float, df_null_float)
        _test_ckde_cdf_null(variable, evidence, df_small_float, df_null_float)

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df)
    assert np.all(
        np.isclose(cpd.cdf(df_null), cpd2.cdf(df_null), equal_nan=True)
    ), "Order of evidence changes cdf() result."

    cpd = pbn.CKDE("d", ["a", "b", "c"])
    cpd.fit(df_float)
    cpd2 = pbn.CKDE("d", ["c", "b", "a"])
    cpd2.fit(df_float)
    assert np.all(
        np.isclose(
            cpd.cdf(df_null_float), cpd2.cdf(df_null_float), atol=0.0005, equal_nan=True
        )
    ), "Order of evidence changes cdf() result."


def test_ckde_sample():
    SAMPLE_SIZE = 1000

    cpd = pbn.CKDE("a", [])
    cpd.fit(df)

    sampled = cpd.sample(SAMPLE_SIZE, None, 0)

    assert sampled.type == pa.float64()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE

    cpd = pbn.CKDE("b", ["a"])
    cpd.fit(df)

    sampling_df = pd.DataFrame({"a": np.full((SAMPLE_SIZE,), 3.0)})
    sampled = cpd.sample(SAMPLE_SIZE, sampling_df, 0)

    assert sampled.type == pa.float64()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE

    cpd = pbn.CKDE("c", ["a", "b"])
    cpd.fit(df)

    sampling_df = pd.DataFrame(
        {"a": np.full((SAMPLE_SIZE,), 3.0), "b": np.full((SAMPLE_SIZE,), 7.45)}
    )
    sampled = cpd.sample(SAMPLE_SIZE, sampling_df, 0)

    assert sampled.type == pa.float64()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE

    cpd = pbn.CKDE("a", [])
    cpd.fit(df_float)

    sampled = cpd.sample(SAMPLE_SIZE, None, 0)

    assert sampled.type == pa.float32()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE

    cpd = pbn.CKDE("b", ["a"])
    cpd.fit(df_float)

    sampling_df = pd.DataFrame({"a": np.full((SAMPLE_SIZE,), 3.0, dtype=np.float32)})
    sampled = cpd.sample(SAMPLE_SIZE, sampling_df, 0)

    assert sampled.type == pa.float32()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE

    cpd = pbn.CKDE("c", ["a", "b"])
    cpd.fit(df_float)

    sampling_df = pd.DataFrame(
        {
            "a": np.full((SAMPLE_SIZE,), 3.0, dtype=np.float32),
            "b": np.full((SAMPLE_SIZE,), 7.45, dtype=np.float32),
        }
    )
    sampled = cpd.sample(SAMPLE_SIZE, sampling_df, 0)

    assert sampled.type == pa.float32()
    assert int(sampled.nbytes / (sampled.type.bit_width / 8)) == SAMPLE_SIZE
