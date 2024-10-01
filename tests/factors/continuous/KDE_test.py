import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import gaussian_kde

import pybnesian as pbn
from pybnesian import BandwidthSelector
from util_test import generate_normal_data

SIZE = 500
df = generate_normal_data(SIZE, seed=0)
df_float = df.astype("float32")


def test_check_type():
    cpd = pbn.KDE(["a"])
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
    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        cpd = pbn.KDE(variables)
        assert cpd.variables() == variables


def test_kde_bandwidth():
    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        for instances in [50, 1000, 10000]:
            npdata = df.loc[:, variables].to_numpy()
            # Test normal reference rule
            scipy_kde = gaussian_kde(
                npdata[:instances, :].T,
                bw_method=lambda s: np.power(4 / (s.d + 2), 1 / (s.d + 4))
                * s.scotts_factor(),
            )

            cpd = pbn.KDE(variables)
            cpd.fit(df.iloc[:instances])
            assert np.all(
                np.isclose(cpd.bandwidth, scipy_kde.covariance)
            ), "Wrong bandwidth computed with normal reference rule."

            cpd.fit(df_float.iloc[:instances])
            assert np.all(
                np.isclose(cpd.bandwidth, scipy_kde.covariance)
            ), "Wrong bandwidth computed with normal reference rule."

            scipy_kde = gaussian_kde(npdata[:instances, :].T)

            cpd = pbn.KDE(variables, pbn.ScottsBandwidth())
            cpd.fit(df.iloc[:instances])
            assert np.all(
                np.isclose(cpd.bandwidth, scipy_kde.covariance)
            ), "Wrong bandwidth computed with Scott's rule."

            cpd.fit(df_float.iloc[:instances])
            assert np.all(
                np.isclose(cpd.bandwidth, scipy_kde.covariance)
            ), "Wrong bandwidth computed with Scott's rule."

    cpd = pbn.KDE(["a"])
    cpd.fit(df)
    cpd.bandwidth = [[1]]
    assert cpd.bandwidth == np.asarray([[1]]), "Could not change bandwidth."

    cpd.fit(df_float)
    cpd.bandwidth = [[1]]
    assert cpd.bandwidth == np.asarray([[1]]), "Could not change bandwidth."


class UnitaryBandwidth(BandwidthSelector):
    def __init__(self):
        BandwidthSelector.__init__(self)

    def bandwidth(self, df, variables):
        return np.eye(len(variables))


def test_kde_new_bandwidth():
    kde = pbn.KDE(["a"], UnitaryBandwidth())
    kde.fit(df)
    assert kde.bandwidth == np.eye(1)

    kde.fit(df_float)
    assert kde.bandwidth == np.eye(1)

    kde = pbn.KDE(["a", "b", "c", "d"], UnitaryBandwidth())
    kde.fit(df)
    assert np.all(kde.bandwidth == np.eye(4))

    kde.fit(df_float)
    assert np.all(kde.bandwidth == np.eye(4))


def test_kde_data_type():
    k = pbn.KDE(["a"])

    with pytest.raises(ValueError) as ex:
        k.data_type()
    assert "KDE factor not fitted" in str(ex.value)

    k.fit(df)
    assert k.data_type() == pa.float64()
    k.fit(df_float)
    assert k.data_type() == pa.float32()


def test_kde_fit():
    def _test_kde_fit_iter(variables, _df, instances):
        cpd = pbn.KDE(variables)
        assert not cpd.fitted()
        cpd.fit(_df.iloc[:instances, :])
        assert cpd.fitted()

        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(
            npdata[:instances, :].T,
            bw_method=lambda s: np.power(4 / (s.d + 2), 1 / (s.d + 4))
            * s.scotts_factor(),
        )

        assert scipy_kde.n == cpd.num_instances(), "Wrong number of training instances."
        assert scipy_kde.d == cpd.num_variables(), "Wrong number of training variables."

    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        for instances in [50, 150, 500]:
            _test_kde_fit_iter(variables, df, instances)
            _test_kde_fit_iter(variables, df_float, instances)


def test_kde_fit_null():
    def _test_kde_fit_null_iter(variables, _df, instances):
        cpd = pbn.KDE(variables)
        assert not cpd.fitted()
        cpd.fit(_df.iloc[:instances, :])
        assert cpd.fitted()

        npdata = _df.loc[:, variables].to_numpy()
        npdata_instances = npdata[:instances, :]

        nan_rows = np.any(np.isnan(npdata_instances), axis=1)
        npdata_no_null = npdata_instances[~nan_rows, :]
        scipy_kde = gaussian_kde(
            npdata_no_null.T,
            bw_method=lambda s: np.power(4 / (s.d + 2), 1 / (s.d + 4))
            * s.scotts_factor(),
        )

        assert (
            scipy_kde.n == cpd.num_instances()
        ), "Wrong number of training instances with null values."
        assert (
            scipy_kde.d == cpd.num_variables()
        ), "Wrong number of training variables with null values."
        assert np.all(
            np.isclose(scipy_kde.covariance, cpd.bandwidth)
        ), "Wrong bandwidth with null values."

    np.random.seed(0)
    a_null = np.random.randint(0, SIZE, size=100)
    b_null = np.random.randint(0, SIZE, size=100)
    c_null = np.random.randint(0, SIZE, size=100)
    d_null = np.random.randint(0, SIZE, size=100)

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

    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        for instances in [50, 150, 500]:
            _test_kde_fit_null_iter(variables, df_null, instances)
            _test_kde_fit_null_iter(variables, df_null_float, instances)


def test_kde_logl():
    """Tests the logl() method of the KDE factor. It compares the results with the ones obtained with scipy's product_kde.
    Both for float64 and float32 data types."""

    def _test_kde_logl_iter(variables, _df, _test_df):
        """Tests that the logl() method of the KDE factor returns the same results as scipy's product_kde.
        It trains _df and tests it with _test_df.
        Args:
            variables (list[str]): Dataset variables to use.
            _df (pd.DataFrame): Training dataset.
            _test_df (pd.DataFrame): Test dataset.
        """
        npdata = _df.loc[:, variables].to_numpy()
        cpd = pbn.KDE(
            variables,
            # bandwidth_selector=pbn.ScottsBandwidth(),
            bandwidth_selector=pbn.NormalReferenceRule(),
        )
        cpd.fit(_df)

        scipy_kde = gaussian_kde(
            dataset=npdata.T,
            # bw_method="scott",
            bw_method=lambda s: np.power(4 / (s.d + 2), 1 / (s.d + 4))
            * s.scotts_factor(),  # Normal Reference Rule multiplies Scott's factor and then standard deviation
        )

        # TODO: Add tests to check this
        # NOTE
        # scipy_kde.factor == scipy_kde.covariance_factor() <-- coefficient (kde.factor) that squared, multiplies the data covariance matrix to obtain the kernel covariance matrix.
        # scipy_kde.covariance == scipy_kde.factor ** 2 * npdata.var()
        # scipy_kde.inv_cov == 1 / scipy_kde.covariance
        # We check that the bandwidth is the same
        # TODO: Add tests to check "scott" bandwidth selectors
        assert np.all(np.isclose(cpd.bandwidth, scipy_kde.covariance))

        test_npdata = _test_df.loc[:, variables].to_numpy()

        logl = cpd.logl(_test_df)
        scipy_logl = scipy_kde.logpdf(test_npdata.T)

        if np.all(_df.dtypes == "float32"):
            assert np.all(np.isclose(logl, scipy_logl, atol=0.0005))
        else:
            assert np.all(np.isclose(logl, scipy_logl))

    test_df = generate_normal_data(50, seed=1)
    test_df_float = test_df.astype("float32")

    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        _test_kde_logl_iter(variables, df, test_df)
        _test_kde_logl_iter(variables, df_float, test_df_float)

    cpd = pbn.KDE(["d", "a", "b", "c"])
    cpd.fit(df)
    cpd2 = pbn.KDE(["a", "c", "d", "b"])
    cpd2.fit(df)
    assert np.all(
        np.isclose(cpd.logl(test_df), cpd2.logl(test_df))
    ), "Order of evidence changes logl() result."

    cpd = pbn.KDE(["d", "a", "b", "c"])
    cpd.fit(df_float)
    cpd2 = pbn.KDE(["a", "c", "d", "b"])
    cpd2.fit(df_float)
    assert np.all(
        np.isclose(cpd.logl(test_df_float), cpd2.logl(test_df_float))
    ), "Order of evidence changes logl() result."


def test_kde_logl_null():
    """Tests the logl() method of the KDE factor with null values. It compares the results with the ones obtained with scipy's product_kde.
    Both for float64 and float32 data types."""

    def _test_kde_logl_null_iter(variables, _df, _test_df):
        """Tests that the logl() method of the KDE factor with null values returns the same results as scipy's product_kde.
        It trains _df and tests it with _test_df.
        Args:
            variables (list[str]): Dataset variables to use.
            _df (pd.DataFrame): Training dataset.
            _test_df (pd.DataFrame): Test dataset.
        """
        cpd = pbn.KDE(variables)
        cpd.fit(_df)

        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(
            npdata.T,
            bw_method=lambda s: np.power(4 / (s.d + 2), 1 / (s.d + 4))
            * s.scotts_factor(),
        )
        # We initialize the logl and scipy_logl columns with NaN
        _test_df["logl"] = np.nan
        _test_df["scipy_logl"] = np.nan

        # We calculate the logl with the KDE factor
        _test_df["logl"] = cpd.logl(_test_df)

        # We calculate the logl with scipy (we have to avoid NaN values)
        non_nan_index = _test_df[variables].notna().all(1)
        _test_df.loc[non_nan_index, "scipy_logl"] = scipy_kde.logpdf(
            _test_df.loc[non_nan_index, variables].T.to_numpy()
        )

        if npdata.dtype == "float32":
            assert np.all(
                np.isclose(
                    _test_df["logl"],
                    _test_df["scipy_logl"],
                    atol=0.0005,
                    equal_nan=True,
                )
            )
        else:
            assert np.all(
                np.isclose(_test_df["logl"], _test_df["scipy_logl"], equal_nan=True)
            )

    TEST_SIZE = 50

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

    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        _test_kde_logl_null_iter(variables, df, df_null)
        _test_kde_logl_null_iter(variables, df_float, df_null_float)

    cpd = pbn.KDE(["d", "a", "b", "c"])
    cpd.fit(df)
    cpd2 = pbn.KDE(["a", "c", "d", "b"])
    cpd2.fit(df)
    assert np.all(
        np.isclose(cpd.logl(df_null), cpd2.logl(df_null), equal_nan=True)
    ), "Order of evidence changes logl() result."

    cpd = pbn.KDE(["d", "a", "b", "c"])
    cpd.fit(df_float)
    cpd2 = pbn.KDE(["a", "c", "d", "b"])
    cpd2.fit(df_float)
    assert np.all(
        np.isclose(
            cpd.logl(df_null_float),
            cpd2.logl(df_null_float),
            atol=0.0005,
            equal_nan=True,
        )
    ), "Order of evidence changes logl() result."


def test_kde_slogl():
    """Tests the slogl() method of the KDE factor. It compares the results with the ones obtained with scipy's product_kde.
    Both for float64 and float32 data types."""

    def _test_kde_slogl_iter(variables, _df, _test_df):
        """Tests that the logl() method of the KDE factor returns the same results as scipy's product_kde.
        It trains _df and tests it with _test_df.
        Args:
            variables (list[str]): Dataset variables to use.
            _df (pd.DataFrame): Training dataset.
            _test_df (pd.DataFrame): Test dataset.
        """
        cpd = pbn.KDE(variables)
        cpd.fit(_df)

        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(
            npdata.T,
            bw_method=lambda s: np.power(4 / (s.d + 2), 1 / (s.d + 4))
            * s.scotts_factor(),
        )

        test_npdata = _test_df.loc[:, variables].to_numpy()
        assert np.all(
            np.isclose(cpd.slogl(_test_df), scipy_kde.logpdf(test_npdata.T).sum())
        )

    test_df = generate_normal_data(50, seed=1)
    test_df_float = test_df.astype("float32")

    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        _test_kde_slogl_iter(variables, df, test_df)
        _test_kde_slogl_iter(variables, df_float, test_df_float)

    cpd = pbn.KDE(["d", "a", "b", "c"])
    cpd.fit(df)
    cpd2 = pbn.KDE(["a", "c", "d", "b"])
    cpd2.fit(df)
    assert np.all(
        np.isclose(cpd.slogl(test_df), cpd2.slogl(test_df))
    ), "Order of evidence changes slogl() result."

    cpd = pbn.KDE(["d", "a", "b", "c"])
    cpd.fit(df_float)
    cpd2 = pbn.KDE(["a", "c", "d", "b"])
    cpd2.fit(df_float)
    assert np.all(
        np.isclose(cpd.slogl(test_df_float), cpd2.slogl(test_df_float))
    ), "Order of evidence changes slogl() result."


def test_kde_slogl_null():
    """Tests the slogl() method of the KDE factor with null values. It compares the results with the ones obtained with scipy's product_kde.
    Both for float64 and float32 data types."""

    def _test_kde_slogl_null_iter(variables, _df, _test_df):
        """Tests that the slogl() method of the KDE factor with null values returns the same results as scipy's product_kde.
        It trains _df and tests it with _test_df.
        Args:
            variables (list[str]): Dataset variables to use.
            _df (pd.DataFrame): Training dataset.
            _test_df (pd.DataFrame): Test dataset.
        """
        cpd = pbn.KDE(variables)
        cpd.fit(_df)

        npdata = _df.loc[:, variables].to_numpy()
        scipy_kde = gaussian_kde(
            npdata.T,
            bw_method=lambda s: np.power(4 / (s.d + 2), 1 / (s.d + 4))
            * s.scotts_factor(),
        )
        # We initialize the logl and scipy_logl columns with NaN
        _test_df["scipy_logl"] = np.nan
        slogl = cpd.slogl(_test_df)
        # We calculate the logl with scipy (we have to avoid NaN values)
        non_nan_index = _test_df[variables].notna().all(1)
        scipy_slogl = scipy_kde.logpdf(
            _test_df.loc[non_nan_index, variables].T.to_numpy()
        ).sum()

        assert np.all(np.isclose(slogl, scipy_slogl))

    TEST_SIZE = 50

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

    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        _test_kde_slogl_null_iter(variables, df, df_null)
        _test_kde_slogl_null_iter(variables, df_float, df_null_float)

    cpd = pbn.KDE(["d", "a", "b", "c"])
    cpd.fit(df)
    cpd2 = pbn.KDE(["a", "c", "d", "b"])
    cpd2.fit(df)
    assert np.all(
        np.isclose(cpd.slogl(df_null), cpd2.slogl(df_null))
    ), "Order of evidence changes slogl() result."

    cpd = pbn.KDE(["d", "a", "b", "c"])
    cpd.fit(df_float)
    cpd2 = pbn.KDE(["a", "c", "d", "b"])
    cpd2.fit(df_float)
    assert np.all(
        np.isclose(cpd.slogl(df_null_float), cpd2.slogl(df_null_float))
    ), "Order of evidence changes slogl() result."
