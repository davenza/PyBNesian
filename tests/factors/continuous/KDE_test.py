import numpy as np
import pyarrow as pa
import pybnesian as pbn
import pytest
from helpers.data import generate_normal_data
from helpers.kde import diagonal_kde_logpdf, normal_reference_bandwidth
from scipy.special import logsumexp

SIZE = 500
df = generate_normal_data(SIZE, seed=0)
df_float = df.astype("float32")


def test_check_type() -> None:
    """
    Tests that the KDE factor raises a ValueError when the data type of the test dataset
    is different from the data type of the training dataset during log-likelihood and
    smoothed log-likelihood computations.
    """

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
    """
    Tests the initialization of the KDE class with different sets of variables.
    For each list of variable names, this test creates a KDE object and asserts
    that the object's variables match the input list. This ensures that the KDE
    class correctly stores and returns its variables upon initialization.
    """

    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        cpd = pbn.KDE(variables)
        assert cpd.variables() == variables


def test_kde_bandwidth():
    """
    Tests the bandwidth selection and assignment functionality of the KDE class.
    This test verifies:
    - That the KDE bandwidth computed using the normal reference rule matches the output of scipy's gaussian_kde with a custom bandwidth method, for various variable sets and sample sizes.
    - That the KDE bandwidth computed using Scott's rule matches the output of scipy's gaussian_kde default bandwidth, for various variable sets and sample sizes.
    - That the bandwidth attribute of the KDE object can be manually set and correctly reflects the assigned value.
    The test is performed for both integer and float dataframes.
    """

    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        for instances in [50, 1000, 10000]:
            training_df = df.iloc[:instances].loc[:, variables].dropna()
            bandwidth = normal_reference_bandwidth(training_df, variables)

            cpd = pbn.KDE(variables)
            cpd.fit(df.iloc[:instances])
            assert np.all(
                np.isclose(cpd.bandwidth, np.diag(bandwidth))
            ), "Wrong bandwidth computed with normal reference rule."

            cpd.fit(df_float.iloc[:instances])
            assert np.all(
                np.isclose(cpd.bandwidth, np.diag(bandwidth))
            ), "Wrong bandwidth computed with normal reference rule."

            scott_bandwidth = (
                np.power(training_df.shape[0], -2 / (len(variables) + 4))
                * training_df.var().to_numpy()
            )

            cpd = pbn.KDE(variables, pbn.ScottsBandwidth())
            cpd.fit(df.iloc[:instances])
            assert np.all(
                np.isclose(np.diag(cpd.bandwidth), scott_bandwidth)
            ), "Wrong bandwidth computed with Scott's rule."

            cpd.fit(df_float.iloc[:instances])
            assert np.all(
                np.isclose(np.diag(cpd.bandwidth), scott_bandwidth)
            ), "Wrong bandwidth computed with Scott's rule."

    cpd = pbn.KDE(["a"])
    cpd.fit(df)
    cpd.bandwidth = [[1]]
    assert cpd.bandwidth == np.asarray([[1]]), "Could not change bandwidth."

    cpd.fit(df_float)
    cpd.bandwidth = [[1]]
    assert cpd.bandwidth == np.asarray([[1]]), "Could not change bandwidth."


class UnitaryBandwidth(pbn.BandwidthSelector):
    """
    A bandwidth selector that returns the identity matrix as the bandwidth.
    This class is a subclass of `pbn.BandwidthSelector` and implements a simple bandwidth selection strategy
    where the bandwidth matrix is always the identity matrix of size equal to the number of variables.
    Methods
    -------
    __init__():
        Initializes the UnitaryBandwidth selector.
    bandwidth(df, variables):
        Returns the identity matrix of shape (len(variables), len(variables)) as the bandwidth matrix.
    Parameters
    ----------
    df : pandas.DataFrame
        The data frame containing the data (not used in this selector).
    variables : list
        The list of variables for which the bandwidth is to be computed.
    Returns
    -------
    numpy.ndarray
        An identity matrix of size equal to the number of variables.
    """

    def __init__(self):
        pbn.BandwidthSelector.__init__(self)

    def bandwidth(self, df, variables):
        return np.eye(len(variables))


def test_kde_new_bandwidth():
    """
    Tests the behavior of the KDE class when using the UnitaryBandwidth bandwidth selector.
    This test verifies that:
    - When fitting a KDE with a single variable, the resulting bandwidth matrix is the 1x1 identity matrix.
    - When fitting a KDE with four variables, the resulting bandwidth matrix is the 4x4 identity matrix.
    - The behavior is consistent for both integer and float dataframes.
    Assertions:
        - The bandwidth matrix after fitting is as expected (identity matrix) for both data types and variable counts.
    """

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
    """
    Tests the `data_type` method of the KDE factor.
    This test verifies that:
    - Calling `data_type` before fitting the KDE raises a ValueError with the message "KDE factor not fitted".
    - After fitting the KDE with a DataFrame `df`, the returned data type is `pa.float64()`.
    - After fitting the KDE with a DataFrame `df_float`, the returned data type is `pa.float32()`.
    """

    k = pbn.KDE(["a"])

    with pytest.raises(ValueError) as ex:
        k.data_type()
    assert "KDE factor not fitted" in str(ex.value)

    k.fit(df)
    assert k.data_type() == pa.float64()
    k.fit(df_float)
    assert k.data_type() == pa.float32()


def test_kde_fit():
    """
    Tests the fitting process of the KDE (Kernel Density Estimation) class in the PyBNesian library.
    This test verifies that:
    - The KDE object is not fitted before calling `fit`.
    - After fitting with a subset of the provided DataFrame, the KDE object is marked as fitted.
    - The number of training instances and variables in the fitted KDE matches those of a reference `scipy.stats.gaussian_kde` object.
    - The test is performed for different combinations of variables and different numbers of training instances, using both integer and float DataFrames.
    Tested scenarios:
    - Single and multiple variable KDEs.
    - Different sample sizes (50, 150, 500).
    - Both integer and float data types.
    """

    def _test_kde_fit_iter(variables, _df, instances):
        cpd = pbn.KDE(variables)
        assert not cpd.fitted()
        cpd.fit(_df.iloc[:instances, :])
        assert cpd.fitted()

        bandwidth = normal_reference_bandwidth(
            _df.iloc[:instances].loc[:, variables].dropna(), variables
        )

        assert cpd.num_instances() == instances, "Wrong number of training instances."
        assert (
            len(variables) == cpd.num_variables()
        ), "Wrong number of training variables."
        assert np.all(np.isclose(cpd.bandwidth, np.diag(bandwidth)))

    for variables in [["a"], ["b", "a"], ["c", "a", "b"], ["d", "a", "b", "c"]]:
        for instances in [50, 150, 500]:
            _test_kde_fit_iter(variables, df, instances)
            _test_kde_fit_iter(variables, df_float, instances)


def test_kde_fit_null():
    """
    Test the fitting of the KDE (Kernel Density Estimator) model when input data contains null (NaN) values.
    This test verifies that:
    - The KDE model is not fitted before calling `fit` and is fitted after.
    - The model correctly ignores rows with null values during fitting.
    - The number of training instances and variables in the fitted model matches those in a reference `scipy.stats.gaussian_kde` fitted on the same data with nulls removed.
    - The computed bandwidth (covariance) of the KDE matches that of the reference implementation.
    The test is performed for different combinations of variables and different numbers of training instances, using both integer and float dataframes with randomly inserted NaN values.
    """

    def _test_kde_fit_null_iter(variables, _df, instances):
        cpd = pbn.KDE(variables)
        assert not cpd.fitted()
        cpd.fit(_df.iloc[:instances, :])
        assert cpd.fitted()

        npdata = _df.loc[:, variables].to_numpy()
        npdata_instances = npdata[:instances, :]

        nan_rows = np.any(np.isnan(npdata_instances), axis=1)
        training_df = _df.iloc[:instances].loc[:, variables].dropna()
        bandwidth = normal_reference_bandwidth(training_df, variables)

        assert (
            training_df.shape[0] == cpd.num_instances()
        ), "Wrong number of training instances with null values."
        assert (
            len(variables) == cpd.num_variables()
        ), "Wrong number of training variables with null values."
        assert np.all(
            np.isclose(cpd.bandwidth, np.diag(bandwidth))
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

        training_df = _df.loc[:, variables].dropna()
        bandwidth = normal_reference_bandwidth(training_df, variables)
        assert np.all(np.isclose(cpd.bandwidth, np.diag(bandwidth)))

        logl = cpd.logl(_test_df)
        scipy_logl = diagonal_kde_logpdf(_test_df, training_df, variables, bandwidth)

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

        training_df = _df.loc[:, variables].dropna()
        bandwidth = normal_reference_bandwidth(training_df, variables)
        # We initialize the logl and scipy_logl columns with NaN
        _test_df["logl"] = np.nan
        _test_df["scipy_logl"] = np.nan

        # We calculate the logl with the KDE factor
        _test_df["logl"] = cpd.logl(_test_df)

        # We calculate the logl with scipy (we have to avoid NaN values)
        non_nan_index = _test_df[variables].notna().all(1)
        _test_df.loc[non_nan_index, "scipy_logl"] = diagonal_kde_logpdf(
            _test_df.loc[non_nan_index], training_df, variables, bandwidth
        )

        if np.all(_df.dtypes == "float32"):
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

        training_df = _df.loc[:, variables].dropna()
        bandwidth = normal_reference_bandwidth(training_df, variables)

        assert np.all(
            np.isclose(
                cpd.slogl(_test_df),
                diagonal_kde_logpdf(_test_df, training_df, variables, bandwidth).sum(),
            )
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

        training_df = _df.loc[:, variables].dropna()
        bandwidth = normal_reference_bandwidth(training_df, variables)
        # We initialize the logl and scipy_logl columns with NaN
        _test_df["scipy_logl"] = np.nan
        slogl = cpd.slogl(_test_df)
        # We calculate the logl with scipy (we have to avoid NaN values)
        non_nan_index = _test_df[variables].notna().all(1)
        scipy_slogl = diagonal_kde_logpdf(
            _test_df.loc[non_nan_index], training_df, variables, bandwidth
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
