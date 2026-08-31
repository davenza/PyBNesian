import numpy as np
import pandas as pd
from scipy.special import logsumexp


def normal_reference_bandwidth(df: pd.DataFrame, variables: list[str]) -> np.ndarray:
    """Return the diagonal normal-reference bandwidth for ``variables``.

    The implementation matches PyBNesian's current normal-reference selector,
    which keeps only the diagonal covariance terms and scales them with the
    diagonal bandwidth factor.
    """
    if not variables:
        return np.empty(0)

    cov = df.loc[:, variables].cov().to_numpy()
    cov = np.diag(np.diag(cov))
    n = df.shape[0]
    d = len(variables)

    k = np.power(4.0 / (n * (d + 2.0)), 2.0 / (d + 4.0))
    return k * np.diag(cov)


def diagonal_kernel_logpdf_matrix(
    test_values: np.ndarray, training_values: np.ndarray, bandwidth: np.ndarray
) -> np.ndarray:
    """Compute per-sample log kernel values for a diagonal Gaussian KDE.

    The returned matrix has one row per test sample and one column per training
    sample.
    """
    if training_values.shape[0] == 0:
        raise ValueError("Training data must contain at least one row.")

    d = training_values.shape[1]
    diff = test_values[:, None, :] - training_values[None, :, :]
    log_kernel = -0.5 * np.sum((diff * diff) / bandwidth, axis=2)
    log_kernel -= 0.5 * np.log(bandwidth).sum()
    log_kernel -= 0.5 * d * np.log(2 * np.pi)
    return log_kernel


def diagonal_kde_logpdf(
    test_data: pd.DataFrame,
    training_data: pd.DataFrame,
    variables: list[str],
    bandwidth: np.ndarray,
) -> np.ndarray:
    """Evaluate the log-density of a diagonal KDE on ``test_data``.

    Rows are scored against the training data using the diagonal bandwidth
    vector and combined with a log-sum-exp reduction.
    """
    test_values = test_data.loc[:, variables].to_numpy()
    training_values = training_data.loc[:, variables].to_numpy()

    if len(variables) == 0:
        return np.zeros(test_values.shape[0], dtype=test_values.dtype)

    log_kernel = diagonal_kernel_logpdf_matrix(test_values, training_values, bandwidth)
    return logsumexp(log_kernel, axis=1) - np.log(training_values.shape[0])
