import pytest
import numpy as np
import pybnesian as pbn
import util_test

SIZE = 10000
df = util_test.generate_normal_data(SIZE)


def numpy_fit_mle_lg(data, variable, evidence):
    if isinstance(variable, str):
        node_data = data.loc[:, [variable] + evidence].dropna()
        variable_data = node_data.loc[:, variable]
        evidence_data = node_data.loc[:, evidence]
    else:
        node_data = data.iloc[:, [variable] + evidence].dropna()
        variable_data = node_data.iloc[:, 0]
        evidence_data = node_data.iloc[:, 1:]

    N = variable_data.shape[0]
    d = evidence_data.shape[1]
    linregress_data = np.column_stack((np.ones(N), evidence_data.to_numpy()))
    (beta, res, _, _) = np.linalg.lstsq(linregress_data, variable_data.to_numpy(), rcond=None)
    var = res / (N - d - 1)

    return beta, var

def test_mle_create():
    with pytest.raises(ValueError) as ex:
        mle = pbn.MLE(pbn.CKDEType())
    assert "MLE not available" in str(ex.value)

    mle = pbn.MLE(pbn.LinearGaussianCPDType())

def test_mle_lg():
    mle = pbn.MLE(pbn.LinearGaussianCPDType())

    p = mle.estimate(df, "a", [])
    np_beta, np_var = numpy_fit_mle_lg(df, "a", [])
    assert np.all(np.isclose(p.beta, np_beta))
    assert np.isclose(p.variance, np_var)

    p = mle.estimate(df, "b", ["a"])
    np_beta, np_var = numpy_fit_mle_lg(df, "b", ["a"])
    assert np.all(np.isclose(p.beta, np_beta))
    assert np.isclose(p.variance, np_var)

    p = mle.estimate(df, "c", ["a", "b"])
    np_beta, np_var = numpy_fit_mle_lg(df, "c", ["a", "b"])
    assert np.all(np.isclose(p.beta, np_beta))
    assert np.isclose(p.variance, np_var)

    p = mle.estimate(df, "d", ["a", "b", "c"])
    np_beta, np_var = numpy_fit_mle_lg(df, "d", ["a", "b", "c"])
    assert np.all(np.isclose(p.beta, np_beta))
    assert np.isclose(p.variance, np_var)