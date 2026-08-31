import itertools

import numpy as np
import pandas as pd
import pybnesian as pbn
from helpers.data import (
    DATA_SIZE,
    N_NEIGHBORS,
    SEED,
    generate_discrete_data,
    generate_discrete_data_independent,
    generate_normal_data,
    generate_normal_data_independent,
)
from scipy.stats import pearsonr

# from sklearn.feature_selection import mutual_info_regression

data = generate_normal_data(DATA_SIZE, SEED)
independent_data = generate_normal_data_independent(DATA_SIZE, SEED)

discrete_data = generate_discrete_data(DATA_SIZE, SEED)
independent_discrete_data = generate_discrete_data_independent(DATA_SIZE, SEED)

# INDEPENDENCE TESTS
# The null hypothesis (H0​) is that the two variables are independent,
# while the alternative hypothesis (H1) is that the two variables are dependent
#
# - If the p-value is less than or equal to the chosen significance level (usually 0.05),
# you reject the null hypothesis (H0H) in favor of the alternative hypothesis (H1).
# This suggests that there is a statistically significant association between the two variables.
#
# - If the p-value is greater than the significance level, you do not reject the null hypothesis.
# This indicates that there is insufficient evidence to conclude that the variables are dependent,
# and it is plausible that they are independent


def test_chi_square():
    """Test the chi-square independence test with discrete data"""
    chi_square = pbn.ChiSquare(discrete_data)
    independent_chi_square = pbn.ChiSquare(independent_discrete_data)

    p_value = chi_square.pvalue("a", "b")
    independent_p_value = independent_chi_square.pvalue("a", "b")

    # Check whether the p-values are below the significance level
    assert p_value < 0.05
    assert independent_p_value > 0.05


# RFE: Test true and false independence
def test_linear_correlation():
    """Test the linear correlation independence test with normal data"""
    df = data[["a", "b"]]
    independent_df = independent_data[["a", "b"]]

    # Pybnesian Linear correlation
    linear_correlation = pbn.LinearCorrelation(df)
    independent_linear_correlation = pbn.LinearCorrelation(independent_df)
    pvalue = linear_correlation.pvalue("a", "b")
    independent_pvalue = independent_linear_correlation.pvalue("a", "b")

    # scipy pearsonr correlation
    correlations = {}
    columns = df.columns.tolist()
    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[col_a + "__" + col_b] = pearsonr(
            df.loc[:, col_a], df.loc[:, col_b]
        )
    result = pd.DataFrame.from_dict(
        correlations, orient="index", columns=["PCC", "p-value"]
    )

    # Compare correlation values
    np.testing.assert_allclose(
        np.array([df.corr().loc["a", "b"]]),
        np.array([result.loc["a__b", "PCC"]]),
        rtol=1e-5,
        atol=1e-8,
    )
    # Compare p-values
    np.testing.assert_allclose(
        np.array([pvalue]),
        np.array([result.loc["a__b", "p-value"]]),
        rtol=1e-5,
        atol=1e-8,
    )

    # Check whether the p-values are below the significance level
    assert pvalue < 0.05
    assert independent_pvalue > 0.05


def test_mutual_info():
    """Test the mutual information independence test with normal data"""
    mutual_info = pbn.MutualInformation(data)
    independent_mutual_info = pbn.MutualInformation(independent_data)

    # Check whether the mutual information is higher when the variables are dependent
    mutual_info_value = mutual_info.mi("a", "b")
    independent_mutual_info_value = independent_mutual_info.mi("a", "b")
    assert mutual_info_value > independent_mutual_info_value

    # Check whether the p-values are below the significance level
    pvalue = mutual_info.pvalue("a", "b")
    independent_pvalue = independent_mutual_info.pvalue("a", "b")
    assert pvalue < 0.05
    assert independent_pvalue > 0.05


def test_k_mutual_info():
    """Test the k-nearest neighbors mutual information independence test with normal data"""
    k_mutual_info = pbn.KMutualInformation(data, k=N_NEIGHBORS)
    independent_k_mutual_info = pbn.KMutualInformation(independent_data, k=N_NEIGHBORS)

    # Check whether the mutual information is higher when the variables are dependent
    k_mutual_info_value = k_mutual_info.mi("a", "b")
    independent_k_mutual_info_value = independent_k_mutual_info.mi("a", "b")
    assert k_mutual_info_value > independent_k_mutual_info_value

    # Check whether the p-values are below the significance level
    # NOTE: Slow execution
    pvalue = k_mutual_info.pvalue("a", "b")
    independent_pvalue = independent_k_mutual_info.pvalue("a", "b")
    assert pvalue < 0.05
    assert independent_pvalue > 0.05

    # RFE: Results vary with scikit-learn, why?

    # sklearn_k_mutual_info_value = mutual_info_regression(
    #     data[["a"]], data["b"], n_neighbors=n_neighbors
    # )[0]
    # print(k_mutual_info_value)
    # print("\n", sklearn_k_mutual_info_value)
    # np.testing.assert_allclose(
    #     sklearn_k_mutual_info_value,
    #     np.array([k_mutual_info_value]),
    #     rtol=1e-5,
    #     atol=1e-8,
    # )
    # RFE: Test alternative https://github.com/syanga/pycit


def test_rcot():
    """Test the Randomized Conditional Correlation Test (pbn.RCoT) independence test with normal data"""
    rcot = pbn.RCoT(data, random_fourier_xy=5, random_fourier_z=100)
    independent_rcot = pbn.RCoT(
        independent_data, random_fourier_xy=5, random_fourier_z=100
    )
    p_value = rcot.pvalue("a", "b")
    independent_p_value = independent_rcot.pvalue("a", "b")

    # Check whether the p-values are below the significance level
    assert p_value < 0.05
    assert independent_p_value > 0.05
