import numpy as np
import pandas as pd
import pybnesian as pbn

import util_test

SIZE = 10000

df = util_test.generate_normal_data(SIZE)

def test_holdout_disjoint():
    hold = pbn.HoldOut(df)

    train_df, test_df = hold.training_data(), hold.test_data()

    assert (train_df.num_rows + test_df.num_rows) == SIZE, "HoldOut do not have the expected number of rows"

    assert train_df.num_rows == round((1-0.2) * df.shape[0]), "Train DataFrame do not have the expected number of instances"
    assert test_df.num_rows == round(0.2 * df.shape[0]), "Test DataFrame do not have the expected number of instances"

    combination = pd.concat([train_df.to_pandas(), test_df.to_pandas()])

    assert df.sort_values("a", axis=0).reset_index(drop=True)\
                .equals(combination.sort_values("a", axis=0).reset_index(drop=True)),\
                 "The combination of train and test dataset is not equal to the original DataFrame."
    
    hold = pbn.HoldOut(df, test_ratio=0.3)
    train_df, test_df = hold.training_data(), hold.test_data()

    assert (train_df.num_rows + test_df.num_rows) == SIZE, "HoldOut do not have the expected number of rows"

    assert train_df.num_rows == round((1-0.3) * df.shape[0]), "Train DataFrame do not have the expected number of instances"
    assert test_df.num_rows == round(0.3 * df.shape[0]), "Test DataFrame do not have the expected number of instances"

    combination = pd.concat([train_df.to_pandas(), test_df.to_pandas()])

    assert df.sort_values("a", axis=0).reset_index(drop=True)\
                .equals(combination.sort_values("a", axis=0).reset_index(drop=True)),\
                 "The combination of train and test dataset is not equal to the original DataFrame."

def test_holdout_seed():
    hold = pbn.HoldOut(df, seed=0)
    hold2 = pbn.HoldOut(df, seed=0)

    train_df, test_df = hold.training_data(), hold.test_data()
    train_df2, test_df2 = hold2.training_data(), hold2.test_data()

    assert train_df.equals(train_df2), "Train CV DataFrames with the same seed are not equal."
    assert test_df.equals(test_df2), "Test CV DataFrames with the same seed are not equal."

    hold3 = pbn.HoldOut(df, seed=1)
    train_df3, test_df3 = hold3.training_data(), hold3.test_data()

    assert not train_df.equals(train_df3), "Train CV DataFrames with different seeds return the same result."
    assert not test_df.equals(test_df3), "Test CV DataFrames with different seeds return the same result."

def test_holdout_null():
    np.random.seed(0)
    a_null = np.random.randint(0, SIZE, size=100)
    b_null = np.random.randint(0, SIZE, size=100)
    c_null = np.random.randint(0, SIZE, size=100)
    d_null = np.random.randint(0, SIZE, size=100)

    df_null = df
    df_null.loc[df_null.index[a_null], 'a'] = np.nan
    df_null.loc[df_null.index[b_null], 'b'] = np.nan
    df_null.loc[df_null.index[c_null], 'c'] = np.nan
    df_null.loc[df_null.index[d_null], 'd'] = np.nan

    non_null = df_null.dropna()
    hold = pbn.HoldOut(df_null)

    train_df, test_df = hold.training_data(), hold.test_data()

    assert (train_df.num_rows + test_df.num_rows) == non_null.shape[0], "HoldOut do not have the expected number of rows"
    assert train_df.num_rows == round((1-0.2) * non_null.shape[0]), "Train DataFrame do not have the expected number of instances"
    assert test_df.num_rows == round(0.2 * non_null.shape[0]), "Test DataFrame do not have the expected number of instances"

    combination = pd.concat([train_df.to_pandas(), test_df.to_pandas()])

    assert combination.sort_values("a", axis=0).reset_index(drop=True)\
                .equals(non_null.sort_values("a", axis=0).reset_index(drop=True)),\
                 "The combination of train and test dataset is not equal to the original DataFrame."

    hold_null = pbn.HoldOut(df_null, include_null=True)
    train_df, test_df = hold_null.training_data(), hold_null.test_data()
    assert (train_df.num_rows + test_df.num_rows) == SIZE, "HoldOut do not have the expected number of rows"
    assert train_df.num_rows == round((1-0.2) * SIZE), "Train DataFrame do not have the expected number of instances"
    assert test_df.num_rows == round(0.2 * SIZE), "Test DataFrame do not have the expected number of instances"

    combination = pd.concat([train_df.to_pandas(), test_df.to_pandas()])

    assert combination.sort_values(["a", "b", "c", "d"], axis=0).reset_index(drop=True)\
                .equals(df.sort_values(["a", "b", "c", "d"], axis=0).reset_index(drop=True)),\
                 "The combination of train and test dataset is not equal to the original DataFrame."

    