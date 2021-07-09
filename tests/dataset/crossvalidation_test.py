import numpy as np
import pybnesian as pbn

import util_test

SIZE = 10000

df = util_test.generate_normal_data(SIZE)

def test_cv_disjoint_indices():
    cv = pbn.CrossValidation(df)

    for (train_df, test_df), (train_indices, test_indices) in zip(cv, cv.indices()):
        nptrain = np.asarray(train_indices)
        nptest = np.asarray(test_indices)
        combination = np.hstack((nptrain, nptest))
        
        assert np.all(np.sort(combination) == np.arange(SIZE)), "Not all the examples are included in the cross validation."
        assert np.all(train_df.to_pandas().to_numpy() == df.iloc[train_indices,:].to_numpy()), \
                                "The CV iterator do not slice the train dataset exactly equal as the CV indices iterator."
        assert np.all(test_df.to_pandas().to_numpy() == df.iloc[test_indices,:].to_numpy()), \
                                "The CV iterator do not slice the test dataset exactly equal as the CV indices iterator."

        assert np.setdiff1d(nptrain, nptest).shape == nptrain.shape, "The train indices includes test indices"
        assert np.setdiff1d(nptest, nptrain).shape == nptest.shape, "The test indices includes train indices"
        assert np.all(np.sort(np.setdiff1d(train_indices, test_indices)) == np.sort(train_indices)), "The train indices includes test indices"
        assert np.all(np.sort(np.setdiff1d(test_indices, train_indices)) == np.sort(test_indices)), "The test indices includes train indices"


def test_cv_fold():
    cv = pbn.CrossValidation(df)

    for i, (train_df, test_df) in enumerate(cv):
        train_fold, test_fold = cv.fold(i)

        assert train_fold.equals(train_df), "Train DataFrame fold() and __iter__ are not equal."
        assert test_fold.equals(test_df), "Test DataFrame fold() and __iter__ are not equal."


def test_cv_seed():
    cv = pbn.CrossValidation(df, seed=0)
    
    dataframes = list(cv)

    cv2 = pbn.CrossValidation(df, seed=0)

    for (train_cv, test_cv), (train_cv2, test_cv2) in zip(dataframes, cv2):
        assert train_cv.equals(train_cv2), "Train CV DataFrames with the same seed are not equal."
        assert test_cv.equals(test_cv2), "Test CV DataFrames with the same seed are not equal."

    cv3 = pbn.CrossValidation(df, seed=1)
    for (train_cv2, test_cv2), (train_cv3, test_cv3) in zip(cv2, cv3):
        assert not train_cv2.equals(train_cv3), "Train CV DataFrames with different seeds return the same result."
        assert not test_cv2.equals(test_cv3), "Test CV DataFrames with different seeds return the same result."

def test_cv_num_folds():
    cv = pbn.CrossValidation(df)
    
    dataframes = list(cv)
    indices = list(cv.indices())

    assert len(dataframes) == 10, "Default number of folds must be 10."
    assert len(indices) == 10, "Default number of folds must be 10."
    
    cv5 = pbn.CrossValidation(df, 5)
    dataframes = list(cv5)
    indices = list(cv5.indices())
    assert len(dataframes) == 5, "Wrong number of folds"
    assert len(indices) == 5, "Wrong number of folds for the indices iterator."


def test_cv_loc():
    cv = pbn.CrossValidation(df)
    
    for (train_df, test_df) in cv.loc("a"):
        assert train_df.num_columns == 1, "Only column \"a\" must be present in train DataFrame."
        assert test_df.num_columns == 1, "Only column \"a\" must be present in test DataFrame."
        train_schema = train_df.schema
        test_schema = test_df.schema
        assert train_schema.names == ["a"], "Only column \"a\" must be present in train DataFrame."
        assert test_schema.names == ["a"], "Only column \"a\" must be present in test DataFrame."

    for (train_df, test_df) in cv.loc(1):
        assert train_df.num_columns == 1, "Only column \"b\" must be present in train DataFrame."
        assert test_df.num_columns == 1, "Only column \"b\" must be present in test DataFrame."
        train_schema = train_df.schema
        test_schema = test_df.schema
        assert train_schema.names == ["b"], "Only column \"b\" must be present in train DataFrame."
        assert test_schema.names == ["b"], "Only column \"b\" must be present in test DataFrame."

    for (train_df, test_df) in cv.loc(["b", "d"]):
        assert train_df.num_columns == 2, "Only columns [\"b\", \"d\"] must be present in train DataFrame."
        assert test_df.num_columns == 2, "Only column [\"b\", \"d\"] must be present in test DataFrame."
        train_schema = train_df.schema
        test_schema = test_df.schema
        assert train_schema.names == ["b", "d"], "Only column [\"b\", \"d\"] must be present in train DataFrame."
        assert test_schema.names == ["b", "d"], "Only column [\"b\", \"d\"] must be present in test DataFrame."

    for (train_df, test_df) in cv.loc([0, 2]):
        assert train_df.num_columns == 2, "Only columns [\"a\", \"c\"] must be present in train DataFrame."
        assert test_df.num_columns == 2, "Only column [\"a\", \"c\"] must be present in test DataFrame."
        train_schema = train_df.schema
        test_schema = test_df.schema
        assert train_schema.names == ["a", "c"], "Only column [\"a\", \"c\"] must be present in train DataFrame."
        assert test_schema.names == ["a", "c"], "Only column [\"a\", \"c\"] must be present in test DataFrame."


def test_cv_null():
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
    cv = pbn.CrossValidation(df_null)

    for (train_df, test_df), (train_indices, test_indices) in zip(cv, cv.indices()):
        assert non_null.shape[0] == (train_df.num_rows + test_df.num_rows), "CV did not remove null instances correctly."

        nptrain = np.asarray(train_indices)
        nptest = np.asarray(test_indices)
        combination = np.hstack((nptrain, nptest))

        actual_combination = np.sort(np.setdiff1d(np.arange(SIZE), np.asarray(list(set(list(a_null) + list(b_null) + list(c_null) + list(d_null))))))
        
        assert np.all(np.sort(combination) == actual_combination), "Not all the examples are included in the cross validation."
        assert np.all(train_df.to_pandas().to_numpy() == df.iloc[train_indices,:].to_numpy()), \
                                "The CV iterator do not slice the train dataset exactly equal as the CV indices iterator."
        assert np.all(test_df.to_pandas().to_numpy() == df.iloc[test_indices,:].to_numpy()), \
                                "The CV iterator do not slice the test dataset exactly equal as the CV indices iterator."

        assert np.setdiff1d(nptrain, nptest).shape == nptrain.shape, "The train indices includes test indices"
        assert np.setdiff1d(nptest, nptrain).shape == nptest.shape, "The test indices includes train indices"
        assert np.all(np.sort(np.setdiff1d(train_indices, test_indices)) == np.sort(train_indices)), "The train indices includes test indices"
        assert np.all(np.sort(np.setdiff1d(test_indices, train_indices)) == np.sort(test_indices)), "The test indices includes train indices"

    cv_include_null = pbn.CrossValidation(df_null, include_null=True)

    for (train_df, test_df), (train_indices, test_indices) in zip(cv_include_null, cv_include_null.indices()):
        assert (train_df.num_rows + test_df.num_rows) == SIZE, "CV did not remove null instances correctly."

        nptrain = np.asarray(train_indices)
        nptest = np.asarray(test_indices)
        combination = np.hstack((nptrain, nptest))

        train_df_mat = train_df.to_pandas().to_numpy()
        train_indices_mat = df.iloc[train_indices,:].to_numpy()
        test_df_mat = test_df.to_pandas().to_numpy()
        test_indices_mat = df.iloc[test_indices,:].to_numpy()
        
        assert np.all(np.sort(combination) == np.arange(SIZE)), "Not all the examples are included in the cross validation."
        assert np.all(np.isnan(train_df_mat) == np.isnan(train_indices_mat)), \
                                                                "The null values are wrongly specified in the train DataFrame."

        assert np.all(train_df_mat[~np.isnan(train_df_mat)] == train_indices_mat[~np.isnan(train_df_mat)]), \
                                "The CV iterator do not slice the train dataset exactly equal as the CV indices iterator."

        assert np.all(np.isnan(test_df_mat) == np.isnan(test_indices_mat)), \
                                                                "The null values are wrongly specified in the test DataFrame."
        assert np.all(test_df_mat[~np.isnan(test_df_mat)] == test_indices_mat[~np.isnan(test_df_mat)]), \
                                "The CV iterator do not slice the test dataset exactly equal as the CV indices iterator."

        assert np.setdiff1d(nptrain, nptest).shape == nptrain.shape, "The train indices includes test indices"
        assert np.setdiff1d(nptest, nptrain).shape == nptest.shape, "The test indices includes train indices"
        assert np.all(np.sort(np.setdiff1d(train_indices, test_indices)) == np.sort(train_indices)), "The train indices includes test indices"
        assert np.all(np.sort(np.setdiff1d(test_indices, train_indices)) == np.sort(test_indices)), "The test indices includes train indices"