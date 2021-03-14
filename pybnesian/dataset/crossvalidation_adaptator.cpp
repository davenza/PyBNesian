#include <dataset/crossvalidation_adaptator.hpp>

namespace dataset {

std::pair<Array_ptr, Array_ptr> generate_cv_pair_column(Array_ptr col,
                                                        bool include_null,
                                                        const_vecit<int> begin,
                                                        const_vecit<int> end,
                                                        const_vecit<int> test_begin,
                                                        const_vecit<int> test_end) {
    switch (col->type_id()) {
        case Type::DOUBLE:
            return split_array_train_test<arrow::DoubleType>(col, include_null, begin, end, test_begin, test_end);
        case Type::FLOAT:
            return split_array_train_test<arrow::FloatType>(col, include_null, begin, end, test_begin, test_end);
        default:
            throw std::invalid_argument("Wrong data type in CrossValidation.");
    }
}

std::pair<DataFrame, DataFrame> CrossValidation::generate_cv_pair(int fold) const {
    Array_vector train_cols;
    train_cols.reserve(m_df->num_columns());
    Array_vector test_cols;
    test_cols.reserve(m_df->num_columns());

    for (auto col : m_df->columns()) {
        auto [train_col, test_col] = generate_cv_pair_column(col,
                                                             prop->include_null,
                                                             prop->indices.begin(),
                                                             prop->indices.end(),
                                                             prop->limits[fold],
                                                             prop->limits[fold + 1]);

        train_cols.push_back(train_col);
        test_cols.push_back(test_col);
    }

    int rows_train = std::distance(prop->indices.cbegin(), prop->limits[fold]) +
                     std::distance(prop->limits[fold + 1], prop->indices.cend());
    int rows_test = std::distance(prop->limits[fold], prop->limits[fold + 1]);

    auto rb_train = arrow::RecordBatch::Make(m_df->schema(), rows_train, train_cols);
    auto rb_test = arrow::RecordBatch::Make(m_df->schema(), rows_test, test_cols);

    return std::make_pair(rb_train, rb_test);
}

std::pair<std::vector<int>, std::vector<int>> CrossValidation::generate_cv_pair_indices(int fold) const {
    int train_size = std::distance(prop->indices.cbegin(), prop->limits[fold]) +
                     std::distance(prop->limits[fold + 1], prop->indices.cend());

    std::vector<int> train_indices(train_size);

    std::copy(prop->indices.cbegin(), prop->limits[fold], train_indices.begin());
    int offset = std::distance(prop->indices.cbegin(), prop->limits[fold]);
    std::copy(prop->limits[fold + 1], prop->indices.cend(), train_indices.begin() + offset);

    std::vector<int> test_indices(prop->limits[fold], prop->limits[fold + 1]);

    return std::make_pair(train_indices, test_indices);
}

}  // namespace dataset