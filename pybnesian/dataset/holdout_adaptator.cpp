#include <dataset/holdout_adaptator.hpp>

namespace dataset {

std::pair<Array_ptr, Array_ptr> generate_holdout_column(Array_ptr col,
                                                        bool include_null,
                                                        const_vecit<int> train_begin,
                                                        const_vecit<int> train_end,
                                                        const_vecit<int> test_end) {
    switch (col->type_id()) {
        case Type::DOUBLE:
            return split_array_train_test<arrow::DoubleType>(col, include_null, train_begin, train_end, test_end);
        case Type::FLOAT:
            return split_array_train_test<arrow::FloatType>(col, include_null, train_begin, train_end, test_end);
        default:
            throw std::invalid_argument("Wrong data type (" + col->type()->ToString() + ") in HoldOut.");
    }
}

std::pair<DataFrame, DataFrame> generate_holdout(const DataFrame& df,
                                                 bool include_null,
                                                 const_vecit<int> train_begin,
                                                 const_vecit<int> train_end,
                                                 const_vecit<int> test_end) {
    Array_vector train_cols;
    train_cols.reserve(df->num_columns());
    Array_vector test_cols;
    test_cols.reserve(df->num_columns());

    for (auto& col : df->columns()) {
        auto [train_col, test_col] = generate_holdout_column(col, include_null, train_begin, train_end, test_end);
        train_cols.push_back(train_col);
        test_cols.push_back(test_col);
    }

    int train_rows = std::distance(train_begin, train_end);
    int test_rows = std::distance(train_end, test_end);

    auto rb_train = arrow::RecordBatch::Make(df->schema(), train_rows, train_cols);
    auto rb_test = arrow::RecordBatch::Make(df->schema(), test_rows, test_cols);

    return std::make_pair(rb_train, rb_test);
}

}  // namespace dataset