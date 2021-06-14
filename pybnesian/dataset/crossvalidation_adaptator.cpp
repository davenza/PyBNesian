#include <dataset/crossvalidation_adaptator.hpp>

namespace dataset {

std::pair<DataFrame, DataFrame> CrossValidation::generate_cv_pair(int fold) const {
    arrow::NumericBuilder<arrow::Int32Type> builder;

    auto test_fold_start = prop->limits[fold];
    auto test_fold_end = prop->limits[fold + 1];
    auto test_fold_size = test_fold_end - test_fold_start;

    auto right_train_size = prop->limits.back() - test_fold_end;

    RAISE_STATUS_ERROR(builder.Reserve(prop->limits.back() - test_fold_size));
    if (test_fold_start > 0) {
        RAISE_STATUS_ERROR(builder.AppendValues(prop->indices.data(), test_fold_start));
    }

    if (right_train_size > 0) {
        RAISE_STATUS_ERROR(builder.AppendValues(prop->indices.data() + test_fold_end, right_train_size));
    }

    Array_ptr arrow_indices;
    RAISE_STATUS_ERROR(builder.Finish(&arrow_indices));
    auto train_df = m_df.take(arrow_indices);

    RAISE_STATUS_ERROR(builder.AppendValues(prop->indices.data() + test_fold_start, test_fold_size));
    RAISE_STATUS_ERROR(builder.Finish(&arrow_indices));

    return std::make_pair(std::move(train_df), m_df.take(arrow_indices));
}

std::pair<std::vector<int>, std::vector<int>> CrossValidation::generate_cv_pair_indices(int fold) const {
    auto test_fold_start = prop->limits[fold];
    auto test_fold_end = prop->limits[fold + 1];

    int test_size = test_fold_end - test_fold_start;
    int train_size = prop->limits.back() - test_size;

    std::vector<int> train_indices(train_size);

    std::copy(prop->indices.cbegin(), prop->indices.cbegin() + test_fold_start, train_indices.begin());
    std::copy(prop->indices.cbegin() + test_fold_end, prop->indices.cend(), train_indices.begin() + test_fold_start);

    std::vector<int> test_indices(test_size);
    std::copy(prop->indices.cbegin() + test_fold_start, prop->indices.cbegin() + test_fold_end, test_indices.begin());

    return std::make_pair(train_indices, test_indices);
}

}  // namespace dataset