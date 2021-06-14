#include <dataset/holdout_adaptator.hpp>

namespace dataset {

std::pair<DataFrame, DataFrame> generate_holdout(const DataFrame& df, const std::vector<int>& indices, int num_train) {
    arrow::NumericBuilder<arrow::Int32Type> builder;

    RAISE_STATUS_ERROR(builder.Reserve(num_train));
    RAISE_STATUS_ERROR(builder.AppendValues(indices.data(), num_train));
    Array_ptr arrow_indices;
    RAISE_STATUS_ERROR(builder.Finish(&arrow_indices));

    auto train_df = df.take(arrow_indices);

    auto num_test = indices.size() - num_train;
    RAISE_STATUS_ERROR(builder.Reserve(num_test));
    RAISE_STATUS_ERROR(builder.AppendValues(indices.data() + num_train, num_test));
    RAISE_STATUS_ERROR(builder.Finish(&arrow_indices));

    return std::make_pair(std::move(train_df), df.take(arrow_indices));
}

}  // namespace dataset