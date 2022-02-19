#ifndef PYBNESIAN_DATASET_HOLDOUT_ADAPTATOR_HPP
#define PYBNESIAN_DATASET_HOLDOUT_ADAPTATOR_HPP

#include <random>
#include <dataset/dataset.hpp>

using Array_ptr = std::shared_ptr<arrow::Array>;
template <typename T>
using const_vecit = typename std::vector<T>::const_iterator;

using arrow::NumericBuilder;

namespace dataset {

std::pair<DataFrame, DataFrame> generate_holdout(const DataFrame& df, const std::vector<int>& indices, int num_train);

class HoldOut {
public:
    HoldOut(DataFrame df,
            double test_ratio = 0.2,
            unsigned int seed = std::random_device{}(),
            bool include_null = false)
        : m_seed(seed) {
        if (test_ratio <= 0 || test_ratio >= 1.0) {
            throw std::invalid_argument("test_ratio must be a number between 0 and 1.");
        }

        std::vector<int> indices;
        if (df.null_count() == 0 || include_null) {
            indices.resize(df->num_rows());
            std::iota(indices.begin(), indices.end(), 0);
        } else {
            auto combined_bitmap = df.combined_bitmap();
            int total_rows = df->num_rows();
            int valid_rows = util::bit_util::non_null_count(combined_bitmap, total_rows);
            indices.reserve(valid_rows);

            auto bitmap_data = combined_bitmap->data();
            for (auto i = 0; i < total_rows; ++i) {
                if (util::bit_util::GetBit(bitmap_data, i)) indices.push_back(i);
            }
        }

        std::mt19937 rng{m_seed};
        std::shuffle(indices.begin(), indices.end(), rng);

        int test_rows = std::round(indices.size() * test_ratio);
        int train_rows = indices.size() - test_rows;

        if (test_rows == 0 || train_rows == 0) {
            throw std::invalid_argument("Wrong test_ratio (" + std::to_string(test_ratio) +
                                        "selected for HoldOut.\n"
                                        "Generated train instances: " +
                                        std::to_string(train_rows) +
                                        "\n"
                                        "Generated test instances: " +
                                        std::to_string(test_rows));
        }

        std::tie(m_train_df, m_test_df) = generate_holdout(df, indices, train_rows);
    }

    const DataFrame& training_data() const { return m_train_df; }
    const DataFrame& test_data() const { return m_test_df; }

private:
    DataFrame m_train_df;
    DataFrame m_test_df;
    unsigned int m_seed;
};

}  // namespace dataset

#endif  // PYBNESIAN_DATASET_HOLDOUT_ADAPTATOR_HPP