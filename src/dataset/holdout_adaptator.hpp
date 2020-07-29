#ifndef PGM_DATASET_HOLDOUT_ADAPTATOR_HPP
#define PGM_DATASET_HOLDOUT_ADAPTATOR_HPP

#include <random>
#include <dataset/dataset.hpp>

using Array_ptr =  std::shared_ptr<arrow::Array>;
using Array_vector =  std::vector<Array_ptr>;
template<typename T>
using const_vecit = typename std::vector<T>::const_iterator;

using arrow::NumericBuilder;


namespace dataset {

    template<typename ArrowType>
    Array_ptr split(Array_ptr col,
                    const_vecit<int> begin,
                    const_vecit<int> end) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        int rows = std::distance(begin, end);

        NumericBuilder<ArrowType> builder;
        auto status = builder.Resize(rows);
        
        if (!status.ok()) {
            throw std::runtime_error("New array could not be created. Error status: " + status.ToString());
        }

        auto dwn_col = std::static_pointer_cast<ArrayType>(col);
        auto raw_values = dwn_col->raw_values();
        for(auto it = begin; it != end; ++it) {
            builder.UnsafeAppend(raw_values[*it]);
        }

        std::shared_ptr<arrow::Array> out;
        status = builder.Finish(&out);
        
        if (!status.ok()) {
            throw std::runtime_error("New array could not be created. Error status: " + status.ToString());
        }
        
        return out;
    }

    template<typename ArrowType>
    Array_ptr split_null(Array_ptr col, 
                         const_vecit<int> begin,
                         const_vecit<int> end) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        int rows = std::distance(begin, end);

        NumericBuilder<ArrowType> builder;
        auto status = builder.Resize(rows);
        
        if (!status.ok()) {
            throw std::runtime_error("New array could not be created. Error status: " + status.ToString());
        }

        auto dwn_col = std::static_pointer_cast<ArrayType>(col);
        auto raw_values = dwn_col->raw_values();
        auto bitmap = col->null_bitmap();
        auto bitmap_data = bitmap->data();
        for(auto it = begin; it != end; ++it) {
            if (arrow::BitUtil::GetBit(bitmap_data, *it))
                builder.UnsafeAppend(raw_values[*it]);
            else
                builder.UnsafeAppendNull();
        }

        std::shared_ptr<arrow::Array> out;
        status = builder.Finish(&out);

        if (!status.ok()) {
            throw std::runtime_error("New array could not be created. Error status: " + status.ToString());
        }

        return out;
    }

    template<typename ArrowType>
    std::pair<Array_ptr, Array_ptr> split_array_train_test(Array_ptr col,
                                                           bool include_null,
                                                           const_vecit<int> train_begin,
                                                           const_vecit<int> train_end,
                                                           const_vecit<int> test_end) {
    
        if (include_null && col->null_count() > 0) {
            auto df_train = split_null<ArrowType>(col, train_begin, train_end);
            auto df_test = split_null<ArrowType>(col, train_end, test_end);
            return std::make_pair(df_train, df_test);
        } else {
            auto df_train = split<ArrowType>(col, train_begin, train_end);
            auto df_test = split<ArrowType>(col, train_end, test_end);
            return std::make_pair(df_train, df_test);
        }
    }

    std::pair<Array_ptr, Array_ptr> generate_holdout_column(Array_ptr col,
                                                            bool include_null,
                                                            const_vecit<int> train_begin,
                                                            const_vecit<int> train_end,
                                                            const_vecit<int> test_end);

    std::pair<DataFrame, DataFrame> generate_holdout(const DataFrame& df,
                                                     bool include_null,
                                                     const_vecit<int> train_begin, 
                                                     const_vecit<int> train_end,
                                                     const_vecit<int> test_end);

    class HoldOut {
    public:
        HoldOut(const DataFrame& df, double test_ratio, bool include_null = false) : 
                                                    HoldOut(df, test_ratio, std::random_device{}(), include_null) { }
        HoldOut(const DataFrame& df, double test_ratio, int seed, bool include_null = false) :
                                                                m_seed(seed)
        {
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
                    if (arrow::BitUtil::GetBit(bitmap_data, i))
                        indices.push_back(i);
                }
            }

            std::mt19937 rng {m_seed};
            std::shuffle(indices.begin(), indices.end(), rng);

            int test_rows = std::round(indices.size()*test_ratio);
            int train_rows = indices.size() - test_rows;

            if (test_rows == 0 || train_rows == 0) {
                throw std::invalid_argument("Wrong test_ratio (" + std::to_string(test_ratio) + "selected for HoldOut.\n"
                                            "Generated train instances: " + std::to_string(train_rows) + "\n"
                                            "Generated test instances: " + std::to_string(test_rows));
            }

            auto split_point = indices.begin();
            std::advance(split_point, train_rows);
            std::tie(m_train_df, m_test_df) = generate_holdout(df, include_null, indices.begin(), split_point, indices.end());
        }

        const DataFrame& training_data() const { return m_train_df; }
        const DataFrame& test_data() const { return m_test_df; }

    private:
        DataFrame m_train_df;
        DataFrame m_test_df;
        int m_seed;
    };
}

#endif //PGM_DATASET_HOLDOUT_ADAPTATOR_HPP