#ifndef PGM_DATASET_CROSSVALIDATION_ADAPTATOR_HPP
#define PGM_DATASET_CROSSVALIDATION_ADAPTATOR_HPP

#include <dataset/dataset.hpp>
#include <random>
#include <arrow/api.h>
#include <util/bit_util.hpp>

using Eigen::MatrixXd;

using Array_ptr = std::shared_ptr<arrow::Array>;
using Array_vector =  std::vector<Array_ptr>;

using arrow::NumericBuilder;

template<typename T>
using c_vecit = typename std::vector<T>::const_iterator;

namespace dataset {

    template<typename ArrowType>
    Array_ptr split_test(Array_ptr col, 
                         c_vecit<int> test_begin, 
                         c_vecit<int> test_end) 
    {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        int rows_test = std::distance(test_begin, test_end);

        NumericBuilder<ArrowType> builder;
        builder.Resize(rows_test);

        auto dwn_col = std::static_pointer_cast<ArrayType>(col);
        auto raw_values = dwn_col->raw_values();
        for(auto it = test_begin; it != test_end; it++) {
            builder.UnsafeAppend(raw_values[*it]);
        }

        std::shared_ptr<arrow::Array> out;
        builder.Finish(&out);
        return out;
    }

    template<typename ArrowType>
    Array_ptr split_train(Array_ptr col, 
                          c_vecit<int> begin, 
                          c_vecit<int> end, 
                          c_vecit<int> test_begin, 
                          c_vecit<int> test_end) 
    {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        int rows_train = std::distance(begin, test_begin) + std::distance(test_end, end);

        NumericBuilder<ArrowType> builder;
        builder.Resize(rows_train);

        auto dwn_col = std::static_pointer_cast<ArrayType>(col);
        auto raw_values = dwn_col->raw_values();
        for(auto it = begin; it != test_begin; it++) {
            builder.UnsafeAppend(raw_values[*it]);
        }

        for(auto it = test_end; it != end; it++) {
            builder.UnsafeAppend(raw_values[*it]);
        }

        std::shared_ptr<arrow::Array> out;
        builder.Finish(&out);
        return out;
    }

    template<typename ArrowType>
    std::pair<Array_ptr, Array_ptr> split_array_train_test(Array_ptr col, 
                                                           c_vecit<int> begin, 
                                                           c_vecit<int> end, 
                                                           c_vecit<int> test_begin, 
                                                           c_vecit<int> test_end) 
    {
        auto df_train = split_train<ArrowType>(col, begin, end, test_begin, test_end);
        auto df_test = split_test<ArrowType>(col, test_begin, test_end);
        return std::make_pair(df_train, df_test);
    }

    std::pair<Array_ptr, Array_ptr> generate_cv_pair_column(Array_ptr col, 
                                                            c_vecit<int> begin, 
                                                            c_vecit<int> end, 
                                                            c_vecit<int> test_begin, 
                                                            c_vecit<int> test_end) 
    {
        switch (col->type_id()) {
            case Type::DOUBLE:
                return split_array_train_test<arrow::DoubleType>(col, begin, end, test_begin, test_end);
            case Type::FLOAT:
                return split_array_train_test<arrow::FloatType>(col, begin, end, test_begin, test_end);
            default:
                throw std::invalid_argument("Wrong data type in CrossValidation.");
        }
    }

    std::pair<DataFrame, DataFrame> generate_cv_pair(const DataFrame& df, 
                                                     int fold, 
                                                     const std::vector<int>& indices, 
                                                     const std::vector<c_vecit<int>>& test_limits) {
        

        Array_vector train_cols;
        train_cols.reserve(df->num_columns());
        Array_vector test_cols;
        test_cols.reserve(df->num_columns());

        for (auto col : df->columns()) {
            auto [train_col, test_col] = generate_cv_pair_column(col, indices.begin(), indices.end(), test_limits[fold], test_limits[fold+1]);

            train_cols.push_back(train_col);
            test_cols.push_back(test_col);
        }

        
        int rows_train = std::distance(indices.begin(), test_limits[fold]) + std::distance(test_limits[fold+1], indices.end());
        int rows_test = std::distance(test_limits[fold], test_limits[fold+1]);

        auto rb_train = arrow::RecordBatch::Make(df->schema(), rows_train, train_cols);
        auto rb_test = arrow::RecordBatch::Make(df->schema(), rows_test, test_cols);

        return std::make_pair(rb_train, rb_test);
    }

    class CrossValidation {
    public:

        CrossValidation(const DataFrame& df, int k) : CrossValidation(df, k, std::random_device{}(), false) {}
        CrossValidation(const DataFrame& df, int k, bool include_null) : CrossValidation(df, k, std::random_device{}(), include_null) {}
        CrossValidation(const DataFrame& df, int k, int seed) : CrossValidation(df, k, seed, false) {}

        CrossValidation(const DataFrame& df, int k, int seed, bool include_null) : df(df), k(k), indices(), limits() {
            
            
            if (k > df->num_rows()) {
                throw std::invalid_argument("Cannot split " + std::to_string(df->num_rows()) + " intances in " + std::to_string(k) + " folds.");
            }

            if (df.null_count() == 0 || include_null) {
                indices.resize(df->num_rows());
                std::iota(indices.begin(), indices.end(), 0);
            } else {
                auto combined_bitmap = df.combined_bitmap();
                auto bitmap_data = combined_bitmap->data();
                
                int total_rows = df->num_rows();
                int valid_rows = util::bit_util::non_null_count(combined_bitmap, total_rows);
                indices.reserve(df->num_rows());

                for (auto i = 0; i < total_rows; ++i) {
                    if (arrow::BitUtil::GetBit(bitmap_data, i))
                        indices.push_back(i);
                }
            }

            auto rng = std::default_random_engine {seed};
            std::shuffle(indices.begin(), indices.end(), rng);

            int fold_size = df->num_rows() / k;
            int folds_extra = df->num_rows() % k;
            
            limits.reserve(k+1);
            limits.push_back(indices.begin());

            auto curr_iter = indices.begin();
            for (int i = 0; i < folds_extra; ++i) {
                curr_iter += fold_size + 1;
                limits.push_back(curr_iter);
            }

            for (int i = folds_extra; i < k; ++i) {
                curr_iter += fold_size;
                limits.push_back(curr_iter);
            }

        }

        class cv_iterator {
            public:
                using difference_type = std::iterator_traits<std::vector<int>::iterator>;
                using value_type = std::pair<DataFrame, DataFrame>;
                using reference = value_type&;
                using pointer = value_type*;
                using iterator_category = std::random_access_iterator_tag; //or another tag

                cv_iterator(int i, const CrossValidation& cv) : i(i), cv(cv), updated_fold(false), current_fold() {}

                reference operator*() { 
                    if (!updated_fold) {
                        update_fold();
                        updated_fold = true;
                    }

                    return current_fold;
                }

                cv_iterator& operator++() { ++i; updated_fold = false; return *this; }
                cv_iterator operator++(int) { ++i; updated_fold = false; return *this; }
                bool operator==(cv_iterator& rhs) const { return (i == rhs.i) && (cv.k == rhs.cv.k) && (&cv.df == &rhs.cv.df) && 
                                                              (&cv.indices == &rhs.cv.indices); }

                bool operator!=(cv_iterator& rhs) const { return (i != rhs.i) || (cv.k != rhs.cv.k) || (&cv.df != &rhs.cv.df) || 
                                                                 (&cv.indices != &rhs.cv.indices); }


            private:

                void update_fold();
                int i;
                const CrossValidation& cv;
                bool updated_fold;
                std::pair<DataFrame, DataFrame> current_fold;
        };
    
    cv_iterator begin() {
        return cv_iterator(0, *this);
    }

    cv_iterator end() {
        return cv_iterator(k, *this);
    }

    private:
        const DataFrame& df;
        int k;
        std::vector<int> indices;
        std::vector<std::vector<int>::iterator> limits;
    };
}

#endif //PGM_DATASET_CROSSVALIDATION_ADAPTATOR_HPP