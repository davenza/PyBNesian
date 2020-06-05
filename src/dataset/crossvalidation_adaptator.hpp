#ifndef PGM_DATASET_CROSSVALIDATION_ADAPTATOR_HPP
#define PGM_DATASET_CROSSVALIDATION_ADAPTATOR_HPP

#include <dataset/dataset.hpp>
#include <random>
#include <arrow/api.h>
#include <util/bit_util.hpp>

using Eigen::MatrixXd;

using Array_ptr = std::shared_ptr<arrow::Array>;
using Array_vector =  std::vector<Array_ptr>;
using Buffer_ptr = std::shared_ptr<arrow::Buffer>;

using arrow::NumericBuilder;

template<typename T>
using c_vecit = typename std::vector<T>::const_iterator;

namespace dataset {


    template<typename ArrowType>
    Array_ptr split_train(Array_ptr col, 
                          c_vecit<int> begin, 
                          c_vecit<int> end,
                          c_vecit<int> test_begin,
                          c_vecit<int> test_end) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        int rows_train = std::distance(begin, test_begin) + std::distance(test_end, end);

        NumericBuilder<ArrowType> builder;
        builder.Resize(rows_train);

        auto dwn_col = std::static_pointer_cast<ArrayType>(col);
        auto raw_values = dwn_col->raw_values();
        for(auto it = begin; it != test_begin; ++it) {
            builder.UnsafeAppend(raw_values[*it]);
        }

        for(auto it = test_end; it != end; ++it) {
            builder.UnsafeAppend(raw_values[*it]);
        }

        std::shared_ptr<arrow::Array> out;
        builder.Finish(&out);
        return out;
    }


    template<typename ArrowType>
    Array_ptr split_train_null(Array_ptr col, 
                               c_vecit<int> begin, 
                               c_vecit<int> end,
                               c_vecit<int> test_begin,
                               c_vecit<int> test_end) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        int rows_train = std::distance(begin, test_begin) + std::distance(test_end, end);

        NumericBuilder<ArrowType> builder;
        builder.Resize(rows_train);

        auto dwn_col = std::static_pointer_cast<ArrayType>(col);
        auto raw_values = dwn_col->raw_values();
        auto bitmap = col->null_bitmap();
        for(auto it = begin; it != test_begin; ++it) {
            if (arrow::BitUtil::GetBit(bitmap->data(), *it))
                builder.UnsafeAppend(raw_values[*it]);
            else
                builder.UnsafeAppendNull();
        }

        for(auto it = test_end; it != end; ++it) {
            if (arrow::BitUtil::GetBit(bitmap->data(), *it))
                builder.UnsafeAppend(raw_values[*it]);
            else
                builder.UnsafeAppendNull();
        }

        std::shared_ptr<arrow::Array> out;
        builder.Finish(&out);
        return out;
    }

    template<typename ArrowType>
    Array_ptr split_test(Array_ptr col,
                         c_vecit<int> test_begin,
                         c_vecit<int> test_end) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        int rows_test = std::distance(test_begin, test_end);

        NumericBuilder<ArrowType> builder;
        builder.Resize(rows_test);

        auto dwn_col = std::static_pointer_cast<ArrayType>(col);
        auto raw_values = dwn_col->raw_values();
        for(auto it = test_begin; it != test_end; ++it) {
            builder.UnsafeAppend(raw_values[*it]);
        }

        std::shared_ptr<arrow::Array> out;
        builder.Finish(&out);
        return out;
    }

    template<typename ArrowType>
    Array_ptr split_test_null(Array_ptr col, 
                              c_vecit<int> test_begin,
                              c_vecit<int> test_end) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        int rows_test = std::distance(test_begin, test_end);

        NumericBuilder<ArrowType> builder;
        builder.Resize(rows_test);

        auto dwn_col = std::static_pointer_cast<ArrayType>(col);
        auto raw_values = dwn_col->raw_values();
        auto bitmap = col->null_bitmap();
        for(auto it = test_begin; it != test_end; ++it) {
            if (arrow::BitUtil::GetBit(bitmap->data(), *it))
                builder.UnsafeAppend(raw_values[*it]);
            else
                builder.UnsafeAppendNull();
        }

        std::shared_ptr<arrow::Array> out;
        builder.Finish(&out);
        return out;
    }

    template<typename ArrowType>
    std::pair<Array_ptr, Array_ptr> split_array_train_test(Array_ptr col,
                                                           bool include_null,
                                                           c_vecit<int> begin, 
                                                           c_vecit<int> end,
                                                           c_vecit<int> test_begin,
                                                           c_vecit<int> test_end) {
        if (include_null && col->null_count() > 0) {
            auto df_train = split_train_null<ArrowType>(col, begin, end, test_begin, test_end);
            auto df_test = split_test_null<ArrowType>(col, test_begin, test_end);
            return std::make_pair(df_train, df_test);
        } else {
            auto df_train = split_train<ArrowType>(col, begin, end, test_begin, test_end);
            auto df_test = split_test<ArrowType>(col, test_begin, test_end);
            return std::make_pair(df_train, df_test);
        }
    }

    std::pair<Array_ptr, Array_ptr> generate_cv_pair_column(Array_ptr col,
                                                            bool include_null,
                                                            c_vecit<int> begin, 
                                                            c_vecit<int> end,
                                                            c_vecit<int> test_begin,
                                                            c_vecit<int> test_end) {
        switch (col->type_id()) {
            case Type::DOUBLE:
                return split_array_train_test<arrow::DoubleType>(col, include_null, begin, end, test_begin, test_end);
            case Type::FLOAT:
                return split_array_train_test<arrow::FloatType>(col, include_null, begin, end, test_begin, test_end);
            default:
                throw std::invalid_argument("Wrong data type in CrossValidation.");
        }
    }


    class CrossValidationProperties {
    public:
        CrossValidationProperties(const DataFrame& df, int k, int seed, bool include_null) : k(k),
                                                                                             m_seed(seed),
                                                                                             indices(), 
                                                                                             limits(), 
                                                                                             include_null(include_null) {
            if (k > df->num_rows()) {
                throw std::invalid_argument("Cannot split " + std::to_string(df->num_rows()) + " instances into " + std::to_string(k) + " folds.");
            }

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

            auto rng = std::default_random_engine {m_seed};
            std::shuffle(indices.begin(), indices.end(), rng);

            int fold_size = df->num_rows() / k;
            int folds_extra = df->num_rows() % k;
            
            limits.reserve(k+1);
            limits.push_back(indices.begin());

            auto curr_iter = indices.cbegin();
            for (int i = 0; i < folds_extra; ++i) {
                curr_iter += fold_size + 1;
                limits.push_back(curr_iter);
            }

            for (int i = folds_extra; i < k; ++i) {
                curr_iter += fold_size;
                limits.push_back(curr_iter);
            }
        }

        friend class CrossValidation;
    private:
        int k;
        unsigned int m_seed;
        std::vector<int> indices;
        std::vector<c_vecit<int>> limits;
        bool include_null;
    };

    class CrossValidation {
    public:

        CrossValidation(const DataFrame df, int k) : CrossValidation(df, k, std::random_device{}(), false) {}
        CrossValidation(const DataFrame df, int k, bool include_null) : CrossValidation(df, k, std::random_device{}(), include_null) {}
        CrossValidation(const DataFrame df, int k, int seed) : CrossValidation(df, k, seed, false) {}

        CrossValidation(const DataFrame df, int k, int seed, bool include_null) : 
                                                    df(df), 
                                                    prop(std::make_shared<CrossValidationProperties>(df, k, seed, include_null)) { }
    
    class cv_iterator {
    public:
        using difference_type = std::iterator_traits<std::vector<int>::iterator>;
        using value_type = std::pair<DataFrame, DataFrame>;
        using reference = value_type&;
        using pointer = value_type*;
        // FIXME: Check the iterator category operations.
        using iterator_category = std::random_access_iterator_tag; //or another tag

        cv_iterator(int i, const CrossValidation& cv) : i(i), cv(cv), updated_fold(false), current_fold() {}

        reference operator*() const { 
            if (!updated_fold) {
                current_fold = cv.generate_cv_pair(i);
                updated_fold = true;
            }

            return current_fold;
        }

        cv_iterator& operator++() { ++i; updated_fold = false; return *this; }
        cv_iterator operator++(int) { ++i; updated_fold = false; return *this; }
        // TODO: Improve equality.
        bool operator==(const cv_iterator& rhs) const { return (i == rhs.i) && (cv.prop->k == rhs.cv.prop->k) && 
                                                            (cv.prop->m_seed == rhs.cv.prop->m_seed) && (&cv.df == &rhs.cv.df);
                                                        }

        bool operator!=(const cv_iterator& rhs) const { return !(*this == rhs); }

    private:
        int i;
        const CrossValidation& cv;
        mutable bool updated_fold;
        mutable std::pair<DataFrame, DataFrame> current_fold;
    };


    cv_iterator begin() {
        return cv_iterator(0, *this);
    }

    cv_iterator end() {
        return cv_iterator(prop->k, *this);
    }

    std::pair<DataFrame, DataFrame> fold(int fold) { return generate_cv_pair(fold); }

    const DataFrame& data() const { return df; }

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    CrossValidation loc(T cols) const { return CrossValidation(df.loc(cols), prop); }
    template<typename V>
    CrossValidation loc(std::initializer_list<V> cols) const { return loc<std::initializer_list<V>>(cols); }
    CrossValidation loc(int i) const { return CrossValidation(df.loc(i), prop); }
    template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    CrossValidation loc(StringType name) const { return CrossValidation(df.loc(name), prop); }
    template<typename ...Args>
    CrossValidation loc(Args... args) const  { return CrossValidation(df.loc(args...), prop); }

    private:
        CrossValidation(const DataFrame df, const std::shared_ptr<CrossValidationProperties> prop) : df(df), prop(prop) {}
        std::pair<DataFrame, DataFrame> generate_cv_pair(int fold) const;

        const DataFrame df;
        const std::shared_ptr<CrossValidationProperties> prop;
    };

    std::pair<DataFrame, DataFrame> CrossValidation::generate_cv_pair(int fold) const {
        Array_vector train_cols;
        train_cols.reserve(df->num_columns());
        Array_vector test_cols;
        test_cols.reserve(df->num_columns());

        for (auto col : df->columns()) {
            auto [train_col, test_col] = generate_cv_pair_column(col, prop->include_null, 
                                                                    prop->indices.begin(), prop->indices.end(), 
                                                                    prop->limits[fold], prop->limits[fold+1]);

            train_cols.push_back(train_col);
            test_cols.push_back(test_col);
        }

        int rows_train = std::distance(prop->indices.cbegin(), prop->limits[fold]) + std::distance(prop->limits[fold+1], prop->indices.cend());
        int rows_test = std::distance(prop->limits[fold], prop->limits[fold+1]);

        auto rb_train = arrow::RecordBatch::Make(df->schema(), rows_train, train_cols);
        auto rb_test = arrow::RecordBatch::Make(df->schema(), rows_test, test_cols);

        return std::make_pair(rb_train, rb_test);
    }
}

#endif //PGM_DATASET_CROSSVALIDATION_ADAPTATOR_HPP