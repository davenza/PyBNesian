#ifndef PYBNESIAN_DATASET_CROSSVALIDATION_ADAPTATOR_HPP
#define PYBNESIAN_DATASET_CROSSVALIDATION_ADAPTATOR_HPP

#include <random>
#include <dataset/dataset.hpp>

using Array_ptr = std::shared_ptr<arrow::Array>;
using arrow::NumericBuilder;

template <typename T>
using const_vecit = typename std::vector<T>::const_iterator;

namespace dataset {

class CrossValidationProperties {
public:
    CrossValidationProperties(const DataFrame& df, int k, unsigned int seed, bool include_null)
        : k(k), m_seed(seed), indices(), limits() {
        if (k <= 1 || k > df->num_rows()) {
            throw std::invalid_argument("Cannot split " + std::to_string(df->num_rows()) + " instances into " +
                                        std::to_string(k) + " folds.");
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
                if (util::bit_util::GetBit(bitmap_data, i)) indices.push_back(i);
            }
        }

        std::mt19937 rng{m_seed};
        std::shuffle(indices.begin(), indices.end(), rng);

        int fold_size = indices.size() / k;
        int folds_extra = indices.size() % k;

        limits.reserve(k + 1);
        limits.push_back(0);

        auto curr_iter = 0;
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
    std::vector<int> limits;
};

class CrossValidation {
public:
    CrossValidation(const DataFrame df,
                    int k = 10,
                    unsigned int seed = std::random_device{}(),
                    bool include_null = false)
        : m_df(df), prop(std::make_shared<CrossValidationProperties>(m_df, k, seed, include_null)) {}

    class cv_iterator {
    public:
        using difference_type = std::iterator_traits<std::vector<int>::iterator>;
        using value_type = std::pair<DataFrame, DataFrame>;
        using reference = value_type&;
        using pointer = value_type*;
        // FIXME: Check the iterator category operations.
        using iterator_category = std::forward_iterator_tag;  // or another tag

        cv_iterator(int i, const CrossValidation& cv) : i(i), cv(cv), updated_fold(false), current_fold() {}

        reference operator*() const {
            if (!updated_fold) {
                current_fold = cv.generate_cv_pair(i);
                updated_fold = true;
            }
            return current_fold;
        }

        cv_iterator& operator++() {
            ++i;
            updated_fold = false;
            return *this;
        }
        cv_iterator operator++(int) {
            ++i;
            updated_fold = false;
            return *this;
        }
        // TODO: Improve equality.
        bool operator==(const cv_iterator& rhs) const {
            return (i == rhs.i) && (cv.prop->k == rhs.cv.prop->k) && (cv.prop->m_seed == rhs.cv.prop->m_seed) &&
                   (&cv.m_df == &rhs.cv.m_df);
        }

        bool operator!=(const cv_iterator& rhs) const { return !(*this == rhs); }

    private:
        int i;
        const CrossValidation& cv;
        mutable bool updated_fold;
        mutable std::pair<DataFrame, DataFrame> current_fold;
    };

    cv_iterator begin() { return cv_iterator(0, *this); }

    cv_iterator end() { return cv_iterator(prop->k, *this); }

    std::pair<DataFrame, DataFrame> fold(int fold) { return generate_cv_pair(fold); }

    const DataFrame& data() const { return m_df; }

    template <typename T, util::enable_if_index_container_t<T, int> = 0>
    CrossValidation loc(const T& cols) const {
        return CrossValidation(m_df.loc(cols), prop);
    }
    template <typename V>
    CrossValidation loc(const std::initializer_list<V>& cols) const {
        return loc<std::initializer_list<V>>(cols);
    }
    CrossValidation loc(int i) const { return CrossValidation(m_df.loc(i), prop); }
    template <typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    CrossValidation loc(const StringType& name) const {
        return CrossValidation(m_df.loc(name), prop);
    }
    template <typename... Args>
    CrossValidation loc(const Args&... args) const {
        return CrossValidation(m_df.loc(args...), prop);
    }

    class cv_iterator_indices {
    public:
        using difference_type = std::iterator_traits<std::vector<int>::iterator>;
        using value_type = std::pair<std::vector<int>, std::vector<int>>;
        using reference = value_type&;
        using pointer = value_type*;
        // FIXME: Check the iterator category operations.
        using iterator_category = std::forward_iterator_tag;  // or another tag

        cv_iterator_indices(int i, const CrossValidation& cv) : i(i), cv(cv), updated_fold(false), current_fold() {}

        reference operator*() const {
            if (!updated_fold) {
                current_fold = cv.generate_cv_pair_indices(i);
                updated_fold = true;
            }
            return current_fold;
        }

        cv_iterator_indices& operator++() {
            ++i;
            updated_fold = false;
            return *this;
        }
        cv_iterator_indices operator++(int) {
            ++i;
            updated_fold = false;
            return *this;
        }
        // TODO: Improve equality.
        bool operator==(const cv_iterator_indices& rhs) const {
            return (i == rhs.i) && (cv.prop->k == rhs.cv.prop->k) && (cv.prop->m_seed == rhs.cv.prop->m_seed) &&
                   (&cv.m_df == &rhs.cv.m_df);
        }

        bool operator!=(const cv_iterator_indices& rhs) const { return !(*this == rhs); }

    private:
        int i;
        const CrossValidation& cv;
        mutable bool updated_fold;
        mutable std::pair<std::vector<int>, std::vector<int>> current_fold;
    };

    cv_iterator_indices begin_indices() { return cv_iterator_indices(0, *this); }

    cv_iterator_indices end_indices() { return cv_iterator_indices(prop->k, *this); }

private:
    CrossValidation(const DataFrame df, const std::shared_ptr<CrossValidationProperties> prop) : m_df(df), prop(prop) {}
    std::pair<DataFrame, DataFrame> generate_cv_pair(int fold) const;
    std::pair<std::vector<int>, std::vector<int>> generate_cv_pair_indices(int fold) const;

    const DataFrame m_df;
    const std::shared_ptr<CrossValidationProperties> prop;
};

}  // namespace dataset

#endif  // PYBNESIAN_DATASET_CROSSVALIDATION_ADAPTATOR_HPP