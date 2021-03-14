#ifndef PYBNESIAN_UTIL_COMBINATIONS_HPP
#define PYBNESIAN_UTIL_COMBINATIONS_HPP

#include <vector>
#include <unordered_set>
#include <boost/math/special_functions/binomial.hpp>
#include <util/parameter_traits.hpp>

namespace util {

template <typename T>
class Combinations {
public:
    Combinations() = default;

    // TODO: Check m_k > fixed.size().
    template <typename Iter, util::enable_if_iterator_t<Iter, int> = 0>
    Combinations(Iter begin, Iter end, int k)
        : m_elements(begin, end),
          m_fixed(),
          m_k(k),
          m_num_combinations(
              std::round(boost::math::binomial_coefficient<double>(m_elements.size(), m_k - m_fixed.size()))) {}

    template <typename Iter,
              typename IterFixed,
              util::enable_if_iterator_t<Iter, int> = 0,
              util::enable_if_iterator_t<IterFixed, int> = 0>
    Combinations(Iter begin, Iter end, IterFixed begin_fixed, IterFixed end_fixed, int k)
        : m_elements(begin, end),
          m_fixed(begin_fixed, end_fixed),
          m_k(k),
          m_num_combinations(
              std::round(boost::math::binomial_coefficient<double>(m_elements.size(), m_k - m_fixed.size()))) {
        static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type,
                                     typename std::iterator_traits<IterFixed>::value_type>,
                      "The type of fixed and movable elements should be the same.");
    }

    template <typename V, util::enable_if_vector_of_type_t<V, T, int> = 0>
    Combinations(V elements, int k)
        : m_elements(elements),
          m_fixed(),
          m_k(k),
          m_num_combinations(
              std::round(boost::math::binomial_coefficient<double>(m_elements.size(), m_k - m_fixed.size()))) {}

    template <typename V,
              typename V2,
              util::enable_if_vector_of_type_t<V, T, int> = 0,
              util::enable_if_vector_of_type_t<V2, T, int> = 0>
    Combinations(V elements, V2 fixed, int k)
        : m_elements(elements),
          m_fixed(fixed),
          m_k(k),
          m_num_combinations(
              std::round(boost::math::binomial_coefficient<double>(m_elements.size(), m_k - m_fixed.size()))) {}

    class combination_iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::vector<T>;
        using difference_type = int;
        using pointer = std::vector<T>*;
        using reference = std::vector<T>&;

        combination_iterator() = default;

        combination_iterator(const Combinations<T>* self, int idx) : m_self(self), m_subset(), m_indices(), m_idx(idx) {
            m_subset.reserve(m_self->m_k);

            for (size_t i = 0; i < m_self->m_fixed.size(); ++i) {
                m_subset.push_back(m_self->m_fixed[i]);
            }

            auto p = m_self->m_k - m_self->m_fixed.size();
            m_indices.reserve(p);

            for (size_t i = 0; i < p; ++i) {
                m_subset.push_back(m_self->m_elements[i]);
                m_indices.push_back(i);
            }
        }

        combination_iterator(const combination_iterator& other)
            : m_self(other.m_self), m_subset(other.m_subset), m_indices(other.m_indices), m_idx(other.m_idx) {}

        combination_iterator& operator=(const combination_iterator& other) {
            m_self = other.m_self;
            m_subset = other.m_subset;
            m_indices = other.m_indices;
            m_idx = other.m_idx;
            return *this;
        }

        void next_subset() {
            int offset = m_self->m_fixed.size();
            int p = m_self->m_k - offset;

            for (int i = p - 1; i >= 0; --i) {
                auto k = i + offset;
                auto max_index = m_self->m_elements.size() - p + i;

                if (m_indices[i] < max_index) {
                    ++m_indices[i];
                    m_subset[k] = m_self->m_elements[m_indices[i]];

                    for (int j = i + 1; j < p; ++j) {
                        m_indices[j] = m_indices[j - 1] + 1;
                        m_subset[j + offset] = m_self->m_elements[m_indices[j]];
                    }

                    break;
                }
            }
        }

        combination_iterator& operator++() {
            ++m_idx;
            next_subset();
            return *this;
        }

        combination_iterator operator++(int) {
            combination_iterator return_it(*this);
            ++m_idx;
            next_subset();
            return return_it;
        }

        reference operator*() { return m_subset; }

        pointer operator->() { return &m_subset; }

        bool operator==(const combination_iterator& rhs) { return (m_idx == rhs.m_idx) && (m_self == rhs.m_self); }

        bool operator!=(const combination_iterator& rhs) { return !(*this == rhs); }

    private:
        const Combinations<T>* m_self;
        std::vector<T> m_subset;
        std::vector<size_t> m_indices;
        int m_idx;
    };

    combination_iterator begin() const { return combination_iterator(this, 0); }
    combination_iterator end() const { return combination_iterator(this, num_combinations()); }

    int num_combinations() const { return m_num_combinations; }

private:
    std::vector<T> m_elements;
    std::vector<T> m_fixed;
    int m_k;
    int m_num_combinations;
};

template <typename Iter>
Combinations(Iter, Iter, int) -> Combinations<typename std::iterator_traits<Iter>::value_type>;
template <typename Iter, typename IterFixed>
Combinations(Iter, Iter, IterFixed, IterFixed, int) -> Combinations<typename std::iterator_traits<Iter>::value_type>;
template <typename V>
Combinations(V, int) -> Combinations<typename V::value_type>;
template <typename V, typename V2>
Combinations(V, V2, int) -> Combinations<typename V::value_type>;

template <typename T>
class Combinations2Sets {
public:
    template <typename Iter,
              typename Iter2,
              util::enable_if_iterator_t<Iter, int> = 0,
              util::enable_if_iterator_t<Iter2, int> = 0>
    Combinations2Sets(Iter begin_set1, Iter end_set1, Iter2 begin_set2, Iter2 end_set2, int k)
        : Combinations2Sets(std::vector(begin_set1, end_set1), std::vector(begin_set2, end_set2), k) {
        static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type,
                                     typename std::iterator_traits<Iter2>::value_type>,
                      "The elements of both sets should be of the same type");
    }

    template <typename V,
              typename V2,
              util::enable_if_vector_of_type_t<V, T, int> = 0,
              util::enable_if_vector_of_type_t<V2, T, int> = 0>
    Combinations2Sets(V v1, V2 v2, int k)
        : m_comb1(), m_comb2(), m_comb2_valid_combinations(), m_num_combinations(-1), m_k(k) {
        std::sort(v1.begin(), v1.end());
        std::sort(v2.begin(), v2.end());

        std::unordered_set<T> common_elements;
        std::set_intersection(
            v1.begin(), v1.end(), v2.begin(), v2.end(), std::inserter(common_elements, common_elements.end()));

        m_comb1 = Combinations<T>(std::move(v1), m_k);
        if (static_cast<int>(common_elements.size()) < k) {
            m_comb2 = Combinations<T>(std::move(v2), m_k);
            m_comb2_valid_combinations = m_comb2.num_combinations();
        } else {
            for (size_t i = 0, common_start = v2.size() - common_elements.size(); i < common_start; ++i) {
                if (common_elements.count(v2[i]) > 0) {
                    for (size_t j = v2.size() - 1; j >= common_start; --j) {
                        if (common_elements.count(v2[j]) == 0) {
                            std::swap(v2[i], v2[j]);
                        }
                    }
                }
            }

            m_comb2 = Combinations<T>(std::move(v2), m_k);
            m_comb2_valid_combinations =
                m_comb2.num_combinations() -
                std::round(boost::math::binomial_coefficient<double>(common_elements.size(), m_k));
        }

        m_num_combinations = m_comb1.num_combinations() + m_comb2_valid_combinations;
    }

    class combination2set_iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::vector<T>;
        using difference_type = int;
        using pointer = std::vector<T>*;
        using reference = std::vector<T>&;

        combination2set_iterator(const Combinations2Sets<T>& self, int idx) : m_self(self), it() {
            if (idx < m_self.m_comb1.num_combinations() || m_self.m_comb2_valid_combinations == 0) {
                it = typename Combinations<T>::combination_iterator(&m_self.m_comb1, idx);
            } else {
                it = typename Combinations<T>::combination_iterator(&m_self.m_comb2,
                                                                    idx - m_self.m_comb1.num_combinations());
            }
        }

        combination2set_iterator& operator++() {
            ++it;
            if (it == m_self.m_comb1.end() && m_self.m_comb2_valid_combinations > 0) {
                it = m_self.m_comb2.begin();
            }
            return *this;
        }

        combination2set_iterator operator++(int) {
            combination2set_iterator return_it(*this);
            ++it;
            if (it == m_self.m_comb1.end() && m_self.m_comb2_valid_combinations > 0) {
                it = m_self.m_comb2.begin();
            }
            return return_it;
        }

        reference operator*() { return *it; }

        pointer operator->() { return &it.m_subset; }

        bool operator==(const combination2set_iterator& rhs) { return it == rhs.it; }

        bool operator!=(const combination2set_iterator& rhs) { return !(*this == rhs); }

    private:
        const Combinations2Sets<T>& m_self;
        typename Combinations<T>::combination_iterator it;
    };

    combination2set_iterator begin() { return combination2set_iterator(*this, 0); }
    combination2set_iterator end() { return combination2set_iterator(*this, num_combinations()); }

    int num_combinations() const { return m_num_combinations; }

private:
    Combinations<T> m_comb1;
    Combinations<T> m_comb2;
    int m_comb2_valid_combinations;
    int m_num_combinations;
    int m_k;
};

template <typename Iter, typename Iter2>
Combinations2Sets(Iter, Iter, Iter2, Iter2, int) -> Combinations2Sets<typename std::iterator_traits<Iter>::value_type>;
template <typename V, typename V2>
Combinations2Sets(V, V2, int) -> Combinations2Sets<typename V::value_type>;

template <typename T>
class AllSubsets {
public:
    AllSubsets() = default;

    template <typename Iter, util::enable_if_iterator_t<Iter, int> = 0>
    AllSubsets(Iter begin, Iter end, int min_k, int max_k)
        : m_elements(begin, end), m_fixed(), m_min_k(min_k), m_max_k(max_k), m_num_combinations(0) {
        for (int i = min_k; i <= max_k; ++i) {
            m_num_combinations += std::round(boost::math::binomial_coefficient<double>(m_elements.size(), i));
        }
    }

    template <typename Iter,
              typename IterFixed,
              util::enable_if_iterator_t<Iter, int> = 0,
              util::enable_if_iterator_t<IterFixed, int> = 0>
    AllSubsets(Iter begin, Iter end, IterFixed begin_fixed, IterFixed end_fixed, int min_k, int max_k)
        : m_elements(begin, end),
          m_fixed(begin_fixed, end_fixed),
          m_min_k(min_k),
          m_max_k(max_k),
          m_num_combinations(0) {
        static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type,
                                     typename std::iterator_traits<IterFixed>::value_type>,
                      "The type of fixed and movable elements should be the same.");
        for (int i = min_k; i <= max_k; ++i) {
            m_num_combinations +=
                std::round(boost::math::binomial_coefficient<double>(m_elements.size(), i - m_fixed.size()));
        }
    }

    template <typename V, util::enable_if_vector_of_type_t<V, T, int> = 0>
    AllSubsets(V elements, int min_k, int max_k)
        : m_elements(elements), m_min_k(min_k), m_max_k(max_k), m_num_combinations(0) {
        for (int i = min_k; i <= max_k; ++i) {
            m_num_combinations += std::round(boost::math::binomial_coefficient<double>(m_elements.size(), i));
        }
    }

    template <typename V,
              typename V2,
              util::enable_if_vector_of_type_t<V, T, int> = 0,
              util::enable_if_vector_of_type_t<V2, T, int> = 0>
    AllSubsets(V elements, V2 fixed, int min_k, int max_k)
        : m_elements(elements), m_fixed(fixed), m_min_k(min_k), m_max_k(max_k), m_num_combinations(0) {
        for (int i = min_k; i <= max_k; ++i) {
            m_num_combinations +=
                std::round(boost::math::binomial_coefficient<double>(m_elements.size(), i - m_fixed.size()));
        }
    }

    class allsubsets_iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::vector<T>;
        using difference_type = int;
        using pointer = std::vector<T>*;
        using reference = std::vector<T>&;

        allsubsets_iterator() = default;

        allsubsets_iterator(const AllSubsets<T>* self, int idx)
            : m_self(self), m_idx(idx), m_current_comb(), m_current_iter(), m_current_k(m_self->m_min_k) {
            if (idx == 0) {
                m_current_comb = Combinations(m_self->m_elements.begin(),
                                              m_self->m_elements.end(),
                                              m_self->m_fixed.begin(),
                                              m_self->m_fixed.end(),
                                              m_current_k);
                m_current_iter = m_current_comb.begin();
            }
        }

        allsubsets_iterator& operator=(const allsubsets_iterator& other) {
            m_self = other.m_self;
            m_idx = other.m_idx;
            m_current_comb = other.m_current_comb;
            m_current_iter = other.m_current_iter;
            m_current_k = other.m_current_k;
            return *this;
        }

        allsubsets_iterator& operator++() {
            ++m_current_iter;
            ++m_idx;

            if (m_current_iter == m_current_comb.end() && m_current_k < m_self->m_max_k) {
                ++m_current_k;
                m_current_comb = Combinations(m_self->m_elements.begin(),
                                              m_self->m_elements.end(),
                                              m_self->m_fixed.begin(),
                                              m_self->m_fixed.end(),
                                              m_current_k);
                m_current_iter = m_current_comb.begin();
            }

            return *this;
        }

        allsubsets_iterator operator++(int) {
            allsubsets_iterator return_it(*this);

            ++m_current_iter;
            ++m_idx;
            if (m_current_iter == m_current_comb.end() && m_current_k < m_self->m_max_k) {
                ++m_current_k;
                m_current_comb = Combinations(m_self->m_elements.begin(),
                                              m_self->m_elements.end(),
                                              m_self->m_fixed.begin(),
                                              m_self->m_fixed.end(),
                                              m_current_k);
                m_current_iter = m_current_comb.begin();
            }

            return return_it;
        }

        reference operator*() { return *m_current_iter; }

        pointer operator->() { return &m_current_iter.m_subset; }

        bool operator==(const allsubsets_iterator& rhs) { return (m_idx == rhs.m_idx) && (m_self == rhs.m_self); }

        bool operator!=(const allsubsets_iterator& rhs) { return !(*this == rhs); }

    private:
        const AllSubsets<T>* m_self;
        int m_idx;
        Combinations<T> m_current_comb;
        typename Combinations<T>::combination_iterator m_current_iter;
        int m_current_k;
    };

    allsubsets_iterator begin() const { return allsubsets_iterator(this, 0); }
    allsubsets_iterator end() const { return allsubsets_iterator(this, num_combinations()); }

    int num_combinations() const { return m_num_combinations; }

private:
    std::vector<T> m_elements;
    std::vector<T> m_fixed;
    int m_min_k;
    int m_max_k;
    int m_num_combinations;
};

template <typename Iter>
AllSubsets(Iter, Iter, int, int) -> AllSubsets<typename std::iterator_traits<Iter>::value_type>;
template <typename Iter, typename IterFixed>
AllSubsets(Iter, Iter, IterFixed, IterFixed, int, int) -> AllSubsets<typename std::iterator_traits<Iter>::value_type>;
template <typename V>
AllSubsets(V, int, int) -> AllSubsets<typename V::value_type>;
template <typename V, typename V2>
AllSubsets(V, V2, int, int) -> AllSubsets<typename V::value_type>;

}  // namespace util

#endif  // PYBNESIAN_UTIL_COMBINATIONS_HPP
