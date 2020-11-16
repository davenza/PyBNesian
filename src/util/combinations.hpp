#ifndef PYBNESIAN_UTIL_COMBINATIONS_HPP
#define PYBNESIAN_UTIL_COMBINATIONS_HPP

#include <iterator>
#include <util/math_constants.hpp>
#include <boost/math/special_functions/binomial.hpp>

namespace util {

template<typename T>
class Combinations {
public:

    Combinations() = default;

    template<typename Iter>
    Combinations(Iter begin, Iter end, int k) : m_elements(begin, end), m_k(k), m_num_combinations(-1) {}

    // Combinations(std::vector<T> elements, int k) : m_elements(std::move(elements)), m_k(k), m_num_combinations(-1) {}
    // Combinations(const std::vector<T>& elements, int k) : m_elements(elements), m_k(k), m_num_combinations(-1) {}
    Combinations(std::vector<T>&& elements, int k) : m_elements(std::forward<std::vector<T>>(elements)), 
                                                        m_k(k), 
                                                        m_num_combinations(-1) {}

    class combination_iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::vector<T>;
        using difference_type = int;
        using pointer = std::vector<T>*;
        using reference = std::vector<T>&;

        combination_iterator() = default;

        combination_iterator(const Combinations<T>* self, int idx) : m_self(self), 
                                                                     m_subset(), 
                                                                     m_indices(), 
                                                                     m_idx(idx) {
            m_subset.reserve(m_self->m_k);
            m_indices.reserve(m_self->m_k);
            for (size_t i = 0, k = m_self->m_k; i < k; ++i) {
                m_subset.push_back(m_self->m_elements[i]);
                m_indices.push_back(i);
            }
        }

        combination_iterator(const combination_iterator& other) : m_self(other.m_self), 
                                                                 m_subset(other.m_subset),
                                                                 m_indices(other.m_indices),
                                                                 m_idx(other.m_idx) {}

        combination_iterator& operator=(const combination_iterator& other) {
            m_self = other.m_self;
            m_subset = other.m_subset;
            m_indices = other.m_indices;
            m_idx = other.m_idx;
            return *this;
        }

        void next_subset() {
            for (int i = m_subset.size()-1; i >= 0; --i) {
                auto max_index = m_self->m_elements.size() - m_subset.size() + i;
                if (m_indices[i] < max_index) {
                    ++m_indices[i];
                    m_subset[i] = m_self->m_elements[m_indices[i]];
                    
                    for (size_t j = i + 1, k = m_subset.size(); j < k; ++j) {
                        m_indices[j] = m_indices[j-1] + 1;
                        m_subset[j] = m_self->m_elements[m_indices[j]];
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

        reference operator*() {
            return m_subset;
        }
        
        pointer operator->() { return &m_subset; }

        bool operator==(const combination_iterator& rhs) {
            return (m_idx == rhs.m_idx) && (m_self == rhs.m_self);
        }

        bool operator!=(const combination_iterator& rhs) {
            return !(*this == rhs);
        }

    private:
        const Combinations<T>* m_self;
        std::vector<T> m_subset;
        std::vector<size_t> m_indices;
        int m_idx;
    };

    combination_iterator begin() const { return combination_iterator(this, 0); }
    combination_iterator end() const { return combination_iterator(this, num_combinations()); }
    
    int num_combinations() const {
        if (m_num_combinations != -1) {
            return m_num_combinations;
        } else {
            m_num_combinations = std::round(boost::math::binomial_coefficient<double>(m_elements.size(), m_k));
            return m_num_combinations;
        }
    }

private:
    std::vector<T> m_elements;
    int m_k;
    mutable int m_num_combinations;
};

template<typename Iter> Combinations(Iter begin, Iter end, int) -> 
                            Combinations<typename std::iterator_traits<Iter>::value_type>;

template<typename T>
class Combinations2Sets {
public:
    template<typename Iter, typename Iter2>
    Combinations2Sets(Iter begin_set1, 
                      Iter end_set1, 
                      Iter2 begin_set2, 
                      Iter2 end_set2, 
                      int k) : m_comb1(),
                               m_comb2(),
                               m_comb2_valid_combinations(),
                               m_num_combinations(-1),
                               m_k(k) {
        static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type,
                                     typename std::iterator_traits<Iter2>::value_type>, 
                                "The elements of both sets should be of the same type");


        std::vector<T> v1 {begin_set1, end_set1};
        std::vector<T> v2 {begin_set2, end_set2};

        std::sort(v1.begin(), v1.end());
        std::sort(v2.begin(), v2.end());

        std::unordered_set<T> common_elements;
        std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), 
                            std::inserter(common_elements, common_elements.end()));

        m_comb1 = Combinations<T>(std::move(v1), m_k);
        if (static_cast<int>(common_elements.size()) < k) {
            m_comb2 = Combinations<T>(std::move(v2), m_k);
            m_comb2_valid_combinations = m_comb2.num_combinations();
        } else {
            auto common_end = common_elements.end();
            for (size_t i = 0, common_start = v2.size() - common_elements.size(); i < common_start; ++i) {
                if (common_elements.find(v2[i]) != common_end) {
                    for (size_t j = v2.size()-1; j >= common_start; --j) {
                        if (common_elements.find(v2[j]) == common_end) {
                            std::swap(v2[i], v2[j]);
                        }
                    }
                }
            }

            m_comb2 = Combinations<T>(std::move(v2), m_k);
            m_comb2_valid_combinations = m_comb2.num_combinations() - 
                        std::round(boost::math::binomial_coefficient<double>(common_elements.size(), m_k));
        }
    }

    class combination2set_iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::vector<T>;
        using difference_type = int;
        using pointer = std::vector<T>*;
        using reference = std::vector<T>&;

        combination2set_iterator(const Combinations2Sets<T>& self, int idx) : m_self(self),
                                                                            it() {
            if (idx < m_self.m_comb1.num_combinations() || m_self.m_comb2_valid_combinations == 0) {
                it = typename Combinations<T>::combination_iterator(&m_self.m_comb1, idx);
            } else {
                it = typename Combinations<T>::combination_iterator(&m_self.m_comb2, idx - m_self.m_comb1.num_combinations());
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

        reference operator*() {
            return *it;
        }
        
        pointer operator->() { return &it.m_subset; }

        bool operator==(const combination2set_iterator& rhs) {
            return it == rhs.it;
        }

        bool operator!=(const combination2set_iterator& rhs) {
            return !(*this == rhs);
        }

    private:
        const Combinations2Sets<T>& m_self;
        typename Combinations<T>::combination_iterator it;
    };


    combination2set_iterator begin() { return combination2set_iterator(*this, 0); }
    combination2set_iterator end() { return combination2set_iterator(*this, num_combinations()); }

    int num_combinations() const {
        if (m_num_combinations != -1) {
            return m_num_combinations;
        } else {
            m_num_combinations = m_comb1.num_combinations() + m_comb2_valid_combinations;
            return m_num_combinations;
        }
    }

private:
    Combinations<T> m_comb1;
    Combinations<T> m_comb2;
    int m_comb2_valid_combinations;
    mutable int m_num_combinations;
    int m_k;
};

template<typename Iter, typename Iter2> Combinations2Sets(Iter, Iter, Iter2, Iter2, int) -> 
        Combinations2Sets<typename std::iterator_traits<Iter>::value_type>;

}

#endif //PYBNESIAN_UTIL_COMBINATIONS_HPP