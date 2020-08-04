#ifndef PGM_DATASET_COMBINATIONS_HPP
#define PGM_DATASET_COMBINATIONS_HPP

#include <iterator>
#include <util/math_constants.hpp>
#include <boost/math/special_functions/binomial.hpp>

namespace util {

    template<typename T>
    class Combinations {
    public:
        template<typename Iter>
        Combinations(Iter begin, Iter end, int k) : m_elements(begin, end), m_k(k) {}

        class combination_iterator {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type = std::vector<int>;
            using difference_type = int;
            using pointer = std::vector<int>*;
            using reference = std::vector<int>&;

            combination_iterator(const Combinations<T>& self, int idx) : m_self(self), 
                                                                         m_subset(), 
                                                                         m_indices(), 
                                                                         m_idx(idx) {
                m_subset.reserve(m_self.m_k);
                m_indices.reserve(m_self.m_k);
                for (int i = 0; i < m_self.m_k; ++i) {
                    m_subset.push_back(m_self.m_elements[i]);
                    m_indices.push_back(i);
                }
            }

            void next_subset() {
                for (int i = m_subset.size()-1; i >= 0; --i) {
                    auto max_index = m_self.m_elements.size() - m_subset.size() + i;
                    if (m_indices[i] < max_index) {
                        ++m_indices[i];
                        m_subset[i] = m_self.m_elements[m_indices[i]];
                        
                        for (int j = i + 1; j < m_subset.size(); ++j) {
                            m_indices[j] = m_indices[j-1] + 1;
                            m_subset[j] = m_self.m_elements[m_indices[j]];
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
                return (this->m_idx == rhs.m_idx);
            }

            bool operator!=(const combination_iterator& rhs) {
                return !(*this == rhs);
            }

        private:
            const Combinations<T>& m_self;
            std::vector<T> m_subset;
            std::vector<int> m_indices;
            int m_idx;
        };

        combination_iterator begin() { return combination_iterator(*this, 0); }
        combination_iterator end() { return combination_iterator(*this, num_elements()); }
        
        int num_elements() {
            return std::round(boost::math::binomial_coefficient<double>(m_elements.size(), m_k));
        }

    private:
        std::vector<T> m_elements;
        int m_k;
    };

    template<typename Iter> Combinations(Iter begin, Iter end, int) -> 
                                Combinations<typename std::iterator_traits<Iter>::value_type>;

}

#endif //PGM_DATASET_COMBINATIONS_HPP