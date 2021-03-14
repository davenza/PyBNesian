#ifndef PYBNESIAN_UTIL_BIDIRECTIONALMAP_INDEX_HPP
#define PYBNESIAN_UTIL_BIDIRECTIONALMAP_INDEX_HPP

#include <util/vector.hpp>

namespace util {

template <typename T>
class BidirectionalMapIndex {
public:
    BidirectionalMapIndex() = default;

    BidirectionalMapIndex(std::vector<T> elems) : m_elems(elems), m_indices() {
        for (size_t i = 0; i < m_elems.size(); ++i) {
            m_indices.insert({m_elems[i], i});
        }
    }

    size_t size() const { return m_elems.size(); }

    const std::vector<T>& elements() const { return m_elems; }

    const std::unordered_map<T, int>& indices() const { return m_indices; }

    T& operator[](size_t index) { return m_elems[index]; }

    const T& operator[](size_t index) const { return m_elems[index]; }

    const T& element(int index) const {
        if (index < 0 || static_cast<size_t>(index) >= m_elems.size())
            throw std::out_of_range("Index " + std::to_string(index) + " not valid for map with " +
                                    std::to_string(size()) + " elements.");
        return m_elems[index];
    }

    int operator[](const T& elem) const { return m_indices.at(elem); }

    int index(const T& elem) const {
        auto it = m_indices.find(elem);
        if (it == m_indices.end()) throw std::out_of_range("Element " + elem + " not present in map");

        return it->second;
    }

    void insert(T elem) {
        if (!contains(elem)) {
            m_elems.push_back(elem);
            m_indices.insert({elem, m_elems.size() - 1});
        }
    }

    template <typename Iter>
    void insert(Iter begin, Iter end) {
        for (auto it = begin; it != end; ++it) {
            insert(*it);
        }
    }

    void remove(int index) {
        if (index >= 0 && static_cast<size_t>(index) < m_elems.size()) {
            m_indices.erase(m_elems[index]);
            util::swap_remove(m_elems, index);
            if (static_cast<size_t>(index) < m_elems.size()) m_indices[m_elems[index]] = index;
        }
    }

    void remove(const T& elem) {
        if (contains(elem)) {
            remove((*this)[elem]);
        }
    }

    bool contains(const T& elem) const { return m_indices.count(elem) > 0; }

    void reserve(size_t new_cap) { m_elems.reserve(new_cap); }

    typename std::vector<T>::const_iterator begin() const { return m_elems.begin(); }

    typename std::vector<T>::const_iterator end() const { return m_elems.end(); }

private:
    std::vector<T> m_elems;
    std::unordered_map<T, int> m_indices;
};

}  // namespace util

#endif  // PYBNESIAN_UTIL_BIDIRECTIONALMAP_INDEX_HPP
