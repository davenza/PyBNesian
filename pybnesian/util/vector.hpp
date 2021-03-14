#ifndef PYBNESIAN_UTIL_VECTOR_HPP
#define PYBNESIAN_UTIL_VECTOR_HPP

namespace util {

template <typename T>
void swap_remove(std::vector<T>& v, size_t idx) {
    if (idx < v.size() - 1) {
        std::swap(v[idx], v.back());
    }

    v.pop_back();
}

template <typename T>
void iter_swap_remove(std::vector<T>& v, typename std::vector<T>::iterator it) {
    if (it < v.end() - 1) {
        std::swap(*it, v.back());
    }

    v.pop_back();
}

template <typename T>
void swap_remove_v(std::vector<T>& v, T value) {
    iter_swap_remove(v, std::find(v.begin(), v.end(), value));
}

}  // namespace util

#endif  // PYBNESIAN_UTIL_VECTOR_HPP