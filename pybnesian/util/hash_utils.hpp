#ifndef PYBNESIAN_UTIL_HASH_UTILS_HPP
#define PYBNESIAN_UTIL_HASH_UTILS_HPP

#include <functional>

namespace util {

template <typename T>
void hash_combine(std::size_t &seed, T const &key) {
    std::hash<T> hasher;
    seed ^= hasher(key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}  // namespace util

#endif  // PYBNESIAN_UTIL_HASH_UTILS_HPP