#ifndef PGM_DATASET_HASH_UTILS_HPP
#define PGM_DATASET_HASH_UTILS_HPP

namespace util {

    template<typename T>
    void hash_combine(std::size_t &seed, T const &key) {
        std::hash<T> hasher;
        seed ^= hasher(key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
}


#endif //PGM_DATASET_HASH_UTILS_HPP