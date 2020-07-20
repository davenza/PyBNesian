#ifndef PGM_DATASET_UTIL_TYPES_HPP
#define PGM_DATASET_UTIL_TYPES_HPP

#include <factors/factors.hpp>

using factors::FactorType;

namespace util {
    using ArcVector = std::vector<std::pair<std::string, std::string>>;
    using FactorTypeVector = std::vector<std::pair<std::string, FactorType>>;
}

#endif //PGM_DATASET_UTIL_TYPES_HPP