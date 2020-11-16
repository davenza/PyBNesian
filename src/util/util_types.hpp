#ifndef PYBNESIAN_UTIL_UTIL_TYPES_HPP
#define PYBNESIAN_UTIL_UTIL_TYPES_HPP

#include <vector>
#include <factors/factors.hpp>

using factors::FactorType;

namespace util {
    using ArcVector = std::vector<std::pair<std::string, std::string>>;
    using EdgeVector = std::vector<std::pair<std::string, std::string>>;
    using FactorTypeVector = std::vector<std::pair<std::string, FactorType>>;
}

#endif //PYBNESIAN_UTIL_UTIL_TYPES_HPP