#ifndef PGM_DATASET_MATH_CONSTANTS_HPP
#define PGM_DATASET_MATH_CONSTANTS_HPP

#include <boost/math/constants/constants.hpp>

namespace util {


    template<typename T>
    inline auto constexpr pi = boost::math::constants::pi<T>();
}

#endif //PGM_DATASET_MATH_CONSTANTS_HPP
