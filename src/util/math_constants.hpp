#ifndef PYBNESIAN_MATH_CONSTANTS_HPP
#define PYBNESIAN_MATH_CONSTANTS_HPP

#include <boost/math/constants/constants.hpp>


namespace util {


    template<typename T>
    inline auto constexpr pi = boost::math::constants::pi<T>();

    template<typename T>
    inline auto constexpr one_div_root_two = boost::math::constants::one_div_root_two<T>();

    template<typename T>
    inline auto constexpr nan = std::numeric_limits<T>::quiet_NaN();
}

#endif //PYBNESIAN_MATH_CONSTANTS_HPP
