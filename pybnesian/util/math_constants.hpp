#ifndef PYBNESIAN_UTIL_MATH_CONSTANTS_HPP
#define PYBNESIAN_UTIL_MATH_CONSTANTS_HPP

#include <boost/math/constants/constants.hpp>

namespace util {

template <typename T>
inline auto constexpr pi = boost::math::constants::pi<T>();

template <typename T>
inline auto constexpr root_two = boost::math::constants::root_two<T>();

template <typename T>
inline auto constexpr one_div_root_two = boost::math::constants::one_div_root_two<T>();

template <typename T>
inline auto constexpr nan = std::numeric_limits<T>::quiet_NaN();

namespace detail {

double constexpr sqrtNewtonRaphson(double x, double curr, double prev) {
    return curr == prev ? curr : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
}

double constexpr sqrt_constexpr(double x) { return sqrtNewtonRaphson(x, x, 0); }

}  // namespace detail

inline auto constexpr machine_tol = detail::sqrt_constexpr(std::numeric_limits<double>::epsilon());

}  // namespace util

#endif  // PYBNESIAN_UTIL_MATH_CONSTANTS_HPP
