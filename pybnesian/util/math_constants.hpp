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

template <typename T>
T constexpr sqrtNewtonRaphson(T x, T curr, T prev) {
    return curr == prev ? curr : sqrtNewtonRaphson<T>(x, 0.5 * (curr + x / curr), curr);
}

template <typename T>
T constexpr sqrt_constexpr(T x) {
    return sqrtNewtonRaphson<T>(x, x, 0);
}

}  // namespace detail

template <typename T>
inline auto constexpr machine_tol = detail::sqrt_constexpr(std::numeric_limits<T>::epsilon());

}  // namespace util

#endif  // PYBNESIAN_UTIL_MATH_CONSTANTS_HPP
