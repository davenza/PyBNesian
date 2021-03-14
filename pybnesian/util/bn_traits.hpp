#ifndef PYBNESIAN_UTIL_BN_TRAITS_HPP
#define PYBNESIAN_UTIL_BN_TRAITS_HPP

#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>

using models::ConditionalBayesianNetworkBase;
using models::GaussianNetwork, models::SemiparametricBN;

namespace util {

template <typename Model, typename _ = void>
struct is_conditionalbn : public std::false_type {};

template <typename Model>
struct is_conditionalbn<Model, std::void_t<std::enable_if_t<std::is_base_of_v<ConditionalBayesianNetworkBase, Model>>>>
    : public std::true_type {};

template <typename Model>
inline constexpr auto is_conditionalbn_v = is_conditionalbn<Model>::value;

template <typename Model, typename R = void>
using enable_if_conditionalbn_t = std::enable_if_t<is_conditionalbn_v<Model>, R>;

template <typename Model, typename R = void>
using enable_if_not_conditionalbn_t = std::enable_if_t<!is_conditionalbn_v<Model>, R>;

}  // namespace util

#endif  // PYBNESIAN_UTIL_BN_TRAITS_HPP