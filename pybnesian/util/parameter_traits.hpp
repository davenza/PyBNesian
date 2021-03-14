#ifndef PYBNESIAN_UTIL_PARAMETER_TRAITS_HPP
#define PYBNESIAN_UTIL_PARAMETER_TRAITS_HPP

#include <string>

namespace util {

template <class...>
constexpr std::false_type always_false{};

template <typename T>
using is_string_cvref = std::is_same<std::string, std::remove_cv_t<std::remove_reference_t<T>>>;

template <typename T>
inline constexpr auto is_string_cvref_v = is_string_cvref<T>::value;

//    Check that type is iterable container:
//    https://stackoverflow.com/questions/12042824/how-to-write-a-type-trait-is-container-or-is-vector
template <typename T, typename _ = void>
struct is_container : std::false_type {};

template <typename T>
struct is_container<T,
                    std::void_t<typename T::value_type,
                                typename T::size_type,
                                typename T::iterator,
                                typename T::const_iterator,
                                decltype(std::declval<T>().size()),
                                decltype(std::declval<T>().begin()),
                                decltype(std::declval<T>().end()),
                                // Remove std::string type and references.
                                std::enable_if_t<std::negation_v<is_string_cvref<T>>>>> : public std::true_type {};

template <typename T>
inline constexpr auto is_container_v = is_container<T>::value;

template <typename T, typename R = void>
using enable_if_container_t = std::enable_if_t<is_container_v<T>, R>;

template <typename T, typename _ = void>
struct is_stringable : std::false_type {};

template <typename T>
struct is_stringable<T, std::void_t<std::enable_if_t<std::is_convertible_v<T, std::string>>>> : public std::true_type {
};

template <typename T>
inline constexpr auto is_stringable_v = is_stringable<T>::value;

template <typename T, typename R = void>
using enable_if_stringable_t = std::enable_if_t<is_stringable_v<T>, R>;

template <typename T, typename _ = void>
struct is_integral_container : std::false_type {};

template <typename T>
struct is_integral_container<
    T,
    std::void_t<enable_if_container_t<T>, std::enable_if_t<std::is_integral_v<typename T::value_type>>>>
    : public std::true_type {};

template <typename T>
inline constexpr auto is_integral_container_v = is_integral_container<T>::value;

template <typename T, typename R = void>
using enable_if_integral_container_t = std::enable_if_t<is_integral_container_v<T>, R>;

template <typename T, typename _ = void>
struct is_string_container : std::false_type {};

template <typename T>
struct is_string_container<T, std::void_t<enable_if_container_t<T>, enable_if_stringable_t<typename T::value_type>>>
    : public std::true_type {};

template <typename T>
inline constexpr auto is_string_container_v = is_string_container<T>::value;

template <typename T, typename R = void>
using enable_if_string_container_t = std::enable_if_t<is_string_container_v<T>, R>;

template <typename T>
using is_index = std::integral_constant<bool, std::is_integral_v<T> || is_stringable_v<T>>;

template <typename T>
inline constexpr auto is_index_v = is_index<T>::value;

template <typename T, typename R = void>
using enable_if_index_t = std::enable_if_t<is_index_v<T>, R>;

template <typename T>
using is_index_container = std::integral_constant<bool, is_integral_container_v<T> || is_string_container_v<T>>;

template <typename T>
inline constexpr auto is_index_container_v = is_index_container<T>::value;

template <typename T, typename R = void>
using enable_if_index_container_t = std::enable_if_t<is_index_container_v<T>, R>;

template <typename T, typename _ = void>
struct is_iterator : std::false_type {};

template <typename T>
struct is_iterator<T,
                   std::void_t<typename std::iterator_traits<T>::difference_type,
                               typename std::iterator_traits<T>::pointer,
                               typename std::iterator_traits<T>::reference,
                               typename std::iterator_traits<T>::value_type,
                               typename std::iterator_traits<T>::iterator_category>> : public std::true_type {};

template <typename T>
inline constexpr auto is_iterator_v = is_iterator<T>::value;

template <typename T, typename R = void>
using enable_if_iterator_t = std::enable_if_t<is_iterator_v<T>, R>;

template <typename T, typename _ = void>
struct is_integral_iterator : std::false_type {};

template <typename T>
struct is_integral_iterator<
    T,
    std::void_t<enable_if_iterator_t<T>,
                std::enable_if_t<std::is_integral_v<typename std::iterator_traits<T>::value_type>>>>
    : public std::true_type {};

template <typename T>
inline constexpr auto is_integral_iterator_v = is_integral_iterator<T>::value;

template <typename T, typename R = void>
using enable_if_integral_iterator_t = std::enable_if_t<is_integral_iterator_v<T>, R>;

template <typename T, typename _ = void>
struct is_string_iterator : std::false_type {};

template <typename T>
struct is_string_iterator<
    T,
    std::void_t<enable_if_iterator_t<T>, enable_if_stringable_t<typename std::iterator_traits<T>::value_type>>>
    : public std::true_type {};

template <typename T>
inline constexpr auto is_string_iterator_v = is_string_iterator<T>::value;

template <typename T, typename R = void>
using enable_if_string_iterator_t = std::enable_if_t<is_string_iterator_v<T>, R>;

template <typename T>
using is_index_iterator = std::integral_constant<bool, is_integral_iterator_v<T> || is_string_iterator_v<T>>;

template <typename T>
inline constexpr auto is_index_iterator_v = is_index_iterator<T>::value;

template <typename T, typename R = void>
using enable_if_index_iterator_t = std::enable_if_t<is_index_iterator_v<T>, R>;

}  // namespace util

namespace dataset {
template <typename Index, typename = util::enable_if_index_t<Index, void>>
struct DynamicVariable;
}

namespace util {

// From https://stackoverflow.com/questions/44012938/how-to-tell-if-template-type-is-an-instance-of-a-template-class
template <template <typename...> typename, typename...>
struct is_template_instantation : public std::false_type {};

template <template <typename...> typename U, typename... Ts>
struct is_template_instantation<U, U<Ts...>> : public std::true_type {};

template <template <typename...> typename U, typename... Ts>
inline constexpr auto is_template_instantation_v = is_template_instantation<U, Ts...>::value;

template <template <typename...> typename TemplatedClass, typename Derived, typename R = void>
using enable_if_template_instantation_t = std::enable_if_t<is_template_instantation_v<TemplatedClass, Derived>, R>;

// Implement is_template_instantation for non-type templates. Based on:
// https://stackoverflow.com/questions/22674347/c-variadic-template-with-non-type-parameters-of-different-types/22675220

// This is only valid with clang, because gcc do not allow non-type variadic template expansion:
// https://stackoverflow.com/questions/51717440/a-compile-type-template-predicate-compiles-with-clang-but-not-with-gcc-or-msvc
//
// template<typename... Types>
// struct GenericInstantation {
//     template<template <Types...> typename U, typename... Ts>
//     struct is_template_instantation : public std::false_type {};

//     template<template <Types...> typename U, Types... Ts>
//     struct is_template_instantation<U, U<Ts...>> : public std::true_type {};

//     template<template <Types...> typename U, typename... Ts>
//     inline static constexpr bool is_template_instantation_v = is_template_instantation<U, Ts...>::value;

//     template<template <Types...> typename TemplatedClass, typename Derived, typename R = void>
//     using enable_if_template_instantation_t = std::enable_if_t<is_template_instantation_v<TemplatedClass, Derived>,
//     R>;
// };

template <typename T>
struct GenericInstantation {
    template <template <T> typename, typename>
    struct is_template_instantation : public std::false_type {};

    template <template <T> typename U, T value>
    struct is_template_instantation<U, U<value>> : public std::true_type {};

    template <template <T> typename U, typename Class>
    inline static constexpr bool is_template_instantation_v = is_template_instantation<U, Class>::value;

    template <template <T> typename TemplatedClass, typename Derived, typename R = void>
    using enable_if_template_instantation_t = std::enable_if_t<is_template_instantation_v<TemplatedClass, Derived>, R>;
};

template <typename T, typename = void>
struct is_dynamic_index : public std::false_type {};

template <typename T>
struct is_dynamic_index<T, std::void_t<enable_if_template_instantation_t<dataset::DynamicVariable, T>>>
    : public std::true_type {};

template <typename T>
inline constexpr auto is_dynamic_index_v = is_dynamic_index<T>::value;

template <typename T, typename R = void>
using enable_if_dynamic_index_t = std::enable_if_t<is_dynamic_index_v<T>, R>;

template <typename T, typename = void>
struct is_dynamic_index_container : public std::false_type {};

template <typename T>
struct is_dynamic_index_container<
    T,
    std::void_t<enable_if_container_t<T>, enable_if_dynamic_index_t<typename T::value_type>>> : public std::true_type {
};

template <typename T>
inline constexpr auto is_dynamic_index_container_v = is_dynamic_index_container<T>::value;

template <typename T, typename R = void>
using enable_if_dynamic_index_container_t = std::enable_if_t<is_dynamic_index_container_v<T>, R>;

template <typename T, typename _ = void>
struct is_dynamic_index_iterator : std::false_type {};

template <typename T>
struct is_dynamic_index_iterator<
    T,
    std::void_t<enable_if_iterator_t<T>, enable_if_dynamic_index_t<typename std::iterator_traits<T>::value_type>>>
    : public std::true_type {};

template <typename T>
inline constexpr auto is_dynamic_index_iterator_v = is_dynamic_index_iterator<T>::value;

template <typename T, typename R = void>
using enable_if_dynamic_index_iterator_t = std::enable_if_t<is_dynamic_index_iterator_v<T>, R>;

template <typename V, typename T>
using is_vector_of_type = std::is_same<std::vector<T>, std::remove_cv_t<std::remove_reference_t<V>>>;

template <typename V, typename T>
inline constexpr auto is_vector_of_type_v = is_vector_of_type<V, T>::value;

template <typename V, typename T, typename R = void>
using enable_if_vector_of_type_t = std::enable_if_t<is_vector_of_type_v<V, T>, R>;

}  // namespace util

#endif  // PYBNESIAN_UTIL_PARAMETER_TRAITS_HPP
