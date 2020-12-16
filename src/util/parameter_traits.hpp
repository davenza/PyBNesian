#ifndef PYBNESIAN_UTIL_PARAMETER_TRAITS_HPP
#define PYBNESIAN_UTIL_PARAMETER_TRAITS_HPP

#include <string>

namespace util {

//    Check that type is iterable container: https://stackoverflow.com/questions/12042824/how-to-write-a-type-trait-is-container-or-is-vector
    template<typename T, typename _ = void>
    struct is_container : std::false_type {};

    template<typename T>
    struct is_container<
            T,
            std::void_t<
            typename T::value_type,
            typename T::size_type,
            typename T::iterator,
            typename T::const_iterator,
            decltype(std::declval<T>().size()),
            decltype(std::declval<T>().begin()),
            decltype(std::declval<T>().end())
            >
    > : public std::true_type {};

    template<typename T>
    inline constexpr auto is_container_v = is_container<T>::value;

    template<typename T, typename _ = void>
    struct is_stringable : std::false_type {};

    template<typename T>
    struct is_stringable<T, std::void_t<std::enable_if_t<std::is_convertible_v<T, std::string>>>> :
            public std::true_type {};

    template<typename T, typename R=void>
    using enable_if_stringable_t = std::enable_if_t<std::is_convertible_v<T, std::string>, R>;

    template<typename T, typename _ = void>
    struct is_integral_container : std::false_type {};

    template<typename T>
    struct is_integral_container<
            T,
            std::void_t<
                std::enable_if_t<is_container_v<T>>,
                std::enable_if_t<std::is_integral_v<typename T::value_type>>
            >
    > : public std::true_type {};

    template<typename T>
    inline constexpr auto is_integral_container_v = is_integral_container<T>::value;

    template<typename T, typename R = void>
    using enable_if_integral_container_t = std::enable_if_t<is_integral_container_v<T>, R>;

    template<typename T, typename _ = void>
    struct is_string_container : std::false_type {};

    template<typename T>
    struct is_string_container<
            T,
            std::void_t<
                std::enable_if_t<is_container_v<T>>,
                std::enable_if_t<std::is_convertible_v<typename T::value_type, const std::string&>>
            >
    > : public std::true_type {};

    template<typename T>
    inline constexpr auto is_string_container_v = is_string_container<T>::value;

    template<typename T>
    using is_index = std::integral_constant<bool,
                                            std::is_integral_v<T> || 
                                            std::is_convertible_v<typename T::value_type, const std::string&>>;

    template<typename T>
    inline constexpr auto is_index_v = is_index<T>::value;

    template<typename T, typename R = void>
    using enable_if_index_t = std::enable_if_t<is_index_v<T>, R>;

    template<typename T, typename R = void>
    using enable_if_index_container_t = std::enable_if_t<
                                            std::conjunction_v<
                                                is_container<T>,
                                                std::negation<is_stringable<T>>,
                                                std::disjunction<
                                                    is_integral_container<T>,
                                                    is_string_container<T>
                                                >
                                            >, R
                                        >;

    // From https://stackoverflow.com/questions/44012938/how-to-tell-if-template-type-is-an-instance-of-a-template-class
    template <template <typename...> typename, typename...>
    struct is_template_instantation : public std::false_type {};

    template <template <typename...> typename U, typename...Ts>
    struct is_template_instantation<U, U<Ts...>> : public std::true_type {};

    template <template <typename...> typename U, typename...Ts>
    inline constexpr auto is_template_instantation_v = is_template_instantation<U, Ts...>::value;


    template<typename>
    class DynamicVariable;
    template<typename T, typename R = void>
    using enable_if_dynamic_index_container_t = std::enable_if_t<
                                            std::conjunction_v<
                                                is_container<T>,
                                                is_template_instantation_v<DynamicVariable, typename T::value_type>
                                            >, R
                                        >;


    template<typename T, typename _ = void>
    struct is_iterator : std::false_type {};

    template<typename T>
    struct is_iterator<
            T,
            std::void_t<
                typename std::iterator_traits<T>::difference_type,
                typename std::iterator_traits<T>::pointer,
                typename std::iterator_traits<T>::reference,
                typename std::iterator_traits<T>::value_type,
                typename std::iterator_traits<T>::iterator_category
            >
    > : public std::true_type {};

    template<typename T>
    inline constexpr auto is_iterator_v = is_iterator<T>::value;

    template<typename T, typename R = void>
    using enable_if_iterator_t = std::enable_if_t<is_iterator_v<T>, R>;

    template<typename T>
    using is_integral_iterator = std::integral_constant<bool, is_iterator_v<T> && std::is_integral_v<typename std::iterator_traits<T>::value_type>>;

    template<typename T>
    inline constexpr auto is_integral_iterator_v = is_integral_iterator<T>::value;

    template<typename T, typename R = void>
    using enable_if_integral_iterator_t = std::enable_if_t<is_integral_iterator_v<T>, R>;

    template<typename T>
    using is_string_iterator = std::integral_constant<bool, is_iterator_v<T> && std::is_convertible_v<typename std::iterator_traits<T>::value_type, 
                                                                                                        const std::string&>>;

    template<typename T>
    inline constexpr auto is_string_iterator_v = is_string_iterator<T>::value;

    template<typename T, typename R = void>
    using enable_if_index_iterator_t = std::enable_if_t<
                                            std::conjunction_v<
                                                is_iterator<T>,
                                                std::disjunction<
                                                    is_integral_iterator<T>,
                                                    is_string_iterator<T>
                                                >
                                            >, R
                                        >;

    template<typename V, typename T>
    using is_vector_of_type = std::is_same<std::vector<T>, std::remove_reference_t<V>>;

    template<typename V, typename T>
    inline constexpr auto is_vector_of_type_v = is_vector_of_type<V,T>::value;

    template<typename V, typename T, typename R = void>
    using enable_if_vector_of_type_t = std::enable_if_t<is_vector_of_type_v<V, T>, R>;


}

#endif //PYBNESIAN_UTIL_PARAMETER_TRAITS_HPP
