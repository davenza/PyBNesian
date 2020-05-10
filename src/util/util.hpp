#ifndef PGM_DATASET_UTIL_HPP
#define PGM_DATASET_UTIL_HPP

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
//            typename T::allocator_type,
            typename T::iterator,
            typename T::const_iterator,
            decltype(std::declval<T>().size()),
            decltype(std::declval<T>().begin()),
            decltype(std::declval<T>().end())
//            decltype(std::declval<T>().cbegin()),
//            decltype(std::declval<T>().cend())
            >
    > : public std::true_type {};

    template<typename T>
    inline constexpr auto is_container_v = is_container<T>::value;

//    template<typename T, typename = std::enable_if_t<std::is_integral_v<typename T::value_type>>>
//    struct is_integral_container : public is_container<T> {};
    template<typename T, typename _ = void>
    struct is_stringable : std::false_type {};

    template<typename T>
    struct is_stringable<T, std::void_t<std::enable_if_t<std::is_convertible_v<T, std::string>>>> :
            public std::true_type {};

    template<typename T, typename R=void>
    using enable_if_stringable_t = std::enable_if_t<std::is_convertible_v<T, std::string>, R>;

    template<typename T>
    using is_integral_container = std::integral_constant<bool, is_container_v<T> && std::is_integral_v<typename T::value_type>>;

    template<typename T>
    inline constexpr auto is_integral_container_v = is_integral_container<T>::value;

//    template<typename T, typename = std::enable_if_t<std::is_same_v<typename T::value_type, std::string>>>
//    struct is_string_container : public is_container<T> {};

    template<typename T>
    using is_string_container = std::integral_constant<bool, is_container_v<T> && std::is_convertible_v<typename T::value_type, std::string>>;

    template<typename T>
    inline constexpr auto is_string_container_v = is_string_container<T>::value;

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


}

#endif //PGM_DATASET_UTIL_HPP
