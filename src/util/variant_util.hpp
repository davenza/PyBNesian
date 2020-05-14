#ifndef PGM_DATASET_VARIANT_UTIL_HPP
#define PGM_DATASET_VARIANT_UTIL_HPP

#include <variant>

namespace util {

    // This code allows casting between a variant to another variant that is a superset.
    // It is based on: https://stackoverflow.com/questions/47203255/convert-stdvariant-to-another-stdvariant-with-super-set-of-types
    // This version includes changes to deal with movable-only types (such as std::unique_ptr).

    template <class... Args>
    struct variant_cast_proxy
    {
        std::variant<Args...> v;

        variant_cast_proxy(const std::variant<Args...>& other) : v(other) {};

        variant_cast_proxy(std::variant<Args...>&& other) : v(std::move(other)) {};

        template <class... ToArgs>
        operator std::variant<ToArgs...>() const &
        {
            return std::visit([](auto&& arg) -> std::variant<ToArgs...>
                   { return arg ; }, v);
        }

        template <class... ToArgs>
        operator std::variant<ToArgs...>() &&
        {
            return std::visit([](auto&& arg) -> std::variant<ToArgs...>
                   { return std::move(arg); }, v);
        }
    };

    template <class... Args>
    auto variant_cast(const std::variant<Args...>& v) -> variant_cast_proxy<Args...>
    {
        return variant_cast_proxy(v);
    }

    template <class... Args>
    auto variant_cast(std::variant<Args...>&& v) -> variant_cast_proxy<Args...>
    {
        return variant_cast_proxy(std::move(v));
    }


    template <bool...> struct bool_pack;
    template <bool... v>
    using all_true = std::is_same<bool_pack<true, v...>, bool_pack<v..., true>>;


    template<class... Ts> struct overloaded_same_type_and_cols : Ts... {


        template<typename PtrT, typename... PtrArgs, std::enable_if_t<

                ((PtrT::element_type::ColsAtCompileTime != PtrArgs::element_type::ColsAtCompileTime) || ...)
                                                ||
                (!std::is_same_v<typename PtrT::element_type::Scalar, typename PtrArgs::element_type::Scalar> || ...)

                                    , int> = 0
                >
        auto operator()(PtrT& first_eigen, PtrArgs&... eigens) {
            throw std::invalid_argument("Unreachable code. This is an indicative of a bug.");
        }

        using Ts::operator()...;
    };

    template<class... Ts> overloaded_same_type_and_cols(Ts...) ->
    overloaded_same_type_and_cols<Ts...>; // not needed as of C++20



}

#endif //PGM_DATASET_VARIANT_UTIL_HPP
