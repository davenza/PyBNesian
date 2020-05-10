#ifndef PGM_DATASET_VARIANT_UTIL_HPP
#define PGM_DATASET_VARIANT_UTIL_HPP

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
}

#endif //PGM_DATASET_VARIANT_UTIL_HPP
