#ifndef PYBNESIAN_UTIL_TEMPORAL_HPP
#define PYBNESIAN_UTIL_TEMPORAL_HPP

#include <string>
#include <vector>
#include <regex>
#include <util/parameter_traits.hpp>

namespace util {
    std::string temporal_name(const std::string& name, int slice_index);
    std::vector<std::string> temporal_names(const std::vector<std::string>& variables, int offset_slice, int markovian_order);
    std::vector<std::vector<std::string>> temporal_slice_names(const std::vector<std::string>& variables,
                                                               int start_slice,
                                                               int markovian_order);

    template<typename Index>
    struct TemporalIndex {
        using variable_type = Index;
        
        TemporalIndex() = default;
        TemporalIndex(Index v, int s) : variable(v), temporal_slice(s) {}
        TemporalIndex(std::pair<Index, int> t) : TemporalIndex(t.first, t.second) {}

        template<typename T = Index, util::enable_if_stringable_t<T, int> = 0>
        std::string temporal_name() const {
            return util::temporal_name(variable, temporal_slice);
        }

        static std::optional<TemporalIndex<std::string>> from_string(const std::string& name) {
            const std::regex tindex_regex("(.*)\\_t\\_([0-9]+)");
            
            std::smatch base_match;
            if (std::regex_match(name, base_match, tindex_regex)) {
                return TemporalIndex(base_match[1], std::stoi(base_match[2]));
            } else {
                return std::nullopt;
            }
        }

        Index variable;
        int temporal_slice;
    };

    template<typename T, typename = void>
    struct is_temporal_index : public std::false_type {};

    template<typename T>
    struct is_temporal_index<T,
                            std::void_t<
                                enable_if_template_instantation_t<
                                    TemporalIndex, 
                                    T
                                >
                            >
    > : public std::true_type {};

    template<typename T>
    inline constexpr auto is_temporal_index_v = is_temporal_index<T>::value;
    
    template<typename T, typename R = void>
    using enable_if_temporal_index_t = std::enable_if_t<is_temporal_index_v<T>, R>;

    template<typename T, typename = void>
    struct is_temporal_index_container : public std::false_type {};

    template<typename T>
    struct is_temporal_index_container<T,
                                      std::void_t<
                                        enable_if_container_t<T>,
                                        enable_if_temporal_index_t<typename T::value_type>
                                     >
    > : public std::true_type {};

    template<typename T>
    inline constexpr auto is_temporal_index_container_v = is_temporal_index_container<T>::value;

    template<typename T, typename R = void>
    using enable_if_temporal_index_container_t = std::enable_if_t<is_temporal_index_container_v<T>, R>;

    template<typename T, typename _ = void>
    struct is_temporal_index_iterator : std::false_type {};

    template<typename T>
    struct is_temporal_index_iterator<
            T,
            std::void_t<
                enable_if_iterator_t<T>,
                enable_if_temporal_index_t<typename std::iterator_traits<T>::value_type>
            >
    > : public std::true_type {};

    template<typename T>
    inline constexpr auto is_temporal_index_iterator_v = is_temporal_index_iterator<T>::value;

    template<typename T, typename R = void>
    using enable_if_temporal_index_iterator_t = std::enable_if_t<is_temporal_index_iterator_v<T>, R>;
}

#endif //PYBNESIAN_UTIL_TEMPORAL_HPP