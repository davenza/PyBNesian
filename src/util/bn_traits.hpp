#ifndef PGM_DATASET_BN_TRAITS_HPP
#define PGM_DATASET_BN_TRAITS_HPP

#include <models/BayesianNetwork.hpp>

using models::GaussianNetwork;


namespace util {

    template<typename Model>
    struct is_gaussian_network : std::false_type {};

    template<typename DagType>
    struct is_gaussian_network<GaussianNetwork<DagType>> : std::true_type {};

    template<typename Model>
    inline constexpr auto is_gaussian_network_v = is_gaussian_network<Model>::value;


    template<typename Model, typename R = void>
    using enable_if_gaussian_network_t = std::enable_if_t<is_gaussian_network_v<Model>, R>;
}


#endif //PGM_DATASET_BN_TRAITS_HPP