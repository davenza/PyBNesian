#ifndef PGM_DATASET_BN_TRAITS_HPP
#define PGM_DATASET_BN_TRAITS_HPP

#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>

using models::GaussianNetwork, models::SemiparametricBN;


namespace util {

    template<typename Model>
    struct is_gaussian_network : std::false_type {};

    template<typename DagType>
    struct is_gaussian_network<GaussianNetwork<DagType>> : std::true_type {};

    template<typename Model>
    inline constexpr auto is_gaussian_network_v = is_gaussian_network<Model>::value;

    template<typename Model, typename R = void>
    using enable_if_gaussian_network_t = std::enable_if_t<is_gaussian_network_v<Model>, R>;


    template<typename Model>
    struct is_semiparametricbn : std::false_type {};

    template<typename DagType>
    struct is_semiparametricbn<SemiparametricBN<DagType>> : std::true_type {};

    template<typename Model>
    inline constexpr auto is_semiparametricbn_v = is_semiparametricbn<Model>::value;

    template<typename Model, typename R = void>
    using enable_if_semiparametricbn_t = std::enable_if_t<is_semiparametricbn_v<Model>, R>;
}


#endif //PGM_DATASET_BN_TRAITS_HPP