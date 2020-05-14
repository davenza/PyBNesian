#ifndef PGM_DATASET_MLEBASE_HPP
#define PGM_DATASET_MLEBASE_HPP

#include <dataset/dataset.hpp>

using namespace dataset;

namespace learning::parameter {

    template <typename MLE_type> 
    struct CPD_traits;

    template<template<typename> typename Estimator, typename CPD>
    struct CPD_traits<Estimator<CPD>> {
        using CPD_type = CPD;
        using CPD_params = typename CPD::ParamsClass;
    };

    template<typename Derived>
    class ParameterEstimator {

    public:
        typename CPD_traits<Derived>::CPD_params estimate(const DataFrame& df, const std::string& variable,  const std::vector<std::string>& evidence) {
            static_cast<Derived*>(this)->estimate(df, variable, evidence);
        }
    };

    template<typename CPD>
    class MLE : ParameterEstimator<MLE<CPD>>{
    public:
        typename CPD::ParamsClass estimate(const DataFrame& df, const std::string& variable,  const std::vector<std::string>& evidence);
    };
}

#endif //PGM_DATASET_MLEBASE_HPP