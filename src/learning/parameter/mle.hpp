#ifndef PGM_DATASET_MLE_HPP
#define PGM_DATASET_MLE_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>

using factors::continuous::LinearGaussianCPD;


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
        typename CPD_traits<Derived>::CPD_params estimate(DataFrame& df, const std::string& variable,  const std::vector<std::string>& evidence) {
            static_cast<Derived*>(this)->estimate(df, variable, evidence);
        }
    };

    template<typename CPD>
    class MLE : ParameterEstimator<MLE<CPD>>{
    public:
        using CPD_type = CPD;

        typename CPD::ParamsClass estimate(DataFrame& df, const std::string& variable,  const std::vector<std::string>& evidence);
    };






}

#endif //PGM_DATASET_MLE_HPP