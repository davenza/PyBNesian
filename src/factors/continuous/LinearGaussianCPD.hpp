//
// Created by david on 17/4/20.
//

#ifndef PGM_DATASET_LINEARGAUSSIANCPD_HPP
#define PGM_DATASET_LINEARGAUSSIANCPD_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dataset/dataset.hpp>
#include <Eigen/Dense>

using Eigen::VectorXd;

namespace py = pybind11;

using namespace dataset;

namespace factors::continuous {
    class LinearGaussianCPD {

    public:
//        LinearGaussianCPD();
        LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence);
        LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence,
                            const std::vector<double> beta);
//
        void fit(py::handle pyobject);

    private:
        template<typename ArrowType, bool contains_null>
        void _fit(DataFrame df);
        template<typename ArrowType, bool contains_null>
        void _fit_1parent(DataFrame df);
        template<typename ArrowType, bool contains_null>
        void _fit_2parent(DataFrame df);
        template<typename ArrowType, bool contains_null>
        void _fit_nparent(DataFrame df);

        std::string m_variable;
        std::vector<std::string> m_evidence;
        VectorXd m_beta;
        double m_variance;
    };
}

#endif //PGM_DATASET_LINEARGAUSSIANCPD_HPP
