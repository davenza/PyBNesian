//
// Created by david on 17/4/20.
//

#ifndef PGM_DATASET_LINEARGAUSSIANCPD_HPP
#define PGM_DATASET_LINEARGAUSSIANCPD_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dataset/dataset.hpp>


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
        void _fit(DataFrame df);
        std::string variable;
        std::vector<std::string> evidence;
        std::vector<double> beta;
    };
}

#endif //PGM_DATASET_LINEARGAUSSIANCPD_HPP
