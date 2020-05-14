#ifndef PGM_DATASET_LINEARGAUSSIANCPD_HPP
#define PGM_DATASET_LINEARGAUSSIANCPD_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

using Eigen::VectorXd;

namespace py = pybind11;

namespace factors::continuous {


    struct LinearGaussianCPD_Params {
        VectorXd beta;
        double variance;
    };

    class LinearGaussianCPD {
    public:
        using ParamsClass = LinearGaussianCPD_Params;
        
        LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence);
        LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence,
                            const std::vector<double> beta, const double variance);
  
        void fit(py::handle pyobject);
    private:
        std::string m_variable;
        std::vector<std::string> m_evidence;
        VectorXd m_beta;
        double m_variance;
    };
}

#endif //PGM_DATASET_LINEARGAUSSIANCPD_HPP
