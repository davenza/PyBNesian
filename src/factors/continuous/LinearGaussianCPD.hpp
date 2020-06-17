#ifndef PGM_DATASET_LINEARGAUSSIANCPD_HPP
#define PGM_DATASET_LINEARGAUSSIANCPD_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <dataset/dataset.hpp>

using Eigen::VectorXd, Eigen::Matrix, Eigen::Dynamic;

using dataset::DataFrame;

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

        const std::string& variable() const { return m_variable; }
        const std::vector<std::string>& evidence() const { return m_evidence; }
        bool fitted() const { return m_fitted; }
  
        void fit(py::handle pyobject);
        void fit(const DataFrame& df);

        //FIXME: Check model is fitted before computing
        VectorXd logpdf(py::handle pyobject) const;
        VectorXd logpdf(const DataFrame& df) const;

        //FIXME: Check model is fitted before computing
        double slogpdf(py::handle pyobject) const;
        double slogpdf(const DataFrame& df) const;

        std::string ToString() const;

    private:
        std::string m_variable;
        std::vector<std::string> m_evidence;
        bool m_fitted;
        VectorXd m_beta;
        double m_variance;
    };

}

#endif //PGM_DATASET_LINEARGAUSSIANCPD_HPP
