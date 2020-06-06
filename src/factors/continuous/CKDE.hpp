#ifndef PGM_DATASET_CKDE_HPP
#define PGM_DATASET_CKDE_HPP

#include<iostream>
#include <CL/cl2.hpp>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <dataset/dataset.hpp>

namespace py = pybind11;
using dataset::DataFrame;
using Eigen::VectorXd;

namespace factors::continuous {
    
    enum class KDEBandwidth {
        SCOTT
    };

    class CKDE {
    public:
        CKDE(const std::string variable, const std::vector<std::string> evidence) : m_variable(variable), 
                                                                                    m_evidence(evidence) {}
        CKDE(const std::string variable, const std::vector<std::string> evidence, KDEBandwidth b_selector) 
                                                                                  : m_variable(variable), 
                                                                                    m_evidence(evidence), 
                                                                                    m_bselector(b_selector) {}

        void fit(py::handle pyobject);
        void fit(const DataFrame& df);

        VectorXd logpdf(py::handle pyobject) const;
        VectorXd logpdf(const DataFrame& df) const;

        double slogpdf(py::handle pyobject) const;
        double slogpdf(const DataFrame& df) const;
    private:
        template<typename ArrowType>
        void _fit(const DataFrame& df);
        template<typename ArrowType>
        void _fit_null(const DataFrame& df);


        std::string m_variable;
        std::vector<std::string> m_evidence;
        KDEBandwidth m_bselector;

    };

    void opencl();

}

#endif //PGM_DATASET_CKDE_HPP