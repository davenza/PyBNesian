#ifndef PGM_DATASET_CKDE_HPP
#define PGM_DATASET_CKDE_HPP

#include<iostream>
#include <CL/cl2.hpp>
#include <opencl/opencl_config.hpp>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <dataset/dataset.hpp>
#include <util/math_constants.hpp>

namespace py = pybind11;
using dataset::DataFrame;
using Eigen::VectorXd, Eigen::Ref, Eigen::LLT;
using opencl::OpenCLConfig;

namespace factors::continuous {
    
    enum class KDEBandwidth {
        SCOTT
    };

    class CKDE {
    public:
        CKDE(const std::string variable, const std::vector<std::string> evidence) : CKDE(variable, evidence, KDEBandwidth::SCOTT) {}
        CKDE(const std::string variable, const std::vector<std::string> evidence, KDEBandwidth b_selector) 
                                                                                  : m_variable(variable), 
                                                                                    m_evidence(evidence), 
                                                                                    m_bselector(b_selector),
                                                                                    m_H_cholesky(),
                                                                                    m_training(),
                                                                                    m_lognorm_const(0) {}

        void fit(py::handle pyobject);
        void fit(const DataFrame& df);

        VectorXd logpdf(py::handle pyobject) const;
        VectorXd logpdf(const DataFrame& df) const;

        double slogpdf(py::handle pyobject) const;
        double slogpdf(const DataFrame& df) const;
    private:
        template<typename ArrowType, bool contains_null>
        void _fit(const DataFrame& df);

        template<typename ArrowType, bool contains_null>
        Matrix<typename ArrowType::c_type, Dynamic, Dynamic> compute_bandwidth(const DataFrame& df);

        // template<typename ArrowType>
        // Matrix<typename ArrowType::c_type, Dynamic, Dynamic> compute_bandwidth();


        std::string m_variable;
        std::vector<std::string> m_evidence;
        KDEBandwidth m_bselector;
        cl::Buffer m_H_cholesky;
        cl::Buffer m_training;
        double m_lognorm_const;
    };

    void opencl();

    template<typename ArrowType, bool contains_null>
    Matrix<typename ArrowType::c_type, Dynamic, Dynamic> CKDE::compute_bandwidth(const DataFrame& df) {
        using CType = typename ArrowType::c_type;
        auto cov = df.cov<ArrowType, contains_null>(m_variable, m_evidence);
        auto bandwidth = std::pow(static_cast<CType>(df->num_rows()), -1 / static_cast<CType>(m_evidence.size() + 5)) * (*cov) ;

        return bandwidth;
    }

    template<typename ArrowType, bool contains_null>
    void CKDE::_fit(const DataFrame& df) {
        using CType = typename ArrowType::c_type;
        using MatrixType = Matrix<CType, Dynamic, Dynamic>;

        auto n = 1 + m_evidence.size();

        auto bandwidth = compute_bandwidth<ArrowType, contains_null>(df);

        LLT<Ref<MatrixType>> llt_cov(bandwidth);

        auto opencl = OpenCLConfig::get();
        opencl.copy_to_buffer(bandwidth.data(), n*n);
        
        auto training_data = df.to_eigen<false, ArrowType, contains_null>(m_variable, m_evidence);
        opencl.copy_to_buffer(training_data->data(), df->num_rows() * n);

        m_lognorm_const = -bandwidth.diagonal().array().log().sum() - 0.5 * n * std::log(2*util::pi<CType>);
    }

}

#endif //PGM_DATASET_CKDE_HPP