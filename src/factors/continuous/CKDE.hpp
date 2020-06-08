#ifndef PGM_DATASET_CKDE_HPP
#define PGM_DATASET_CKDE_HPP

#include <iostream>

#include <CL/cl2.hpp>
#include <opencl/opencl_config.hpp>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <dataset/dataset.hpp>
#include <util/math_constants.hpp>

namespace py = pybind11;
using dataset::DataFrame;
using Eigen::VectorXd, Eigen::Ref, Eigen::LLT;
using opencl::OpenCLConfig, opencl::OpenCL_kernel_traits;

namespace factors::continuous {
    
    enum class KDEBandwidth {
        SCOTT
    };

    class KDE {
    public:
        KDE(std::vector<std::string> variables) : KDE(variables, KDEBandwidth::SCOTT) {}
        KDE(std::vector<std::string> variables, KDEBandwidth b_selector) : m_variables(variables), 
                                                                            m_bselector(b_selector),
                                                                            m_H_cholesky(),
                                                                            m_training(),
                                                                            m_lognorm_const(0),
                                                                            N(0),
                                                                            m_training_type(arrow::Type::type::NA) {}

        void fit(const DataFrame& df);
        void fit(cl::Buffer cholesky, cl::Buffer training, int training_instances, arrow::Type::type training_type) {
            m_H_cholesky = cholesky;
            m_training = training;
            N = training_instances;
            m_training_type = training_type;
        }

        cl::Buffer logpdf(const DataFrame& df) const;

    private:
        template<typename ArrowType, bool contains_null>
        void _fit(const DataFrame& df);

        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_1d(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_1d_iterate_test(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_1d_iterate_train(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_1d_iterate_train_vec(const DataFrame& df, cl::Buffer& cache_vec, int m) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_1d_iterate_train_mat(const DataFrame& df, cl::Buffer& cache_mat, int m) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_iterate_test(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_iterate_train(const DataFrame& df) const;

        template<typename ArrowType, bool contains_null>
        Matrix<typename ArrowType::c_type, Dynamic, Dynamic> compute_bandwidth(const DataFrame& df) const;

        std::vector<std::string> m_variables;
        KDEBandwidth m_bselector;
        cl::Buffer m_H_cholesky;
        cl::Buffer m_training;
        double m_lognorm_const;
        int N;
        arrow::Type::type m_training_type;
    };

    template<typename ArrowType, bool contains_null>
    Matrix<typename ArrowType::c_type, Dynamic, Dynamic> KDE::compute_bandwidth(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;
        auto cov = df.cov<ArrowType, contains_null>(m_variables);
        auto bandwidth = std::pow(static_cast<CType>(df->num_rows()), -1 / static_cast<CType>(m_variables.size() + 4)) * (*cov) ;

        return bandwidth;
    }

    template<typename ArrowType, bool contains_null>
    void KDE::_fit(const DataFrame& df) {
        using CType = typename ArrowType::c_type;
        using MatrixType = Matrix<CType, Dynamic, Dynamic>;

        auto d = m_variables.size();
        N = df->num_rows();

        auto bandwidth = compute_bandwidth<ArrowType, contains_null>(df);

        LLT<Ref<MatrixType>> llt_cov(bandwidth);

        auto opencl = OpenCLConfig::get();
        m_H_cholesky = opencl.copy_to_buffer(bandwidth.data(), d*d);
        
        auto training_data = df.to_eigen<false, ArrowType, contains_null>(m_variables);
        m_training = opencl.copy_to_buffer(training_data->data(), N * d);

        m_lognorm_const = -bandwidth.diagonal().array().log().sum() - 0.5 * d * std::log(2*util::pi<CType>);
    }

    template<typename ArrowType, bool contains_null>
    cl::Buffer KDE::_logpdf(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;
        using MatrixType = Matrix<CType, Dynamic, Dynamic>;

        auto logpdf_buffer = _logpdf_impl<ArrowType, contains_null>(df);
    }

    template<typename ArrowType, bool contains_null>
    cl::Buffer KDE::_logpdf_impl(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;
        using MatrixType = Matrix<CType, Dynamic, Dynamic>;

        if (m_variables.size() == 1)
            return _logpdf_impl_1d<ArrowType, contains_null>(df);

        auto m = df.valid_count(m_variables);
        if (N >= m) {
            return _logpdf_impl_iterate_test<ArrowType, contains_null>(df);
        } else {
            return _logpdf_impl_iterate_train<ArrowType, contains_null>(df);
        }
    }

    template<typename ArrowType, bool contains_null>
    cl::Buffer KDE::_logpdf_impl_1d(const DataFrame& df) const {
        auto m = df.valid_count(m_variables[0]);
        if (N >= m) {
            return _logpdf_impl_1d_iterate_test<ArrowType, contains_null>(df);
        } else {
            return _logpdf_impl_1d_iterate_train<ArrowType, contains_null>(df);
        }
    }

    template<typename ArrowType, bool contains_null>
    cl::Buffer KDE::_logpdf_impl_1d_iterate_test(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;
        auto opencl = OpenCLConfig::get();

        auto test_matrix = df.to_eigen<false, ArrowType, false>(m_variables[0]);
        auto m = test_matrix->rows();
        auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m);

        auto logpdf_buffer = opencl.new_buffer<CType>(m);
        auto max_buffer = opencl.create_reduction_buffer<ArrowType>(m);
        auto sum_buffer = opencl.create_reduction_buffer<ArrowType>(m);
        auto res = opencl.new_buffer<CType>(m);

        auto k_logpdf_values = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values_1d);
        k_logpdf_values.setArg(0, m_training);
        k_logpdf_values.setArg(1, test_buffer);
        k_logpdf_values.setArg(3, m_H_cholesky);
        k_logpdf_values.setArg(4, m_lognorm_const);
        k_logpdf_values.setArg(5, logpdf_buffer);

        auto k_logsumexp_coeffs = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logsumexp_coeffs);
        k_logsumexp_coeffs.setArg(0, logpdf_buffer);
        k_logsumexp_coeffs.setArg(1, max_buffer);

        auto k_copy_logpdf_result = opencl.kernel(OpenCL_kernel_traits<ArrowType>::copy_logpdf_result);
        k_copy_logpdf_result.setArg(0, sum_buffer);
        k_copy_logpdf_result.setArg(1, max_buffer);
        k_copy_logpdf_result.setArg(2, res);

        auto queue = opencl.queue();

        for(unsigned int i = 0; i < m; ++i) {
            k_logpdf_values.setArg(2, i);
            queue.enqueueNDRangeKernel(k_logpdf_values, cl::NullRange,  cl::NDRange(m),cl::NullRange);
            opencl.amax1d<ArrowType>(logpdf_buffer, m, max_buffer);
            queue.enqueueNDRangeKernel(k_logsumexp_coeffs, cl::NullRange,  cl::NDRange(m),cl::NullRange);
            opencl.sum1d<ArrowType>(logpdf_buffer, m, sum_buffer);
            k_copy_logpdf_result.setArg(3, i);
        }

        return res;
    }

    template<typename ArrowType, bool contains_null>
    cl::Buffer KDE::_logpdf_impl_1d_iterate_train(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;

        auto opencl = OpenCLConfig::get();

        int64_t m;
        if constexpr(contains_null) {
            m = df.valid_count(m_variables[0]);
        } else {
            m = df->num_rows();
        }
        
        try {
            auto cache_max = opencl.new_buffer<CType>(N * m);
            return _logpdf_impl_1d_iterate_train_mat<ArrowType, contains_null>(df, cache_max, m);
        } catch(std::runtime_error) {
            if (m > 2*N) {
                auto cache_max = opencl.new_buffer<CType>(m);
                return _logpdf_impl_1d_iterate_train_vec<ArrowType, contains_null>(df, cache_max, m);
            } else {
                return _logpdf_impl_1d_iterate_test<ArrowType, contains_null>(df);
            }
        }

    }

    template<typename ArrowType, bool contains_null>
    cl::Buffer KDE::_logpdf_impl_1d_iterate_train_vec(const DataFrame& df, cl::Buffer& cache_vec, int m) const {
        using CType = typename ArrowType::c_type;

        auto test_matrix = df.to_eigen<false, ArrowType, contains_null>(m_variables[0]);
        auto opencl = OpenCLConfig::get();
        auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m);

        auto logpdf_buffer = opencl.new_buffer<CType>(m);
        auto res = opencl.new_buffer<CType>(m);

        opencl.fill_buffer(cache_vec, std::numeric_limits<CType>::lowest(), m);
        opencl.fill_buffer(res, 0, m);

        auto k_logpdf_values = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values_1d);
        k_logpdf_values.setArg(0, test_buffer);
        k_logpdf_values.setArg(1, m_training);
        k_logpdf_values.setArg(3, m_H_cholesky);
        k_logpdf_values.setArg(4, m_lognorm_const);
        k_logpdf_values.setArg(5, logpdf_buffer);

        auto k_maxwise = opencl.kernel(OpenCL_kernel_traits<ArrowType>::maxwise);
        k_maxwise.setArg(0, cache_vec);
        k_maxwise.setArg(1, logpdf_buffer);

        auto k_sum_lse_coefficient = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_lse_coefficient);
        k_sum_lse_coefficient.setArg(0, logpdf_buffer);
        k_sum_lse_coefficient.setArg(1, cache_vec);
        k_sum_lse_coefficient.setArg(2, res);

        auto k_finish_lse = opencl.kernel(OpenCL_kernel_traits<ArrowType>::finish_lse);
        k_finish_lse.setArg(0, res);
        k_finish_lse.setArg(1, cache_vec);

        auto queue = opencl.queue();

        for(unsigned int i = 0; i < m; ++i) {
            k_logpdf_values.setArg(2, i);
            queue.enqueueNDRangeKernel(k_logpdf_values, cl::NullRange,  cl::NDRange(m), cl::NullRange);
            queue.enqueueNDRangeKernel(k_maxwise, cl::NullRange,  cl::NDRange(m), cl::NullRange);
        }

        for(unsigned int i = 0; i < m; ++i) {
            k_logpdf_values.setArg(2, i);
            queue.enqueueNDRangeKernel(k_logpdf_values, cl::NullRange,  cl::NDRange(m), cl::NullRange);
            queue.enqueueNDRangeKernel(k_sum_lse_coefficient, cl::NullRange,  cl::NDRange(m), cl::NullRange);
        }

        queue.enqueueNDRangeKernel(k_finish_lse, cl::NullRange,  cl::NDRange(m), cl::NullRange);

        return res;
    }


    template<typename ArrowType, bool contains_null>
    cl::Buffer KDE::_logpdf_impl_1d_iterate_train_mat(const DataFrame& df, cl::Buffer& cache_vec, int m) const {
        using CType = typename ArrowType::c_type;

        auto test_matrix = df.to_eigen<false, ArrowType, contains_null>(m_variables[0]);
        auto opencl = OpenCLConfig::get();
        auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m);

        auto k_logpdf_values_matrix = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values_1d_matrix);
        k_logpdf_values_matrix.setArg(0, m_training);
        k_logpdf_values_matrix.setArg(1, static_cast<unsigned int>(N));
        k_logpdf_values_matrix.setArg(2, test_buffer);
        k_logpdf_values_matrix.setArg(3, m_H_cholesky);
        k_logpdf_values_matrix.setArg(4, m_lognorm_const);
        k_logpdf_values_matrix.setArg(5, cache_vec);

        queue = opencl.queue();
        
        queue.enqueueNDRangeKernel(k_logpdf_values_matrix, cl::NullRange,  cl::NDRange(N*m), cl::NullRange);

        auto [max_buffer, max_rows] = opencl.amax_cols<ArrowType>(cache_vec, N, m);
        auto logsumexp_coeffs_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logsumexp_coeffs_mat);
        logsumexp_coeffs_mat.setArg(0, cache_vec);
        logsumexp_coeffs_mat.setArg(1, N);
        logsumexp_coeffs_mat.setArg(2, max_buffer);
        logsumexp_coeffs_mat.setArg(3, max_rows);
        queue.enqueueNDRangeKernel(logsumexp_coeffs_mat, cl::NullRange,  cl::NDRange(N*m), cl::NullRange);
    }

    
    template<typename ArrowType, bool contains_null>
    cl::Buffer KDE::_logpdf_impl_iterate_test(const DataFrame& df) const {
        // using CType = typename ArrowType::c_type;
        // using MatrixType = Matrix<CType, Dynamic, Dynamic>;

        // auto test_matrix = df.to_eigen<false, ArrowType>(m_variable, m_evidence);
        // auto m = test_matrix->rows();
        // auto d = 1 + m_evidence.size();

        // auto opencl = OpenCLConfig::get();
        // opencl.copy_to_buffer(test_matrix->data(), m*d)

        // for (int i = 0; i < m; ++i) {

        // }


        // auto m = df->num_rows();
        // auto opencl = OpenCLConfig::get();

        // auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m*d);

    }

    template<typename ArrowType, bool contains_null>
    cl::Buffer KDE::_logpdf_impl_iterate_train(const DataFrame& df) const {

    }


    class CKDE {
    public:
        CKDE(const std::string variable, const std::vector<std::string> evidence) : CKDE(variable, evidence, KDEBandwidth::SCOTT) {}
        CKDE(const std::string variable, const std::vector<std::string> evidence, KDEBandwidth b_selector) 
                                                                                  : m_variable(variable), 
                                                                                    m_evidence(evidence), 
                                                                                    m_bselector(b_selector),
                                                                                    m_H_cholesky(),
                                                                                    m_training(),
                                                                                    m_lognorm_const(0),
                                                                                    m_n(0),
                                                                                    m_training_type(arrow::Type::type::NA) {}

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
        VectorXd _logpdf(const DataFrame& df) const;

        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_1d(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_1d_iterate_test(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_1d_iterate_train(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_1d_iterate_train_vec(const DataFrame& df, cl::Buffer& cache_vec, int m) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_1d_iterate_train_mat(const DataFrame& df, cl::Buffer& cache_mat, int m) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_iterate_test(const DataFrame& df) const;
        template<typename ArrowType, bool contains_null>
        cl::Buffer _logpdf_impl_iterate_train(const DataFrame& df) const;
        


        std::string m_variable;
        std::vector<std::string> m_evidence;
        KDEBandwidth m_bselector;
        cl::Buffer m_H_cholesky;
        cl::Buffer m_training;
        double m_lognorm_const;
        int m_n;
        arrow::Type::type m_training_type;
    };



}

#endif //PGM_DATASET_CKDE_HPP