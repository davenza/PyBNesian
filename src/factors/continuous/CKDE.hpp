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

    struct UnivariateKDE {
        template<typename ArrowType>
        void static init_logpdf_vec(const cl::Buffer& training_vec, 
                                    const cl::Buffer& test_vec, 
                                    const cl::Buffer& cholesky, 
                                    typename ArrowType::c_type lognorm_const, 
                                    cl::Buffer& output_vec) 
        {
            auto opencl = OpenCLConfig::get();
            auto k_logpdf_values = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values_1d);
            k_logpdf_values.setArg(0, training_vec);
            k_logpdf_values.setArg(1, test_vec);
            k_logpdf_values.setArg(3, cholesky);
            k_logpdf_values.setArg(4, lognorm_const);
            k_logpdf_values.setArg(5, output_vec);
        }

        template<typename ArrowType>
        void static execute_logpdf_vec(int test_index, int training_length) {
            auto opencl = OpenCLConfig::get();
            auto k_logpdf_values = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values_1d);
            k_logpdf_values.setArg(2, static_cast<unsigned int>(test_index));
            auto queue = opencl.queue();
            queue.enqueueNDRangeKernel(k_logpdf_values, cl::NullRange,  cl::NDRange(training_length), cl::NullRange);
        }

        template<typename ArrowType>
        void static execute_logpdf_mat(const cl::Buffer& training_vec, 
                                       int training_length, 
                                       const cl::Buffer& test_vec,
                                       int test_length,
                                       const cl::Buffer& cholesky, 
                                       typename ArrowType::c_type lognorm_const, 
                                       cl::Buffer& output_mat) 
        {
            auto opencl = OpenCLConfig::get();
            auto k_logpdf_values_matrix = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values_1d_matrix);
            k_logpdf_values_matrix.setArg(0, training_vec);
            k_logpdf_values_matrix.setArg(1, static_cast<unsigned int>(training_length));
            k_logpdf_values_matrix.setArg(2, test_vec);
            k_logpdf_values_matrix.setArg(3, cholesky);
            k_logpdf_values_matrix.setArg(4, lognorm_const);
            k_logpdf_values_matrix.setArg(5, output_mat);
            auto queue = opencl.queue();        
            queue.enqueueNDRangeKernel(k_logpdf_values_matrix, cl::NullRange,  cl::NDRange(training_length*test_length), cl::NullRange);
        }

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

        template<typename ArrowType>
        cl::Buffer _logpdf(const DataFrame& df) const;
        template<typename ArrowType>
        cl::Buffer _logpdf_impl(const DataFrame& df) const;
        template<typename ArrowType, typename KDEType>
        cl::Buffer _logpdf_impl_iterate_test(const DataFrame& df) const;
        template<typename ArrowType>
        cl::Buffer _logpdf_impl_iterate_train(const DataFrame& df) const;
        template<typename ArrowType, typename KDEType>
        cl::Buffer _logpdf_impl_iterate_train_vec(const DataFrame& df, cl::Buffer& cache_vec, int m) const;
        template<typename ArrowType, typename KDEType>
        cl::Buffer _logpdf_impl_iterate_train_mat(const DataFrame& df, cl::Buffer& cache_mat, int m) const;

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

    template<typename ArrowType>
    cl::Buffer KDE::_logpdf(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;
        using MatrixType = Matrix<CType, Dynamic, Dynamic>;

        auto logpdf_buffer = _logpdf_impl<ArrowType>(df);
    }

    template<typename ArrowType>
    cl::Buffer KDE::_logpdf_impl(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;
        using MatrixType = Matrix<CType, Dynamic, Dynamic>;


        auto m = df.valid_count(m_variables);
        if (N >= m) {
            if (m_variables.size() == 1)
                return _logpdf_impl_iterate_test<ArrowType, UnivariateKDE>(df);
        } else {
            return _logpdf_impl_iterate_train<ArrowType>(df);
        }
    }


    template<typename ArrowType, typename KDEType>
    cl::Buffer KDE::_logpdf_impl_iterate_test(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;
        auto opencl = OpenCLConfig::get();

        auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
        auto m = test_matrix->rows();
        auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m*m_variables.size());

        auto logpdf_buffer = opencl.new_buffer<CType>(N);
        auto reduction_buffers = opencl.create_reduction1d_buffers<ArrowType>(N);
        auto res = opencl.new_buffer<CType>(m);

        KDEType::template init_logpdf_vec<ArrowType>(m_training, test_buffer, m_H_cholesky, m_lognorm_const, logpdf_buffer);

        for(auto i = 0; i < m; ++i) {
            KDEType::template execute_logpdf_vec<ArrowType>(i, N);
            opencl.logsumexp1d<ArrowType>(logpdf_buffer, N, reduction_buffers, res, i);
        }

        return std::move(res);
    }

    template<typename ArrowType>
    cl::Buffer KDE::_logpdf_impl_iterate_train(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;

        auto opencl = OpenCLConfig::get();

        auto m = df.valid_count(m_variables);
        
        try {
            auto cache_max = opencl.new_buffer<CType>(N * m);
            if (m_variables.size() == 1)
                return _logpdf_impl_iterate_train_mat<ArrowType, UnivariateKDE>(df, cache_max, m);
        } catch(std::runtime_error) {
            if (m > 2*N) {
                auto cache_max = opencl.new_buffer<CType>(m);
                if (m_variables.size() == 1)
                    return _logpdf_impl_iterate_train_vec<ArrowType, UnivariateKDE>(df, cache_max, m);
            } else {
                if (m_variables.size() == 1)
                    return _logpdf_impl_iterate_test<ArrowType, UnivariateKDE>(df);
            }
        }
    }

    template<typename ArrowType, typename KDEType>
    cl::Buffer KDE::_logpdf_impl_iterate_train_vec(const DataFrame& df, cl::Buffer& max_vec, int m) const {
        using CType = typename ArrowType::c_type;

        auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
        auto opencl = OpenCLConfig::get();
        auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m*m_variables.size());

        auto logpdf_buffer = opencl.new_buffer<CType>(m);
        auto res = opencl.new_buffer<CType>(m);

        opencl.fill_buffer(max_vec, std::numeric_limits<CType>::lowest(), m);
        opencl.fill_buffer(res, 0, m);

        auto k_maxwise = opencl.kernel(OpenCL_kernel_traits<ArrowType>::maxwise);
        k_maxwise.setArg(0, max_vec);
        k_maxwise.setArg(1, logpdf_buffer);

        auto k_sum_lse_coefficient = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_lse_coefficient);
        k_sum_lse_coefficient.setArg(0, logpdf_buffer);
        k_sum_lse_coefficient.setArg(1, max_vec);
        k_sum_lse_coefficient.setArg(2, res);

        auto k_finish_lse = opencl.kernel(OpenCL_kernel_traits<ArrowType>::finish_lse);
        k_finish_lse.setArg(0, res);
        k_finish_lse.setArg(1, max_vec);

        auto queue = opencl.queue();

        KDEType::template init_logpdf_vec<ArrowType>(test_buffer, m_training, m_H_cholesky, m_lognorm_const, logpdf_buffer);

        for(unsigned int i = 0; i < N; ++i) {
            KDEType::template execute_logpdf_vec<ArrowType>(i, m);
            queue.enqueueNDRangeKernel(k_maxwise, cl::NullRange,  cl::NDRange(m), cl::NullRange);
        }

        for(unsigned int i = 0; i < N; ++i) {
            KDEType::template execute_logpdf_vec<ArrowType>(i, m);
            queue.enqueueNDRangeKernel(k_sum_lse_coefficient, cl::NullRange,  cl::NDRange(m), cl::NullRange);
        }

        queue.enqueueNDRangeKernel(k_finish_lse, cl::NullRange,  cl::NDRange(m), cl::NullRange);
        return std::move(res);
    }


    template<typename ArrowType, typename KDEType>
    cl::Buffer KDE::_logpdf_impl_iterate_train_mat(const DataFrame& df, cl::Buffer& cache_vec, int m) const {
        using CType = typename ArrowType::c_type;

        auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
        auto opencl = OpenCLConfig::get();
        auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m*m_variables.size());
        
        KDEType::template execute_logpdf_mat<ArrowType>(m_training, N, test_buffer, m, m_H_cholesky, m_lognorm_const, cache_vec);

        auto reduc_buffers = opencl.create_reduction_mat_buffers<ArrowType>(N, m);
        return opencl.logsumexp_cols<ArrowType>(cache_vec, N, m, reduc_buffers);
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