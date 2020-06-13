#ifndef PGM_DATASET_CKDE_HPP
#define PGM_DATASET_CKDE_HPP

#include <iostream>

#include <CL/cl2.hpp>
#include <opencl/opencl_config.hpp>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <pybind11/eigen.h>
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
                                    const unsigned int,
                                    const cl::Buffer& test_vec,
                                    const unsigned int,
                                    const unsigned int,
                                    const cl::Buffer& cholesky, 
                                    const typename ArrowType::c_type lognorm_const,
                                    cl::Buffer&,
                                    cl::Buffer& output_vec);

        template<typename ArrowType>
        void static execute_logpdf_vec(const unsigned int test_index, const unsigned int training_length, const unsigned int);
        
        template<typename ArrowType>
        void static execute_logpdf_mat(const cl::Buffer& training_vec, 
                                       const unsigned int training_length,
                                       const cl::Buffer& test_vec,
                                       const unsigned int test_length,
                                       const unsigned int,
                                       const cl::Buffer& cholesky, 
                                       const typename ArrowType::c_type lognorm_const,
                                       cl::Buffer&,
                                       cl::Buffer& output_mat);
    };

    template<typename ArrowType>
    void UnivariateKDE::init_logpdf_vec(const cl::Buffer& training_vec,
                                const unsigned int,
                                const cl::Buffer& test_vec,
                                const unsigned int,
                                const unsigned int,
                                const cl::Buffer& cholesky, 
                                const typename ArrowType::c_type lognorm_const,
                                cl::Buffer&,
                                cl::Buffer& output_vec) 
    {
        auto& opencl = OpenCLConfig::get();
        auto k_logpdf_values = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values_1d);
        k_logpdf_values.setArg(0, training_vec);
        k_logpdf_values.setArg(1, test_vec);
        k_logpdf_values.setArg(3, cholesky);
        k_logpdf_values.setArg(4, lognorm_const);
        k_logpdf_values.setArg(5, output_vec);
    }

    template<typename ArrowType>
    void UnivariateKDE::execute_logpdf_vec(const unsigned int test_index, const unsigned int training_length, const unsigned int) {
        auto& opencl = OpenCLConfig::get();
        auto k_logpdf_values = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values_1d);
        k_logpdf_values.setArg(2, test_index);
        auto queue = opencl.queue();
        queue.enqueueNDRangeKernel(k_logpdf_values, cl::NullRange,  cl::NDRange(training_length), cl::NullRange);
    }

    template<typename ArrowType>
    void UnivariateKDE::execute_logpdf_mat(const cl::Buffer& training_vec, 
                                    const unsigned int training_length,
                                    const cl::Buffer& test_vec,
                                    const unsigned int test_length,
                                    const unsigned int,
                                    const cl::Buffer& cholesky, 
                                    const typename ArrowType::c_type lognorm_const,
                                    cl::Buffer&,
                                    cl::Buffer& output_mat) 
    {
        auto& opencl = OpenCLConfig::get();
        auto k_logpdf_values_matrix = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values_1d_matrix);
        k_logpdf_values_matrix.setArg(0, training_vec);
        k_logpdf_values_matrix.setArg(1, training_length);
        k_logpdf_values_matrix.setArg(2, test_vec);
        k_logpdf_values_matrix.setArg(3, cholesky);
        k_logpdf_values_matrix.setArg(4, lognorm_const);
        k_logpdf_values_matrix.setArg(5, output_mat);
        auto queue = opencl.queue();        
        queue.enqueueNDRangeKernel(k_logpdf_values_matrix, cl::NullRange,  cl::NDRange(training_length*test_length), cl::NullRange);
    }

    struct MultivariateKDE {
        template<typename ArrowType>
        void static init_logpdf_vec(const cl::Buffer& training_mat,
                                    const unsigned int training_rows,
                                    const cl::Buffer& test_mat,
                                    const unsigned int test_rows,
                                    const unsigned int matrices_cols,
                                    const cl::Buffer& cholesky, 
                                    const typename ArrowType::c_type lognorm_const, 
                                    cl::Buffer& tmp_mat,
                                    cl::Buffer& output_vec);

        template<typename ArrowType>
        void static execute_logpdf_vec(const unsigned int test_index, const unsigned int training_rows, const unsigned int cols);

        template<typename ArrowType>
        void static execute_logpdf_mat(const cl::Buffer& training_mat, 
                                       const unsigned int training_rows, 
                                       const cl::Buffer& test_mat,
                                       const unsigned int test_rows, 
                                       const unsigned int matrices_cols,
                                       const cl::Buffer& cholesky, 
                                       const typename ArrowType::c_type lognorm_const,
                                       cl::Buffer& tmp_mat,
                                       cl::Buffer& output_mat);
    };

    template<typename ArrowType>
    void MultivariateKDE::init_logpdf_vec(const cl::Buffer& training_mat,
                                    const unsigned int training_rows,
                                    const cl::Buffer& test_mat,
                                    const unsigned int test_rows,
                                    const unsigned int matrices_cols,
                                    const cl::Buffer& cholesky, 
                                    const typename ArrowType::c_type lognorm_const, 
                                    cl::Buffer& tmp_mat,
                                    cl::Buffer& output_vec) 
    {
        auto& opencl = OpenCLConfig::get();
        auto k_substract = opencl.kernel(OpenCL_kernel_traits<ArrowType>::substract_matrix);
        k_substract.setArg(0, training_mat);
        k_substract.setArg(1, training_rows);
        k_substract.setArg(2, test_mat);
        k_substract.setArg(3, test_rows);
        k_substract.setArg(5, tmp_mat);

        auto k_solve = opencl.kernel(OpenCL_kernel_traits<ArrowType>::solve);
        k_solve.setArg(0, tmp_mat);
        k_solve.setArg(1, training_rows);
        k_solve.setArg(2, matrices_cols);
        k_solve.setArg(3, cholesky);

        auto k_square = opencl.kernel(OpenCL_kernel_traits<ArrowType>::square);
        k_square.setArg(0, tmp_mat);
        
        auto k_logpdf_values = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values);
        k_logpdf_values.setArg(0, tmp_mat);
        k_logpdf_values.setArg(1, matrices_cols);
        k_logpdf_values.setArg(2, output_vec);
        k_logpdf_values.setArg(3, lognorm_const);
    }
            
    template<typename ArrowType>
    void MultivariateKDE::execute_logpdf_vec(const unsigned int test_index, const unsigned int training_rows, const unsigned int cols) {
        auto& opencl = OpenCLConfig::get();
        auto k_substract = opencl.kernel(OpenCL_kernel_traits<ArrowType>::substract_matrix);
        k_substract.setArg(4, test_index);

        auto queue = opencl.queue();
        queue.enqueueNDRangeKernel(k_substract, cl::NullRange,  cl::NDRange(training_rows*cols), cl::NullRange);
        
        auto k_solve = opencl.kernel(OpenCL_kernel_traits<ArrowType>::solve);
        queue.enqueueNDRangeKernel(k_solve, cl::NullRange,  cl::NDRange(training_rows), cl::NullRange);

        auto k_square = opencl.kernel(OpenCL_kernel_traits<ArrowType>::square);
        queue.enqueueNDRangeKernel(k_square, cl::NullRange,  cl::NDRange(training_rows*cols), cl::NullRange);

        auto k_logpdf_values = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values);
        queue.enqueueNDRangeKernel(k_logpdf_values, cl::NullRange,  cl::NDRange(training_rows), cl::NullRange);
    }

    template<typename ArrowType>
    void MultivariateKDE::execute_logpdf_mat(const cl::Buffer& training_mat, 
                                    const unsigned int training_rows, 
                                    const cl::Buffer& test_mat,
                                    const unsigned int test_rows, 
                                    const unsigned int matrices_cols,
                                    const cl::Buffer& cholesky, 
                                    const typename ArrowType::c_type lognorm_const,
                                    cl::Buffer& tmp_mat,
                                    cl::Buffer& output_mat) 
    {
        auto& opencl = OpenCLConfig::get();
        auto k_substract = opencl.kernel(OpenCL_kernel_traits<ArrowType>::substract_matrix);
        k_substract.setArg(0, test_mat);
        k_substract.setArg(1, test_rows);
        k_substract.setArg(2, training_mat);
        k_substract.setArg(3, training_rows);
        k_substract.setArg(5, tmp_mat);

        auto k_solve = opencl.kernel(OpenCL_kernel_traits<ArrowType>::solve);
        k_solve.setArg(0, tmp_mat);
        k_solve.setArg(1, test_rows);
        k_solve.setArg(2, matrices_cols);
        k_solve.setArg(3, cholesky);

        auto k_square = opencl.kernel(OpenCL_kernel_traits<ArrowType>::square);
        k_square.setArg(0, tmp_mat);
        
        auto k_logpdf_values_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logpdf_values_mat);
        k_logpdf_values_mat.setArg(0, tmp_mat);
        k_logpdf_values_mat.setArg(1, matrices_cols);
        k_logpdf_values_mat.setArg(2, output_mat);
        k_logpdf_values_mat.setArg(3, training_rows);
        k_logpdf_values_mat.setArg(5, lognorm_const);

        auto queue = opencl.queue();
        for (unsigned int i = 0; i < training_rows; ++i) {
            k_substract.setArg(4, i);
            queue.enqueueNDRangeKernel(k_substract, cl::NullRange,  cl::NDRange(test_rows*matrices_cols), cl::NullRange);
            queue.enqueueNDRangeKernel(k_solve, cl::NullRange,  cl::NDRange(test_rows), cl::NullRange);
            queue.enqueueNDRangeKernel(k_square, cl::NullRange,  cl::NDRange(test_rows*matrices_cols), cl::NullRange);
            k_logpdf_values_mat.setArg(4, i);
            queue.enqueueNDRangeKernel(k_logpdf_values_mat, cl::NullRange,  cl::NDRange(test_rows), cl::NullRange);
        }
    }



    class KDE {
    public:

        KDE() : m_variables(), 
                m_bselector(KDEBandwidth::SCOTT), 
                m_H_cholesky(), 
                m_training(), 
                m_lognorm_const(0), 
                N(0), 
                m_training_type(arrow::Type::type::NA) {}

        KDE(std::vector<std::string> variables) : KDE(variables, KDEBandwidth::SCOTT) {
            if (m_variables.empty()) {
                throw std::invalid_argument("Cannot create a KDE model with 0 variables");
            }
        }

        KDE(std::vector<std::string> variables, KDEBandwidth b_selector) : m_variables(variables), 
                                                                            m_bselector(b_selector),
                                                                            m_H_cholesky(),
                                                                            m_training(),
                                                                            m_lognorm_const(0),
                                                                            N(0),
                                                                            m_training_type(arrow::Type::type::NA) {
            if (m_variables.empty()) {
                throw std::invalid_argument("Cannot create a KDE model with 0 variables");
            }
        }

        void fit(py::handle pyobject);
        void fit(const DataFrame& df);

        template<typename ArrowType, typename EigenMatrix>
        void fit(EigenMatrix bandwidth, cl::Buffer training_data, int training_instances);

        VectorXd logpdf(py::handle pyobject) const;
        VectorXd logpdf(const DataFrame& df) const;

    private:
        template<typename ArrowType, bool contains_null>
        void _fit(const DataFrame& df);

        template<typename ArrowType>
        VectorXd _logpdf(const DataFrame& df) const;
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
        void compute_bandwidth(const DataFrame& df, std::vector<std::string>& variables);

        std::vector<std::string> m_variables;
        KDEBandwidth m_bselector;
        MatrixXd m_bandwidth;
        cl::Buffer m_H_cholesky;
        cl::Buffer m_training;
        double m_lognorm_const;
        int N;
        arrow::Type::type m_training_type;
    };

    template<typename ArrowType, bool contains_null>
    void KDE::compute_bandwidth(const DataFrame& df, std::vector<std::string>& variables) {
        using CType = typename ArrowType::c_type;
        auto cov = df.cov<ArrowType, contains_null>(variables);

        if constexpr(std::is_same_v<ArrowType, arrow::DoubleType>) {
            m_bandwidth = std::pow(static_cast<CType>(df.valid_count(variables)), -2 / static_cast<CType>(variables.size() + 4)) * (*cov);
        } else {
            m_bandwidth = std::pow(static_cast<CType>(df.valid_count(variables)), -2 / static_cast<CType>(variables.size() + 4)) * 
                                (cov->template cast<double>());
        }
    }

    template<typename ArrowType, bool contains_null>
    void KDE::_fit(const DataFrame& df) {
        using CType = typename ArrowType::c_type;

        auto d = m_variables.size();

        compute_bandwidth<ArrowType, contains_null>(df, m_variables);

        auto llt_cov = m_bandwidth.llt();
        auto llt_matrix = llt_cov.matrixLLT();

        auto& opencl = OpenCLConfig::get();
        m_H_cholesky = opencl.copy_to_buffer(llt_matrix.data(), d*d);
        
        auto training_data = df.to_eigen<false, ArrowType, contains_null>(m_variables);
        N = training_data->rows();
        m_training = opencl.copy_to_buffer(training_data->data(), N * d);

        m_lognorm_const = -llt_matrix.diagonal().array().log().sum() 
                          - 0.5 * d * std::log(2*util::pi<CType>)
                          - std::log(N);
    }

    template<typename ArrowType, typename EigenMatrix>
    void KDE::fit(EigenMatrix bandwidth, cl::Buffer training_data, int training_instances) {
        using CType = typename ArrowType::c_type;
        if (bandwidth.rows() != bandwidth.cols() != m_variables.size()) {
            throw std::invalid_argument("Bandwidth matrix must be a square matrix with dimensionality " + std::to_string(m_variables.size()));
        }
        auto d = m_variables.size();

        auto llt_cov = bandwidth.llt();
        auto& opencl = OpenCLConfig::get();
        m_H_cholesky = opencl.copy_to_buffer(llt_cov.data(), d*d);
        m_training = training_data;
        N = training_instances;
        m_lognorm_const = -llt_cov.diagonal().array().log().sum() 
                          - 0.5 * d * std::log(2*util::pi<CType>)
                          - std::log(N);
    }

    template<typename ArrowType>
    VectorXd KDE::_logpdf(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;
        using VectorType = Matrix<CType, Dynamic, 1>;

        auto logpdf_buffer = _logpdf_impl<ArrowType>(df);
        auto& opencl = OpenCLConfig::get();
        
        if (df.null_count(m_variables) == 0) {
            VectorType read_data(df->num_rows());
            opencl.read_from_buffer(read_data.data(), logpdf_buffer, df->num_rows());
            if constexpr (!std::is_same_v<typename ArrowType::c_type, double>)
                return read_data.template cast<double>();
            else
                return read_data;
        } else {
            auto m = df.valid_count(m_variables);
            VectorType read_data(m);
            auto bitmap = df.combined_bitmap(m_variables);
            auto bitmap_data = bitmap->data();

            opencl.read_from_buffer(read_data.data(), logpdf_buffer, m);

            VectorXd res(df->num_rows());

            for (int i = 0, k = 0; i < df->num_rows(); ++i) {
                if(arrow::BitUtil::GetBit(bitmap_data, i)) {
                    res(i) = static_cast<double>(read_data[k++]);
                } else {
                    res(i) = util::nan<double>;
                }
            }

            return res;
        }
    }

    template<typename ArrowType>
    cl::Buffer KDE::_logpdf_impl(const DataFrame& df) const {
        auto m = df.valid_count(m_variables);
        if (N >= m) {
            if (m_variables.size() == 1)
                return _logpdf_impl_iterate_test<ArrowType, UnivariateKDE>(df);
            else
                return _logpdf_impl_iterate_test<ArrowType, MultivariateKDE>(df);
        } else {
            return _logpdf_impl_iterate_train<ArrowType>(df);
        }
    }


    template<typename ArrowType, typename KDEType>
    cl::Buffer KDE::_logpdf_impl_iterate_test(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;
        auto& opencl = OpenCLConfig::get();

        auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
        auto m = test_matrix->rows();
        auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m*m_variables.size());

        cl::Buffer tmp_mat_buffer;
        if constexpr(std::is_same_v<KDEType, MultivariateKDE>) {
            tmp_mat_buffer = opencl.new_buffer<CType>(N*m_variables.size());
        }

        auto logpdf_buffer = opencl.new_buffer<CType>(N);
        auto reduction_buffers = opencl.create_reduction1d_buffers<ArrowType>(N);
        auto res = opencl.new_buffer<CType>(m);

        KDEType::template init_logpdf_vec<ArrowType>(m_training, N, test_buffer, m, m_variables.size(), 
                                                     m_H_cholesky, m_lognorm_const, tmp_mat_buffer, logpdf_buffer);

        for(auto i = 0; i < m; ++i) {
            KDEType::template execute_logpdf_vec<ArrowType>(i, N, m_variables.size());
            opencl.logsumexp1d<ArrowType>(logpdf_buffer, N, reduction_buffers, res, i);
        }

        return std::move(res);
    }

    template<typename ArrowType>
    cl::Buffer KDE::_logpdf_impl_iterate_train(const DataFrame& df) const {
        using CType = typename ArrowType::c_type;

        auto& opencl = OpenCLConfig::get();

        auto m = df.valid_count(m_variables);
        
        try {
            auto cache_max = opencl.new_buffer<CType>(N * m);
            if (m_variables.size() == 1)
                return _logpdf_impl_iterate_train_mat<ArrowType, UnivariateKDE>(df, cache_max, m);
            else
                return _logpdf_impl_iterate_train_mat<ArrowType, MultivariateKDE>(df, cache_max, m);
        } catch(std::runtime_error) {
            if (m > 2*N) {
                auto cache_max = opencl.new_buffer<CType>(m);
                if (m_variables.size() == 1)
                    return _logpdf_impl_iterate_train_vec<ArrowType, UnivariateKDE>(df, cache_max, m);
                else
                    return _logpdf_impl_iterate_train_vec<ArrowType, MultivariateKDE>(df, cache_max, m);
            } else {
                if (m_variables.size() == 1)
                    return _logpdf_impl_iterate_test<ArrowType, UnivariateKDE>(df);
                else
                    return _logpdf_impl_iterate_test<ArrowType, MultivariateKDE>(df);
            }
        }
    }

    template<typename ArrowType, typename KDEType>
    cl::Buffer KDE::_logpdf_impl_iterate_train_vec(const DataFrame& df, cl::Buffer& max_vec, int m) const {
        using CType = typename ArrowType::c_type;

        auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
        auto& opencl = OpenCLConfig::get();
        auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m*m_variables.size());

        auto logpdf_buffer = opencl.new_buffer<CType>(m);
        auto res = opencl.new_buffer<CType>(m);

        opencl.fill_buffer(max_vec, std::numeric_limits<CType>::lowest(), m);
        opencl.fill_buffer(res, static_cast<CType>(0), m);

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

        cl::Buffer tmp_mat_buffer;
        if constexpr(std::is_same_v<KDEType, MultivariateKDE>) {
            tmp_mat_buffer = opencl.new_buffer<CType>(m*m_variables.size());
        }

        KDEType::template init_logpdf_vec<ArrowType>(test_buffer, m, m_training, N, m_variables.size(), 
                                                    m_H_cholesky, m_lognorm_const, tmp_mat_buffer, logpdf_buffer);

        auto queue = opencl.queue();

        for(int i = 0; i < N; ++i) {
            KDEType::template execute_logpdf_vec<ArrowType>(i, m, m_variables.size());
            queue.enqueueNDRangeKernel(k_maxwise, cl::NullRange,  cl::NDRange(m), cl::NullRange);
        }

        for(int i = 0; i < N; ++i) {
            KDEType::template execute_logpdf_vec<ArrowType>(i, m, m_variables.size());
            queue.enqueueNDRangeKernel(k_sum_lse_coefficient, cl::NullRange,  cl::NDRange(m), cl::NullRange);
        }

        queue.enqueueNDRangeKernel(k_finish_lse, cl::NullRange,  cl::NDRange(m), cl::NullRange);
        return std::move(res);
    }

    template<typename ArrowType, typename KDEType>
    cl::Buffer KDE::_logpdf_impl_iterate_train_mat(const DataFrame& df, cl::Buffer& cache_mat, int m) const {
        using CType = typename ArrowType::c_type;

        auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
        auto& opencl = OpenCLConfig::get();
        auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m*m_variables.size());

        cl::Buffer tmp_mat_buffer;
        if constexpr(std::is_same_v<KDEType, MultivariateKDE>) {
            tmp_mat_buffer = opencl.new_buffer<CType>(m*m_variables.size());
        }

        KDEType::template execute_logpdf_mat<ArrowType>(m_training, N, test_buffer, m, m_variables.size(), 
                                                        m_H_cholesky, m_lognorm_const, tmp_mat_buffer, cache_mat);

        auto reduc_buffers = opencl.create_reduction_mat_buffers<ArrowType>(N, m);
        return opencl.logsumexp_cols<ArrowType>(cache_mat, N, m, reduc_buffers);
    }

    class CKDE {
    public:
        CKDE(const std::string variable, const std::vector<std::string> evidence) : CKDE(variable, evidence, KDEBandwidth::SCOTT) {}
        CKDE(const std::string variable, const std::vector<std::string> evidence, KDEBandwidth b_selector) 
                                                                                  : m_variable(variable), 
                                                                                    m_evidence(evidence),
                                                                                    m_variables(),
                                                                                    m_bselector(b_selector),
                                                                                    m_training_type(arrow::Type::type::NA),
                                                                                    m_joint(),
                                                                                    m_marg() {
            
            m_variables.reserve(evidence.size() + 1);
            m_variables.push_back(variable);
            for (auto it = evidence.begin(); it != evidence.end(); ++it) {
                m_variables.push_back(*it);
            }
        }

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

        


        std::string m_variable;
        std::vector<std::string> m_evidence;
        std::vector<std::string> m_variables;
        KDEBandwidth m_bselector;
        arrow::Type::type m_training_type;
        KDE m_joint;
        KDE m_marg;
    };

    template<typename ArrowType, bool contains_null>
    void CKDE::_fit(const DataFrame& df) {
        
        m_joint = KDE(m_variables, m_bselector);
        m_joint.fit(df);
    }


}

#endif //PGM_DATASET_CKDE_HPP