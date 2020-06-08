#ifndef PGM_OPENCL_CONFIG_HPP
#define PGM_OPENCL_CONFIG_HPP

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <fstream>
#include <cmath>
#include <arrow/api.h>
#include <CL/cl2.hpp>

namespace opencl {

    template<typename ArrowType>
    struct OpenCL_kernel_traits;

    template<>
    struct OpenCL_kernel_traits<arrow::DoubleType> {
        inline constexpr static const char* logpdf_values_1d = "logpdf_values_1d_double";
        inline constexpr static const char* logpdf_values_1d_matrix = "logpdf_values_1d_matrix_double";
        inline constexpr static const char* max1d = "max1d_double";
        inline constexpr static const char* max_mat_cols_copy = "max_mat_cols_copy_double";
        inline constexpr static const char* max_mat_cols = "max_mat_cols_double";
        inline constexpr static const char* max_mat_cols_single_wg = "max_mat_cols_single_wg_double";
        inline constexpr static const char* sum1d = "sum1d_double";
        inline constexpr static const char* max1d_copy = "max1d_copy_double";
        inline constexpr static const char* sum1d_copy = "sum1d_copy_double";
        inline constexpr static const char* max1d_single_wg = "max1d_single_wg_double";
        inline constexpr static const char* sum1d_single_wg = "sum1d_single_wg_double";
        inline constexpr static const char* logsumexp_coeffs = "logsumexp_coeffs_double";
        inline constexpr static const char* logsumexp_coeffs_mat = "logsumexp_coeffs_mat_double";
        inline constexpr static const char* copy_logpdf_result = "copy_logpdf_result_double";
        inline constexpr static const char* maxwise = "maxwise_double";
        inline constexpr static const char* sum_lse_coefficient = "sum_lse_coefficient_double";
        inline constexpr static const char* finish_lse = "finish_lse_double";
    };

    template<>
    struct OpenCL_kernel_traits<arrow::FloatType> {
        inline constexpr static const char* logpdf_values_1d = "logpdf_values_1d_float";
        inline constexpr static const char* logpdf_values_1d_matrix = "logpdf_values_1d_matrix_float";
        inline constexpr static const char* max1d = "max1d_float";
        inline constexpr static const char* max_mat_cols_copy = "max_mat_cols_copy_float";
        inline constexpr static const char* max_mat_cols = "max_mat_cols_float";
        inline constexpr static const char* max_mat_cols_single_wg = "max_mat_cols_single_wg_float";
        inline constexpr static const char* sum1d = "sum1d_float";
        inline constexpr static const char* max1d_copy = "max1d_copy_float";
        inline constexpr static const char* sum1d_copy = "sum1d_copy_float";
        inline constexpr static const char* max1d_single_wg = "max1d_single_wg_float";
        inline constexpr static const char* sum1d_single_wg = "sum1d_single_wg_float";
        inline constexpr static const char* logsumexp_coeffs = "logsumexp_coeffs_float";
        inline constexpr static const char* logsumexp_coeffs_mat = "logsumexp_coeffs_mat_float";
        inline constexpr static const char* copy_logpdf_result = "copy_logpdf_result_float";
        inline constexpr static const char* maxwise = "maxwise_float";
        inline constexpr static const char* sum_lse_coefficient = "sum_lse_coefficient_float";
        inline constexpr static const char* finish_lse = "finish_lse_float";
    };


    inline constexpr int default_platform_idx = 0;
    inline constexpr int default_device_idx = 0;

    class OpenCLConfig {
    public:
        static OpenCLConfig& get();

        cl::Context& context() { return m_context; }
        cl::Program& program() { return m_program; }

        template<typename T>
        cl::Buffer copy_to_buffer(const T* d, int size);
        template<typename T>
        cl::Buffer new_buffer(int size);

        cl::Kernel kernel(const char* name);
        cl::CommandQueue& queue() { return m_queue; }

        template<typename ArrowType>
        cl::Buffer create_reduction_buffer(int original_length);

        template<typename T>
        void fill_buffer(cl::Buffer& buffer, T value, int length);

        template<typename ArrowType>
        void amax1d(cl::Buffer& input, int input_length, cl::Buffer& output);

        template<typename ArrowType>
        std::pair<cl::Buffer, int> amax_cols(cl::Buffer& input, int input_length, cl::Buffer& output);

        template<typename ArrowType>
        void sum1d(cl::Buffer& input, int input_length, cl::Buffer& output);

        int max_local_size() { return m_max_local_size; }
    private:
        OpenCLConfig() {}
        OpenCLConfig(cl::Context cont, cl::CommandQueue queue, cl::Program program, int max_local_size) 
                                                                    : m_context(cont), 
                                                                      m_queue(queue), 
                                                                      m_program(program),
                                                                      m_max_local_size(max_local_size) {}

        static OpenCLConfig singleton;
        static bool initialized;

        cl::Context m_context;
        cl::CommandQueue m_queue;
        cl::Program m_program;
        int m_max_local_size;
    };

    template<typename T>
    cl::Buffer OpenCLConfig::copy_to_buffer(const T* d, int size) {
        cl_int err_code = CL_SUCCESS;
        cl::Buffer b(m_context, CL_MEM_READ_WRITE,  sizeof(T)*size, NULL, &err_code);

        if (err_code != CL_SUCCESS) {
            throw std::runtime_error("Error creating OpenCL buffer.");
        }

        err_code = m_queue.enqueueWriteBuffer(b, CL_TRUE, 0, sizeof(T)*size, d);

        if (err_code != CL_SUCCESS) {
            throw std::runtime_error("Error copying OpenCL buffer.");
        }

        return std::move(b);
    }

    template<typename T>
    cl::Buffer OpenCLConfig::new_buffer(int size) {
        cl_int err_code = CL_SUCCESS;
        cl::Buffer b(m_context, CL_MEM_READ_WRITE,  sizeof(T)*size, NULL, &err_code);

        if (err_code != CL_SUCCESS) {
            throw std::runtime_error("Error creating OpenCL buffer.");
        }

        return std::move(b);
    }

    template<typename ArrowType>
    cl::Buffer OpenCLConfig::create_reduction_buffer(int original_length) {
        using CType = typename ArrowType::c_type;
        auto num_groups = static_cast<int>(std::ceil(static_cast<double>(original_length) / static_cast<double>(m_max_local_size)));
        auto reduc_buffer = new_buffer<CType>(num_groups);
        return std::move(reduc_buffer);
    }

    template<typename T>
    void OpenCLConfig::fill_buffer(cl::Buffer& buffer, T value, int length) {
        m_queue.enqueueFillBuffer(buffer, value, 0, length);
    }

    void update_reduc_status(int& length, int& num_groups, int& local_size, int& global_size, int max_local_size);

    template<typename ArrowType>
    void OpenCLConfig::amax1d(cl::Buffer& input, int input_length, cl::Buffer& reduc_buffer) {
        auto length = input_length;
        auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(m_max_local_size)));
        auto local_size = (length > m_max_local_size) ? m_max_local_size : length;
        auto global_size = local_size * num_groups;

        auto k_max_copy = kernel(OpenCL_kernel_traits<ArrowType>::max1d_copy);
        k_max_copy.setArg(0, input);
        k_max_copy.setArg(1, static_cast<unsigned_int>(length));
        k_max_copy.setArg(2, cl::Local(local_size));
        k_max_copy.setArg(3, reduc_buffer);

        m_queue.enqueueNDRangeKernel(k_max_copy, cl::NullRange,  cl::NDRange(global_size), cl::NDRange(local_size));

        update_reduc_status(length, num_groups, local_size, global_size, m_max_local_size);

        while (length > m_max_local_size) {

            auto k_max = kernel(OpenCL_kernel_traits<ArrowType>::max1d);
            k_max.setArg(0, reduc_buffer);
            k_max.setArg(1, static_cast<unsigned_int>(length));
            k_max.setArg(2, cl::Local(local_size));

            m_queue.enqueueNDRangeKernel(k_max, cl::NullRange,  cl::NDRange(global_size), cl::NDRange(local_size));
            
            update_reduc_status(length, num_groups, local_size, global_size, m_max_local_size);
        }

        if (length > 1) {
            auto k_max_single_wg = kernel(OpenCL_kernel_traits<ArrowType>::max1d_single_wg);
            k_max_single_wg.setArg(0, reduc_buffer);
            k_max_single_wg.setArg(1, cl::Local(local_size));

            m_queue.enqueueNDRangeKernel(k_max_single_wg, cl::NullRange,  cl::NDRange(global_size), cl::NDRange(local_size));
        }
    }

    template<typename ArrowType>
    void OpenCLConfig::sum1d(cl::Buffer& input, int input_length, cl::Buffer& reduc_buffer) {
        auto length = input_length;
        auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(m_max_local_size)));
        auto local_size = (length > m_max_local_size) ? m_max_local_size : length;
        auto global_size = local_size * num_groups;

        auto k_sum_copy = kernel(OpenCL_kernel_traits<ArrowType>::sum1d_copy);
        k_sum_copy.setArg(0, input);
        k_sum_copy.setArg(1, length);
        k_sum_copy.setArg(2, cl::Local(local_size));
        k_sum_copy.setArg(3, reduc_buffer);

        m_queue.enqueueNDRangeKernel(k_sum_copy, cl::NullRange,  cl::NDRange(global_size), cl::NDRange(local_size));

        update_reduc_status(length, num_groups, local_size, global_size, m_max_local_size);

        while (length > m_max_local_size) {
            auto k_sum = kernel(OpenCL_kernel_traits<ArrowType>::sum1d);
            k_sum.setArg(0, reduc_buffer);
            k_sum.setArg(1, length);
            k_sum.setArg(2, cl::Local(local_size));

            m_queue.enqueueNDRangeKernel(k_sum, cl::NullRange,  cl::NDRange(global_size), cl::NDRange(local_size));
            
            update_reduc_status(length, num_groups, local_size, global_size, m_max_local_size);
        }

        if (length > 1) {
            auto k_sum_single_wg = kernel(OpenCL_kernel_traits<ArrowType>::sum1d_single_wg);
            k_sum_single_wg.setArg(0, reduc_buffer);
            k_sum_single_wg.setArg(1, cl::Local(local_size));

            m_queue.enqueueNDRangeKernel(k_sum_single_wg, cl::NullRange,  cl::NDRange(global_size), cl::NDRange(local_size));
        }
    }

    template<typename ArrowType>
    cl::Buffer OpenCLConfig::amax_cols(cl::Buffer& input, int input_rows, int input_cols) {
        using CType = typename ArrowType::c_type

        auto length = input_rows;
        auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(m_max_local_size)));
        auto local_size = (length > m_max_local_size) ? m_max_local_size : length;
        auto global_size = local_size * num_groups;

        auto reduc_buffer = new_buffer<CType>(num_groups*m);

        auto k_max_mat_cols_copy = kernel(OpenCL_kernel_traits<ArrowType>::max_mat_cols_copy);
        k_max_mat_cols_copy.setArg(0, input);
        k_max_mat_cols_copy.setArg(1, static_cast<unsigned int>(length));
        k_max_mat_cols_copy.setArg(2, cl::Local(local_size));
        k_max_mat_cols_copy.setArg(3, reduc_buffer);
    
        m_queue.enqueueNDRangeKernel(k_max_mat_cols_copy, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));

        update_reduc_status(length, num_groups, local_size, global_size, m_max_local_size);

        while (length > m_max_local_size) {
            auto k_max_mat_cols = kernel(OpenCL_kernel_traits<ArrowType>::max_mat_cols);
            k_max_mat_cols.setArg(0, reduc_buffer);
            k_max_mat_cols.setArg(1, static_cast<unsigned int>(length));
            k_max_mat_cols.setArg(2, cl::Local(local_size));
            
            m_queue.enqueueNDRangeKernel(k_max_mat_cols, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));
            update_reduc_status(length, num_groups, local_size, global_size, m_max_local_size);
        }

        if (length > 1) {
            auto k_max_mat_cols_single_wg = kernel(OpenCL_kernel_traits<ArrowType>::max_mat_cols_single_wg);
            k_max_mat_cols_single_wg.setArg(0, reduc_buffer);
            k_max_mat_cols_single_wg.setArg(1, static_cast<unsigned int>(length));
            k_max_mat_cols_single_wg.setArg(2, cl::Local(local_size));
        }

        int original_num_groups = static_cast<int>(std::ceil(static_cast<double>(input_rows) / static_cast<double>(m_max_local_size)))
        return std::make_pair(reduc_buffer, original_num_groups);
    }

}

// #define CL_HPP_ENABLE_EXCEPTIONS
// #define CL_HPP_TARGET_OPENCL_VERSION 120

#endif //PGM_OPENCL_CONFIG_HPP