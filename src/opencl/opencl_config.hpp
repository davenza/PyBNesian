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
        inline constexpr static const char* max_mat_cols = "max_mat_cols_double";
        inline constexpr static const char* sum1d = "sum1d_double";
        inline constexpr static const char* sum_mat_cols = "sum_mat_cols_double";
        inline constexpr static const char* logsumexp_coeffs = "logsumexp_coeffs_double";
        inline constexpr static const char* logsumexp_coeffs_mat = "logsumexp_coeffs_mat_double";
        inline constexpr static const char* copy_logpdf_result = "copy_logpdf_result_double";
        inline constexpr static const char* maxwise = "maxwise_double";
        inline constexpr static const char* sum_lse_coefficient = "sum_lse_coefficient_double";
        inline constexpr static const char* finish_lse = "finish_lse_double";
        inline constexpr static const char* substract_matrix = "substract_matrix_double";
        inline constexpr static const char* solve = "solve_double";
        inline constexpr static const char* square = "square_double";
        inline constexpr static const char* logpdf_values = "logpdf_values_double";
        inline constexpr static const char* logpdf_values_mat = "logpdf_values_mat_double";
    };

    template<>
    struct OpenCL_kernel_traits<arrow::FloatType> {
        inline constexpr static const char* logpdf_values_1d = "logpdf_values_1d_float";
        inline constexpr static const char* logpdf_values_1d_matrix = "logpdf_values_1d_matrix_float";
        inline constexpr static const char* max1d = "max1d_float";
        inline constexpr static const char* max_mat_cols = "max_mat_cols_float";
        inline constexpr static const char* sum1d = "sum1d_float";
        inline constexpr static const char* sum_mat_cols = "sum_mat_cols_float";
        inline constexpr static const char* logsumexp_coeffs = "logsumexp_coeffs_float";
        inline constexpr static const char* logsumexp_coeffs_mat = "logsumexp_coeffs_mat_float";
        inline constexpr static const char* copy_logpdf_result = "copy_logpdf_result_float";
        inline constexpr static const char* maxwise = "maxwise_float";
        inline constexpr static const char* sum_lse_coefficient = "sum_lse_coefficient_float";
        inline constexpr static const char* finish_lse = "finish_lse_float";
        inline constexpr static const char* substract_matrix = "substract_matrix_float";
        inline constexpr static const char* solve = "solve_float";
        inline constexpr static const char* square = "square_float";
        inline constexpr static const char* logpdf_values = "logpdf_values_float";
        inline constexpr static const char* logpdf_values_mat = "logpdf_values_mat_float";
    };

    template<typename ArrowType>
    struct MaxReduction {
        inline constexpr static const char* reduction1d = OpenCL_kernel_traits<ArrowType>::max1d;
        inline constexpr static const char* reduction_mat = OpenCL_kernel_traits<ArrowType>::max_mat_cols;
    };

    template<typename ArrowType>
    struct SumReduction {
        inline constexpr static const char* reduction1d = OpenCL_kernel_traits<ArrowType>::sum1d;
        inline constexpr static const char* reduction_mat = OpenCL_kernel_traits<ArrowType>::sum_mat_cols;
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
        void read_from_buffer(T* dest, const cl::Buffer from, int size);

        template<typename T>
        cl::Buffer new_buffer(int size);

        cl::Kernel& kernel(const char* name);
        cl::CommandQueue& queue() { return m_queue; }

        template<typename ArrowType>
        std::vector<cl::Buffer> create_reduction1d_buffers(int length);

        template<typename ArrowType>
        std::vector<cl::Buffer> create_reduction_mat_buffers(int length, int cols_mat);

        template<typename T>
        void fill_buffer(cl::Buffer& buffer, T value, int length);

        template<typename ArrowType>
        void logsumexp1d(cl::Buffer& input_vec, int input_length, std::vector<cl::Buffer>& reduc_buffers, cl::Buffer& output, int output_offset);

        template<typename ArrowType, typename Reduction>
        void reduction1d(cl::Buffer& input_vec, int input_length, std::vector<cl::Buffer>& reduc_buffers, cl::Buffer& output_buffer, int ouput_offset);

        template<typename ArrowType>
        void amax1d(cl::Buffer& input_vec, int input_length, std::vector<cl::Buffer>& reduc_buffers, cl::Buffer& output_buffer, int output_offset) {
            reduction1d<ArrowType, MaxReduction<ArrowType>>(input_vec, input_length, reduc_buffers, output_buffer, output_offset);
        }

        template<typename ArrowType>
        void sum1d(cl::Buffer& input_vec, int input_length, std::vector<cl::Buffer>& reduc_buffers, cl::Buffer& output_buffer, int output_offset) {
            reduction1d<ArrowType, SumReduction<ArrowType>>(input_vec, input_length, reduc_buffers, output_buffer, output_offset);
        }

        template<typename ArrowType>
        cl::Buffer logsumexp_cols(cl::Buffer& input_mat, int input_rows, int input_cols, std::vector<cl::Buffer>& reduc_buffers);

        template<typename ArrowType, typename Reduction>
        cl::Buffer reduction_cols(const cl::Buffer& input_mat, int input_rows, int input_cols, std::vector<cl::Buffer>& reduc_buffers);
        
        template<typename ArrowType>
        cl::Buffer amax_cols(const cl::Buffer& input_mat, int input_rows, int input_cols, std::vector<cl::Buffer>& reduc_buffers) {
            return reduction_cols<ArrowType, MaxReduction<ArrowType>>(input_mat, input_rows, input_cols, reduc_buffers);
        }

        template<typename ArrowType>
        cl::Buffer sum_cols(const cl::Buffer& input_mat, int input_rows, int input_cols, std::vector<cl::Buffer>& reduc_buffers) {
            return reduction_cols<ArrowType, SumReduction<ArrowType>>(input_mat, input_rows, input_cols, reduc_buffers);
        }
 

        int max_local_size() { return m_max_local_size; }

        OpenCLConfig(const OpenCLConfig&)    = delete;
        void operator=(const OpenCLConfig&)  = delete;

    private:
        OpenCLConfig();
        // OpenCLConfig(cl::Context cont, cl::CommandQueue queue, cl::Program program, int max_local_size) 
        //                                                             : m_context(cont), 
        //                                                               m_queue(queue), 
        //                                                               m_program(program),
        //                                                               m_kernels(),
        //                                                               m_max_local_size(max_local_size) {}

        // static bool initialized;
        cl::Context m_context;
        cl::CommandQueue m_queue;
        cl::Program m_program;
        std::unordered_map<const char*, cl::Kernel> m_kernels;
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
    void OpenCLConfig::read_from_buffer(T* dest, const cl::Buffer from, int size) {
        cl_int err_code = CL_SUCCESS;

        err_code = m_queue.enqueueReadBuffer(from, CL_TRUE, 0, sizeof(T)*size, dest);

        if (err_code != CL_SUCCESS) {
            throw std::runtime_error("Error copying reading buffer.");
        }
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

    template<typename T>
    void OpenCLConfig::fill_buffer(cl::Buffer& buffer, T value, int length) {
        m_queue.enqueueFillBuffer(buffer, value, 0, length);
    }

    template<typename ArrowType>
    std::vector<cl::Buffer> OpenCLConfig::create_reduction1d_buffers(int length) {
        using CType = typename ArrowType::c_type;
        std::vector<cl::Buffer> res;
        
        auto current_length = length;
        while(current_length > m_max_local_size) {
            auto num_groups = static_cast<int>(std::ceil(static_cast<double>(current_length) / static_cast<double>(m_max_local_size)));
            auto reduc_buffer = new_buffer<CType>(num_groups);
            res.push_back(std::move(reduc_buffer));
            current_length = num_groups;
        }

        return res;
    }

    template<typename ArrowType>
    std::vector<cl::Buffer> OpenCLConfig::create_reduction_mat_buffers(int length, int cols_mat) {
        using CType = typename ArrowType::c_type;
        std::vector<cl::Buffer> res;
        
        auto current_length = length;
        while(current_length > m_max_local_size) {
            auto num_groups = static_cast<int>(std::ceil(static_cast<double>(current_length) / static_cast<double>(m_max_local_size)));
            auto reduc_buffer = new_buffer<CType>(num_groups*cols_mat);
            res.push_back(std::move(reduc_buffer));
            current_length = num_groups;
        }

        return res;
    }

    template<typename ArrowType>
    void OpenCLConfig::logsumexp1d(cl::Buffer& input_vec, int input_length, std::vector<cl::Buffer>& reduc_buffers, cl::Buffer& output, int output_offset) {
        amax1d<ArrowType>(input_vec, input_length, reduc_buffers, output, output_offset);

        auto k_logsumexp_coeffs = kernel(OpenCL_kernel_traits<ArrowType>::logsumexp_coeffs);
        k_logsumexp_coeffs.setArg(0, input_vec);
        k_logsumexp_coeffs.setArg(1, output);
        k_logsumexp_coeffs.setArg(2, static_cast<unsigned int>(output_offset));
        m_queue.enqueueNDRangeKernel(k_logsumexp_coeffs, cl::NullRange,  cl::NDRange(input_length),cl::NullRange);

        sum1d<ArrowType>(input_vec, input_length, reduc_buffers, input_vec, 0);
        // opencl.sum1d<ArrowType>(logpdf_buffer, m, sum_buffer);

        auto k_copy_logpdf_result = kernel(OpenCL_kernel_traits<ArrowType>::copy_logpdf_result);
        k_copy_logpdf_result.setArg(0, input_vec);
        k_copy_logpdf_result.setArg(1, 0u);
        k_copy_logpdf_result.setArg(2, output);
        k_copy_logpdf_result.setArg(3, static_cast<unsigned int>(output_offset));
        k_copy_logpdf_result.setArg(4, output);
        k_copy_logpdf_result.setArg(5, static_cast<unsigned int>(output_offset));
        m_queue.enqueueNDRangeKernel(k_copy_logpdf_result, cl::NullRange,  cl::NDRange(1), cl::NullRange);
    }

    void update_reduction_status(int& length, int& num_groups, int& local_size, int& global_size, int max_local_size);

    template<typename ArrowType, typename Reduction>
    void OpenCLConfig::reduction1d(cl::Buffer& input_vec, 
                                   int input_length, 
                                   std::vector<cl::Buffer>& reduc_buffers, 
                                   cl::Buffer& output_buffer, 
                                   int output_offset) {
        auto length = input_length;
        auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(m_max_local_size)));
        auto local_size = (length > m_max_local_size) ? m_max_local_size : length;
        auto global_size = local_size * num_groups;

        auto k_reduction = kernel(Reduction::reduction1d);
        k_reduction.setArg(0, input_vec);
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size));
        if (num_groups == 1) {
            k_reduction.setArg(3, output_buffer);
            k_reduction.setArg(4, output_offset);
        } else {
            k_reduction.setArg(3, reduc_buffers[0]);
            k_reduction.setArg(4, 0u);
        }

        m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size), cl::NDRange(local_size));

        if (num_groups == 1)
            return;

        update_reduction_status(length, num_groups, local_size, global_size, m_max_local_size);

        for(auto i = 0; length > m_max_local_size; ++i) {
            k_reduction.setArg(0, reduc_buffers[i]);
            k_reduction.setArg(1, static_cast<unsigned int>(length));
            k_reduction.setArg(2, cl::Local(local_size));
            k_reduction.setArg(3, reduc_buffers[i+1]);
            k_reduction.setArg(4, 0u);

            m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size), cl::NDRange(local_size));
            
            update_reduction_status(length, num_groups, local_size, global_size, m_max_local_size);
        }

        k_reduction.setArg(0, reduc_buffers.back());
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size));
        k_reduction.setArg(3, output_buffer);
        k_reduction.setArg(4, output_offset);

        m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size), cl::NDRange(local_size));
    }

    template<typename ArrowType, typename Reduction>
    cl::Buffer OpenCLConfig::reduction_cols(const cl::Buffer& input_mat,
                                            int input_rows,
                                            int input_cols, 
                                            std::vector<cl::Buffer>& reduc_buffers) {
        auto length = input_rows;
        auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(m_max_local_size)));
        auto local_size = (length > m_max_local_size) ? m_max_local_size : length;
        auto global_size = local_size * num_groups;

        auto res = new_buffer<ArrowType>(input_cols);

        auto k_reduction = kernel(Reduction::reduction_mat);
        k_reduction.setArg(0, input_mat);
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size));
        if (num_groups == 1) {
            k_reduction.setArg(3, res);
        } else {
            k_reduction.setArg(3, reduc_buffers[0]);
        }

        m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));

        if (num_groups == 1)
            return std::move(res);

        update_reduction_status(length, num_groups, local_size, global_size, m_max_local_size);

        for(auto i = 0; length > m_max_local_size; ++i) {
            k_reduction.setArg(0, reduc_buffers[i]);
            k_reduction.setArg(1, static_cast<unsigned int>(length));
            k_reduction.setArg(2, cl::Local(local_size));
            k_reduction.setArg(3, reduc_buffers[i+1]);

            m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));
            
            update_reduction_status(length, num_groups, local_size, global_size, m_max_local_size);
        }

        k_reduction.setArg(0, reduc_buffers.back());
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size));
        k_reduction.setArg(3, res);

        m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));
        return std::move(res);
    }

    template<typename ArrowType>
    cl::Buffer OpenCLConfig::logsumexp_cols(cl::Buffer& input_mat, int input_rows, int input_cols, std::vector<cl::Buffer>& reduc_buffers) {
        auto max_buffer = amax_cols<ArrowType>(input_mat, input_rows, input_cols, reduc_buffers);

        auto logsumexp_coeffs_mat = kernel(OpenCL_kernel_traits<ArrowType>::logsumexp_coeffs_mat);
        logsumexp_coeffs_mat.setArg(0, input_mat);
        logsumexp_coeffs_mat.setArg(1, static_cast<unsigned int>(input_rows));
        logsumexp_coeffs_mat.setArg(2, max_buffer);
        m_queue.enqueueNDRangeKernel(logsumexp_coeffs_mat, cl::NullRange,  cl::NDRange(input_rows*input_cols),cl::NullRange);

        auto sum_buffer = sum_cols<ArrowType>(input_mat, input_rows, input_cols, reduc_buffers);

        auto finish_lse = kernel(OpenCL_kernel_traits<ArrowType>::finish_lse);
        finish_lse.setArg(0, sum_buffer);
        finish_lse.setArg(1, max_buffer);
        m_queue.enqueueNDRangeKernel(finish_lse, cl::NullRange,  cl::NDRange(input_cols), cl::NullRange);

        return std::move(sum_buffer);
    }

}


#endif //PGM_OPENCL_CONFIG_HPP