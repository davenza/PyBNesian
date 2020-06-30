#ifndef PGM_DATASET_OPENCL_CONFIG_HPP
#define PGM_DATASET_OPENCL_CONFIG_HPP

#include <fstream>
#include <cmath>
#include <arrow/api.h>
#include <CL/cl2.hpp>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

namespace opencl {

    const char* opencl_error(cl_int error);

    template<typename ArrowType>
    struct OpenCL_kernel_traits;

    template<>
    struct OpenCL_kernel_traits<arrow::DoubleType> {
        inline constexpr static const char* max1d = "max1d_double";
        inline constexpr static const char* max_mat_cols = "max_mat_cols_double";
        inline constexpr static const char* sum1d = "sum1d_double";
        inline constexpr static const char* sum_mat_cols = "sum_mat_cols_double";
        inline constexpr static const char* logsumexp_coeffs = "logsumexp_coeffs_double";
        inline constexpr static const char* solve = "solve_double";
        inline constexpr static const char* square = "square_double";
        inline constexpr static const char* logpdf_values_1d_mat = "logpdf_values_1d_mat_double";
        inline constexpr static const char* substract = "substract_double";
        inline constexpr static const char* logpdf_values_mat_column = "logpdf_values_mat_column_double";
        inline constexpr static const char* logpdf_values_mat_row = "logpdf_values_mat_row_double";
        inline constexpr static const char* finish_lse_offset = "finish_lse_offset_double";
        inline constexpr static const char* substract_vectors = "substract_vectors_double";
    };

    template<>
    struct OpenCL_kernel_traits<arrow::FloatType> {
        inline constexpr static const char* max1d = "max1d_float";
        inline constexpr static const char* max_mat_cols = "max_mat_cols_float";
        inline constexpr static const char* sum1d = "sum1d_float";
        inline constexpr static const char* sum_mat_cols = "sum_mat_cols_float";
        inline constexpr static const char* logsumexp_coeffs = "logsumexp_coeffs_float";
        inline constexpr static const char* solve = "solve_float";
        inline constexpr static const char* square = "square_float";
        inline constexpr static const char* logpdf_values_1d_mat = "logpdf_values_1d_mat_float";
        inline constexpr static const char* substract = "substract_float";
        inline constexpr static const char* logpdf_values_mat_column = "logpdf_values_mat_column_float";
        inline constexpr static const char* logpdf_values_mat_row = "logpdf_values_mat_row_float";
        inline constexpr static const char* finish_lse_offset = "finish_lse_offset_float";
        inline constexpr static const char* substract_vectors = "substract_vectors_float";
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
        cl::Device& device() { return m_device; }

        template<typename T>
        cl::Buffer copy_to_buffer(const T* d, int size);

        template<typename T>
        void read_from_buffer(T* dest, const cl::Buffer from, int size);

        template<typename T>
        cl::Buffer new_buffer(int size, cl_mem_flags flags = CL_MEM_READ_WRITE);

        template<typename T>
        cl::Buffer copy_buffer(const cl::Buffer& input, unsigned int offset, unsigned int length, cl_mem_flags flags = CL_MEM_READ_WRITE);
        
        cl::Kernel& kernel(const char* name);
        cl::CommandQueue& queue() { return m_queue; }

        template<typename ArrowType>
        std::vector<cl::Buffer> create_reduction1d_buffers(int length);

        template<typename ArrowType>
        std::vector<cl::Buffer> create_reduction_mat_buffers(int length, int cols_mat);

        template<typename ArrowType, typename Reduction>
        void reduction1d(cl::Buffer& input_vec, int input_length, std::vector<cl::Buffer>& reduc_buffers, cl::Buffer& output_buffer, int ouput_offset);

        template<typename ArrowType>
        cl::Buffer sum1d(cl::Buffer& input_vec, int input_length) {
            auto reduction_buffers = create_reduction1d_buffers<ArrowType>(input_length);
            cl::Buffer output = new_buffer<typename ArrowType::c_type>(1);
            reduction1d<ArrowType, SumReduction<ArrowType>>(input_vec, input_length, reduction_buffers, output, 0);
            return std::move(output);
        }

        template<typename ArrowType>
        cl::Buffer sum1d(cl::Buffer& input_vec, int input_length, std::vector<cl::Buffer>& reduc_buffers) {
            cl::Buffer output = new_buffer<typename ArrowType::c_type>(1);
            reduction1d<ArrowType, SumReduction<ArrowType>>(input_vec, input_length, reduc_buffers, output, 0);
            return std::move(output);
        }

        template<typename ArrowType>
        void sum1d(cl::Buffer& input_vec, int input_length, std::vector<cl::Buffer>& reduc_buffers, cl::Buffer& output_buffer, int output_offset) {
            reduction1d<ArrowType, SumReduction<ArrowType>>(input_vec, input_length, reduc_buffers, output_buffer, output_offset);
        }

        template<typename ArrowType, typename Reduction>
        cl::Buffer reduction_cols(const cl::Buffer& input_mat, int input_rows, int input_cols, std::vector<cl::Buffer>& reduc_buffers);

        template<typename ArrowType, typename Reduction>
        void reduction_cols_offset(const cl::Buffer& input_mat, int input_rows, int input_cols, 
                            cl::Buffer& output_vec, int output_offset, std::vector<cl::Buffer>& reduc_buffers);
        
        template<typename ArrowType>
        cl::Buffer amax_cols(const cl::Buffer& input_mat, int input_rows, int input_cols, std::vector<cl::Buffer>& reduc_buffers) {
            return reduction_cols<ArrowType, MaxReduction<ArrowType>>(input_mat, input_rows, input_cols, reduc_buffers);
        }

        template<typename ArrowType>
        void sum_cols_offset(const cl::Buffer& input_mat, int input_rows, int input_cols,
                              cl::Buffer& output_vec, int output_offset, std::vector<cl::Buffer>& reduc_buffers) {
            reduction_cols_offset<ArrowType, SumReduction<ArrowType>>(input_mat, input_rows, input_cols, 
                                                                       output_vec, output_offset, reduc_buffers);
        }

        template<typename ArrowType>
        void logsumexp_cols_offset(cl::Buffer& input_mat, int input_rows, int input_cols, 
                                    cl::Buffer& output_vec, int output_offset, std::vector<cl::Buffer>& reduc_buffers);

        int max_local_size() { return m_max_local_size; }

        OpenCLConfig(const OpenCLConfig&)    = delete;
        void operator=(const OpenCLConfig&)  = delete;

    private:
        OpenCLConfig();

        cl::Context m_context;
        cl::CommandQueue m_queue;
        cl::Program m_program;
        cl::Device m_device;
        std::unordered_map<const char*, cl::Kernel> m_kernels;
        int m_max_local_size;
    };

    template<typename T>
    cl::Buffer OpenCLConfig::copy_to_buffer(const T* d, int size) {
        cl::Buffer b = new_buffer<T>(size);

        cl_int err_code = CL_SUCCESS;
        err_code = m_queue.enqueueWriteBuffer(b, CL_TRUE, 0, sizeof(T)*size, d);

        if (err_code != CL_SUCCESS) {
            throw std::runtime_error(std::string("Error copying OpenCL buffer. ") + 
                                     opencl::opencl_error(err_code) + " (" + std::to_string(err_code) + ").");
        }

        return std::move(b);
    }

    template<typename T>
    void OpenCLConfig::read_from_buffer(T* dest, const cl::Buffer from, int size) {
        cl_int err_code = CL_SUCCESS;

        err_code = m_queue.enqueueReadBuffer(from, CL_TRUE, 0, sizeof(T)*size, dest);

        if (err_code != CL_SUCCESS) {
            throw std::runtime_error(std::string("Error reading buffer. ") + 
                                     opencl::opencl_error(err_code) + " (" + std::to_string(err_code) + ").");
        }
    }

    template<typename T>
    cl::Buffer OpenCLConfig::new_buffer(int size, cl_mem_flags flags) {
        cl_int err_code = CL_SUCCESS;
        cl::Buffer b(m_context, flags,  sizeof(T)*size, NULL, &err_code);

        if (err_code != CL_SUCCESS) {
            throw std::runtime_error(std::string("Error creating OpenCL buffer. ") + 
                                                 opencl::opencl_error(err_code) + " (" + std::to_string(err_code) + ").");
        }

        return std::move(b);
    }

    template<typename T>
    cl::Buffer OpenCLConfig::copy_buffer(const cl::Buffer& input, unsigned int offset, unsigned int length, cl_mem_flags flags) {
        cl::Buffer b = new_buffer<T>(length, flags);

        cl_int err_code = CL_SUCCESS;
        err_code = m_queue.enqueueCopyBuffer(input, b, sizeof(T)*offset, 0, sizeof(T)*length);

        if (err_code != CL_SUCCESS) {
            throw std::runtime_error(std::string("Error copying OpenCL buffer. ") + 
                                                 opencl::opencl_error(err_code) + " (" + std::to_string(err_code) + ").");
        }

        return std::move(b);
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
        k_reduction.setArg(2, cl::Local(local_size*sizeof(typename ArrowType::c_type)));
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
            k_reduction.setArg(2, cl::Local(local_size*sizeof(typename ArrowType::c_type)));
            k_reduction.setArg(3, reduc_buffers[i+1]);
            k_reduction.setArg(4, 0u);

            m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size), cl::NDRange(local_size));
            
            update_reduction_status(length, num_groups, local_size, global_size, m_max_local_size);
        }

        k_reduction.setArg(0, reduc_buffers.back());
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size*sizeof(typename ArrowType::c_type)));
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
        k_reduction.setArg(2, cl::Local(local_size*sizeof(typename ArrowType::c_type)));
        k_reduction.setArg(4, 0u);
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
            k_reduction.setArg(2, cl::Local(local_size*sizeof(typename ArrowType::c_type)));
            k_reduction.setArg(3, reduc_buffers[i+1]);

            m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));
            
            update_reduction_status(length, num_groups, local_size, global_size, m_max_local_size);
        }

        k_reduction.setArg(0, reduc_buffers.back());
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size*sizeof(typename ArrowType::c_type)));
        k_reduction.setArg(3, res);

        m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));
        return std::move(res);
    }

    template<typename ArrowType, typename Reduction>
    void OpenCLConfig::reduction_cols_offset(const cl::Buffer& input_mat,
                                                    int input_rows,
                                                    int input_cols,
                                                    cl::Buffer& output_vec,
                                                    int output_offset,
                                                    std::vector<cl::Buffer>& reduc_buffers) {
        auto length = input_rows;
        auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(m_max_local_size)));
        auto local_size = (length > m_max_local_size) ? m_max_local_size : length;
        auto global_size = local_size * num_groups;


        if (num_groups == 1) {
            auto k_reduction = kernel(Reduction::reduction_mat);
            k_reduction.setArg(0, input_mat);
            k_reduction.setArg(1, static_cast<unsigned int>(length));
            k_reduction.setArg(2, cl::Local(local_size*sizeof(typename ArrowType::c_type)));
            k_reduction.setArg(3, output_vec);
            k_reduction.setArg(4, static_cast<unsigned int>(output_offset));
            m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));
            return;
        } else {
            auto k_reduction = kernel(Reduction::reduction_mat);
            k_reduction.setArg(0, input_mat);
            k_reduction.setArg(1, static_cast<unsigned int>(length));
            k_reduction.setArg(2, cl::Local(local_size*sizeof(typename ArrowType::c_type)));
            k_reduction.setArg(3, reduc_buffers[0]);
            k_reduction.setArg(4, 0u);
        
            m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));
            
            update_reduction_status(length, num_groups, local_size, global_size, m_max_local_size);

            for(auto i = 0; length > m_max_local_size; ++i) {
                k_reduction.setArg(0, reduc_buffers[i]);
                k_reduction.setArg(1, static_cast<unsigned int>(length));
                k_reduction.setArg(2, cl::Local(local_size*sizeof(typename ArrowType::c_type)));
                k_reduction.setArg(3, reduc_buffers[i+1]);

                m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));
                
                update_reduction_status(length, num_groups, local_size, global_size, m_max_local_size);
            }

            auto k_reduction_offset = kernel(Reduction::reduction_mat);
            k_reduction_offset.setArg(0, reduc_buffers.back());
            k_reduction_offset.setArg(1, static_cast<unsigned int>(length));
            k_reduction_offset.setArg(2, cl::Local(local_size*sizeof(typename ArrowType::c_type)));
            k_reduction_offset.setArg(3, output_vec);
            k_reduction_offset.setArg(4, static_cast<unsigned int>(output_offset));

            m_queue.enqueueNDRangeKernel(k_reduction_offset, cl::NullRange,  cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1));
        }
    }

    template<typename ArrowType>
    void OpenCLConfig::logsumexp_cols_offset(cl::Buffer& input_mat, int input_rows, int input_cols, 
                                              cl::Buffer& output_vec, int output_offset, std::vector<cl::Buffer>& reduc_buffers) {

        auto max_buffer = amax_cols<ArrowType>(input_mat, input_rows, input_cols, reduc_buffers);

        auto logsumexp_coeffs = kernel(OpenCL_kernel_traits<ArrowType>::logsumexp_coeffs);
        logsumexp_coeffs.setArg(0, input_mat);
        logsumexp_coeffs.setArg(1, static_cast<unsigned int>(input_rows));
        logsumexp_coeffs.setArg(2, max_buffer);
        m_queue.enqueueNDRangeKernel(logsumexp_coeffs, cl::NullRange,  cl::NDRange(input_rows*input_cols),cl::NullRange);

        sum_cols_offset<ArrowType>(input_mat, input_rows, input_cols, output_vec, static_cast<unsigned int>(output_offset), reduc_buffers);

        auto finish_lse = kernel(OpenCL_kernel_traits<ArrowType>::finish_lse_offset);
        finish_lse.setArg(0, output_vec);
        finish_lse.setArg(1, static_cast<unsigned int>(output_offset));
        finish_lse.setArg(2, max_buffer);
        m_queue.enqueueNDRangeKernel(finish_lse, cl::NullRange, cl::NDRange(input_cols), cl::NullRange);
    }
}


#endif //PGM_DATASET_OPENCL_CONFIG_HPP