#ifndef PYBNESIAN_OPENCL_OPENCL_CONFIG_HPP
#define PYBNESIAN_OPENCL_OPENCL_CONFIG_HPP

#include <cmath>
#include <arrow/api.h>
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION  120
#include <CL/cl2.hpp>
#include <util/bit_util.hpp>

// #define CL_HPP_ENABLE_EXCEPTIONS
// #ifdef CL_HPP_MINIMUM_OPENCL_VERSION
// #undef CL_HPP_MINIMUM_OPENCL_VERSION
// #endif
// #ifdef CL_HPP_TARGET_OPENCL_VERSION
// #undef CL_HPP_TARGET_OPENCL_VERSION
// #endif

#define RAISE_ENQUEUEKERNEL_ERROR(enqueue)                                                                             \
    {                                                                                                                  \
        cl_int err_code = CL_SUCCESS;                                                                                  \
        err_code = enqueue;                                                                                            \
        if (err_code != CL_SUCCESS) {                                                                                  \
            throw std::runtime_error(std::string("Error enqueuing OpenCL kernel. ") + opencl::opencl_error(err_code) + \
                                     " (" + std::to_string(err_code) + ").");                                          \
        }                                                                                                              \
    }

namespace opencl {

const char* opencl_error(cl_int error);

template <typename ArrowType>
struct OpenCL_kernel_traits;

template <>
struct OpenCL_kernel_traits<arrow::DoubleType> {
    inline constexpr static const char* max1d = "max1d_double";
    inline constexpr static const char* max_mat_cols = "max_mat_cols_double";
    inline constexpr static const char* sum1d = "sum1d_double";
    inline constexpr static const char* sum_mat_cols = "sum_mat_cols_double";
    inline constexpr static const char* logsumexp_coeffs = "logsumexp_coeffs_double";
    inline constexpr static const char* solve = "solve_double";
    inline constexpr static const char* square = "square_double";
    inline constexpr static const char* logl_values_1d_mat = "logl_values_1d_mat_double";
    inline constexpr static const char* add_logl_values_1d_mat = "add_logl_values_1d_mat_double";
    inline constexpr static const char* substract = "substract_double";
    inline constexpr static const char* logl_values_mat_column = "logl_values_mat_column_double";
    inline constexpr static const char* logl_values_mat_row = "logl_values_mat_row_double";
    inline constexpr static const char* finish_lse_offset = "finish_lse_offset_double";
    inline constexpr static const char* substract_vectors = "substract_vectors_double";
    inline constexpr static const char* exp_elementwise = "exp_elementwise_double";
    inline constexpr static const char* accum_sum_mat_cols = "accum_sum_mat_cols_double";
    inline constexpr static const char* add_accum_sum_mat_cols = "add_accum_sum_mat_cols_double";
    inline constexpr static const char* normalize_accum_sum_mat_cols = "normalize_accum_sum_mat_cols_double";
    inline constexpr static const char* find_random_indices = "find_random_indices_double";
    inline constexpr static const char* conditional_means_1d = "conditional_means_1d_double";
    inline constexpr static const char* conditional_means_column = "conditional_means_column_double";
    inline constexpr static const char* conditional_means_row = "conditional_means_row_double";
    inline constexpr static const char* univariate_normal_cdf = "univariate_normal_cdf_double";
    inline constexpr static const char* normal_cdf = "normal_cdf_double";
    inline constexpr static const char* product_elementwise = "product_elementwise_double";
    inline constexpr static const char* division_elementwise = "division_elementwise_double";
    inline constexpr static const char* sum_ucv_1d = "sum_ucv_1d_double";
    inline constexpr static const char* triangular_substract_mat = "triangular_substract_mat_double";
    inline constexpr static const char* sum_ucv_mat = "sum_ucv_mat_double";
    inline constexpr static const char* ucv_diag = "ucv_diag_double";
    inline constexpr static const char* sum_ucv_diag = "sum_ucv_diag_double";
    inline constexpr static const char* copy_ucv_diag = "copy_ucv_diag_double";
};

template <>
struct OpenCL_kernel_traits<arrow::FloatType> {
    inline constexpr static const char* max1d = "max1d_float";
    inline constexpr static const char* max_mat_cols = "max_mat_cols_float";
    inline constexpr static const char* sum1d = "sum1d_float";
    inline constexpr static const char* sum_mat_cols = "sum_mat_cols_float";
    inline constexpr static const char* logsumexp_coeffs = "logsumexp_coeffs_float";
    inline constexpr static const char* solve = "solve_float";
    inline constexpr static const char* square = "square_float";
    inline constexpr static const char* logl_values_1d_mat = "logl_values_1d_mat_float";
    inline constexpr static const char* add_logl_values_1d_mat = "add_logl_values_1d_mat_float";
    inline constexpr static const char* substract = "substract_float";
    inline constexpr static const char* logl_values_mat_column = "logl_values_mat_column_float";
    inline constexpr static const char* logl_values_mat_row = "logl_values_mat_row_float";
    inline constexpr static const char* finish_lse_offset = "finish_lse_offset_float";
    inline constexpr static const char* substract_vectors = "substract_vectors_float";
    inline constexpr static const char* exp_elementwise = "exp_elementwise_float";
    inline constexpr static const char* accum_sum_mat_cols = "accum_sum_mat_cols_float";
    inline constexpr static const char* add_accum_sum_mat_cols = "add_accum_sum_mat_cols_float";
    inline constexpr static const char* normalize_accum_sum_mat_cols = "normalize_accum_sum_mat_cols_float";
    inline constexpr static const char* find_random_indices = "find_random_indices_float";
    inline constexpr static const char* conditional_means_1d = "conditional_means_1d_float";
    inline constexpr static const char* conditional_means_column = "conditional_means_column_float";
    inline constexpr static const char* conditional_means_row = "conditional_means_row_float";
    inline constexpr static const char* univariate_normal_cdf = "univariate_normal_cdf_float";
    inline constexpr static const char* normal_cdf = "normal_cdf_float";
    inline constexpr static const char* product_elementwise = "product_elementwise_float";
    inline constexpr static const char* division_elementwise = "division_elementwise_float";
    inline constexpr static const char* sum_ucv_1d = "sum_ucv_1d_float";
    inline constexpr static const char* triangular_substract_mat = "triangular_substract_mat_float";
    inline constexpr static const char* sum_ucv_mat = "sum_ucv_mat_float";
    inline constexpr static const char* ucv_diag = "ucv_diag_float";
    inline constexpr static const char* sum_ucv_diag = "sum_ucv_diag_float";
    inline constexpr static const char* copy_ucv_diag = "copy_ucv_diag_float";
};

template <typename ArrowType>
struct MaxReduction {
    inline constexpr static const char* reduction1d = OpenCL_kernel_traits<ArrowType>::max1d;
    inline constexpr static const char* reduction_mat = OpenCL_kernel_traits<ArrowType>::max_mat_cols;
};

template <typename ArrowType>
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

    template <typename T>
    cl::Buffer copy_to_buffer(const T* d, int size);

    template <typename T>
    void read_from_buffer(T* dest, const cl::Buffer& from, int size);

    template <typename T>
    cl::Buffer new_buffer(int size, cl_mem_flags flags = CL_MEM_READ_WRITE);

    template <typename T>
    cl::Buffer copy_buffer(const cl::Buffer& input,
                           unsigned int offset,
                           unsigned int length,
                           cl_mem_flags flags = CL_MEM_READ_WRITE);

    template <typename T>
    void fill_buffer(cl::Buffer& b, const T value, unsigned int length);

    template <typename ArrowType>
    std::pair<cl::Buffer, uint64_t> allocate_temp_mat(size_t rows, size_t cols, size_t max_cols = 64) {
        using CType = typename ArrowType::c_type;
        auto allocated_m = std::min(cols, max_cols);
        return std::make_pair(new_buffer<CType>(rows * allocated_m), allocated_m);
    }

    cl::Kernel& kernel(const char* name);
    cl::CommandQueue& queue() { return m_queue; }

    template <typename ArrowType>
    std::vector<cl::Buffer> create_reduction1d_buffers(int length, const char* kernel_name);

    template <typename ArrowType>
    std::vector<cl::Buffer> create_reduction_mat_buffers(int length, int cols_mat, const char* kernel_name);

    template <typename ArrowType, typename Reduction>
    void reduction1d(cl::Buffer& input_vec, int input_length, cl::Buffer& output_buffer, int ouput_offset);

    template <typename ArrowType>
    cl::Buffer sum1d(cl::Buffer& input_vec, int input_length) {
        cl::Buffer output = new_buffer<typename ArrowType::c_type>(1);
        reduction1d<ArrowType, SumReduction<ArrowType>>(input_vec, input_length, output, 0);
        return output;
    }

    template <typename ArrowType, typename Reduction>
    cl::Buffer reduction_cols(const cl::Buffer& input_mat, int input_rows, int input_cols);

    template <typename ArrowType, typename Reduction>
    void reduction_cols_offset(
        const cl::Buffer& input_mat, int input_rows, int input_cols, cl::Buffer& output_vec, int output_offset);

    template <typename ArrowType>
    cl::Buffer amax_cols(const cl::Buffer& input_mat, int input_rows, int input_cols) {
        return reduction_cols<ArrowType, MaxReduction<ArrowType>>(input_mat, input_rows, input_cols);
    }

    template <typename ArrowType>
    void sum_cols_offset(
        const cl::Buffer& input_mat, int input_rows, int input_cols, cl::Buffer& output_vec, int output_offset) {
        reduction_cols_offset<ArrowType, SumReduction<ArrowType>>(
            input_mat, input_rows, input_cols, output_vec, output_offset);
    }

    template <typename ArrowType>
    void logsumexp_cols_offset(
        cl::Buffer& input_mat, int input_rows, int input_cols, cl::Buffer& output_vec, int output_offset);

    template <typename ArrowType>
    cl::Buffer accum_sum_cols(cl::Buffer& mat, int input_rows, int input_cols);

    size_t kernel_local_size(const char* kernel_name);

    cl_ulong kernel_local_memory(const char* kernel_name);

    size_t max_local_size() { return m_max_local_size; }

    cl_ulong max_local_memory() { return m_max_local_memory_bytes; }

    OpenCLConfig(const OpenCLConfig&) = delete;
    void operator=(const OpenCLConfig&) = delete;

private:
    OpenCLConfig();

    cl::Context m_context;
    cl::CommandQueue m_queue;
    cl::Program m_program;
    cl::Device m_device;
    std::unordered_map<const char*, cl::Kernel> m_kernels;
    std::unordered_map<const char*, size_t> m_kernels_local_size;
    std::unordered_map<const char*, cl_ulong> m_kernels_local_memory;
    size_t m_max_local_size;
    cl_ulong m_max_local_memory_bytes;
};

template <typename T>
cl::Buffer OpenCLConfig::copy_to_buffer(const T* d, int size) {
    cl::Buffer b = new_buffer<T>(size);

    cl_int err_code = CL_SUCCESS;
    err_code = m_queue.enqueueWriteBuffer(b, CL_TRUE, 0, sizeof(T) * size, d);

    if (err_code != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error copying OpenCL buffer. ") + opencl::opencl_error(err_code) + " (" +
                                 std::to_string(err_code) + ").");
    }

    return b;
}

template <typename T>
void OpenCLConfig::read_from_buffer(T* dest, const cl::Buffer& from, int size) {
    cl_int err_code = CL_SUCCESS;
    err_code = m_queue.enqueueReadBuffer(from, CL_TRUE, 0, sizeof(T) * size, dest);

    if (err_code != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error reading buffer. ") + opencl::opencl_error(err_code) + " (" +
                                 std::to_string(err_code) + ").");
    }
}

template <typename T>
cl::Buffer OpenCLConfig::new_buffer(int size, cl_mem_flags flags) {
    cl_int err_code = CL_SUCCESS;
    cl::Buffer b(m_context, flags, sizeof(T) * size, NULL, &err_code);

    if (err_code != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error creating OpenCL buffer of size ") + std::to_string(size) +
                                 opencl::opencl_error(err_code) + " (" + std::to_string(err_code) + ").");
    }

    return b;
}

template <typename T>
cl::Buffer OpenCLConfig::copy_buffer(const cl::Buffer& input,
                                     unsigned int offset,
                                     unsigned int length,
                                     cl_mem_flags flags) {
    cl::Buffer b = new_buffer<T>(length, flags);

    cl_int err_code = CL_SUCCESS;
    err_code = m_queue.enqueueCopyBuffer(input, b, sizeof(T) * offset, 0, sizeof(T) * length);

    if (err_code != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error copying OpenCL buffer. ") + opencl::opencl_error(err_code) + " (" +
                                 std::to_string(err_code) + ").");
    }

    return b;
}

template <typename T>
void OpenCLConfig::fill_buffer(cl::Buffer& buffer, const T value, unsigned int length) {
    cl_int err_code = CL_SUCCESS;
    err_code = m_queue.enqueueFillBuffer<T>(buffer, value, 0, length * sizeof(T));

    if (err_code != CL_SUCCESS) {
        throw std::runtime_error(std::string("Error filling OpenCL buffer. ") + opencl::opencl_error(err_code) + " (" +
                                 std::to_string(err_code) + ").");
    }
}

template <typename ArrowType>
std::vector<cl::Buffer> OpenCLConfig::create_reduction1d_buffers(int length, const char* kernel_name) {
    using CType = typename ArrowType::c_type;
    std::vector<cl::Buffer> res;

    auto k_local_size = kernel_local_size(kernel_name);
    auto k_local_memory = kernel_local_memory(kernel_name);
    auto free_local_memory = m_max_local_memory_bytes - k_local_memory;

    auto device_max_local_size = std::min(static_cast<int>(free_local_memory / static_cast<double>(sizeof(CType))),
                                          static_cast<int>(k_local_size));

    auto current_length = length;
    while (current_length > device_max_local_size) {
        auto num_groups = static_cast<int>(
            std::ceil(static_cast<double>(current_length) / static_cast<double>(device_max_local_size)));
        auto reduc_buffer = new_buffer<CType>(num_groups);
        res.push_back(std::move(reduc_buffer));
        current_length = num_groups;
    }

    return res;
}

template <typename ArrowType>
std::vector<cl::Buffer> OpenCLConfig::create_reduction_mat_buffers(int length, int cols_mat, const char* kernel_name) {
    using CType = typename ArrowType::c_type;
    std::vector<cl::Buffer> res;

    auto k_local_size = kernel_local_size(kernel_name);
    auto k_local_memory = kernel_local_memory(kernel_name);
    auto free_local_memory = m_max_local_memory_bytes - k_local_memory;

    auto device_max_local_size = std::min(static_cast<int>(free_local_memory / static_cast<double>(sizeof(CType))),
                                          static_cast<int>(k_local_size));

    auto current_length = length;
    while (current_length > device_max_local_size) {
        auto num_groups = static_cast<int>(
            std::ceil(static_cast<double>(current_length) / static_cast<double>(device_max_local_size)));
        auto reduc_buffer = new_buffer<CType>(num_groups * cols_mat);
        res.push_back(std::move(reduc_buffer));
        current_length = num_groups;
    }

    return res;
}

void update_reduction_status(int& length, int& num_groups, int& local_size, int& global_size, int max_local_size);

template <typename ArrowType, typename Reduction>
void OpenCLConfig::reduction1d(cl::Buffer& input_vec, int input_length, cl::Buffer& output_buffer, int output_offset) {
    using CType = typename ArrowType::c_type;
    auto reduc_buffers = create_reduction1d_buffers<ArrowType>(input_length, Reduction::reduction1d);

    auto k_local_size = kernel_local_size(Reduction::reduction1d);
    auto k_local_memory = kernel_local_memory(Reduction::reduction1d);
    auto free_local_memory = m_max_local_memory_bytes - k_local_memory;
    auto device_max_local_size = std::min(static_cast<int>(free_local_memory / static_cast<double>(sizeof(CType))),
                                          static_cast<int>(k_local_size));
    auto length = input_length;
    auto local_size = std::min(length, device_max_local_size);
    auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(local_size)));
    auto global_size = local_size * num_groups;

    auto k_reduction = kernel(Reduction::reduction1d);
    k_reduction.setArg(0, input_vec);
    k_reduction.setArg(1, static_cast<unsigned int>(length));
    k_reduction.setArg(2, cl::Local(local_size * sizeof(CType)));
    if (num_groups == 1) {
        k_reduction.setArg(3, output_buffer);
        k_reduction.setArg(4, output_offset);
    } else {
        k_reduction.setArg(3, reduc_buffers[0]);
        k_reduction.setArg(4, 0u);
    }

    RAISE_ENQUEUEKERNEL_ERROR(
        m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size)));

    if (num_groups == 1) return;

    update_reduction_status(length, num_groups, local_size, global_size, device_max_local_size);

    for (auto i = 0; length > device_max_local_size; ++i) {
        k_reduction.setArg(0, reduc_buffers[i]);
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size * sizeof(typename ArrowType::c_type)));
        k_reduction.setArg(3, reduc_buffers[i + 1]);
        k_reduction.setArg(4, 0u);

        RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
            k_reduction, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size)));
        update_reduction_status(length, num_groups, local_size, global_size, device_max_local_size);
    }

    k_reduction.setArg(0, reduc_buffers.back());
    k_reduction.setArg(1, static_cast<unsigned int>(length));
    k_reduction.setArg(2, cl::Local(local_size * sizeof(typename ArrowType::c_type)));
    k_reduction.setArg(3, output_buffer);
    k_reduction.setArg(4, output_offset);
    RAISE_ENQUEUEKERNEL_ERROR(
        m_queue.enqueueNDRangeKernel(k_reduction, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size)));
}

template <typename ArrowType, typename Reduction>
cl::Buffer OpenCLConfig::reduction_cols(const cl::Buffer& input_mat, int input_rows, int input_cols) {
    using CType = typename ArrowType::c_type;

    auto reduc_buffers = create_reduction_mat_buffers<ArrowType>(input_rows, input_cols, Reduction::reduction_mat);

    auto k_local_size = kernel_local_size(Reduction::reduction_mat);
    auto k_local_memory = kernel_local_memory(Reduction::reduction_mat);
    auto free_local_memory = m_max_local_memory_bytes - k_local_memory;
    auto device_max_local_size = std::min(static_cast<int>(free_local_memory / static_cast<double>(sizeof(CType))),
                                          static_cast<int>(k_local_size));
    auto length = input_rows;
    auto local_size = std::min(length, device_max_local_size);
    auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(local_size)));
    auto global_size = local_size * num_groups;

    auto res = new_buffer<ArrowType>(input_cols);

    auto k_reduction = kernel(Reduction::reduction_mat);
    k_reduction.setArg(0, input_mat);
    k_reduction.setArg(1, static_cast<unsigned int>(length));
    k_reduction.setArg(2, cl::Local(local_size * sizeof(CType)));
    k_reduction.setArg(4, 0u);
    if (num_groups == 1) {
        k_reduction.setArg(3, res);
    } else {
        k_reduction.setArg(3, reduc_buffers[0]);
    }

    RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
        k_reduction, cl::NullRange, cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1)));

    if (num_groups == 1) return res;

    update_reduction_status(length, num_groups, local_size, global_size, device_max_local_size);

    for (auto i = 0; length > device_max_local_size; ++i) {
        k_reduction.setArg(0, reduc_buffers[i]);
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size * sizeof(CType)));
        k_reduction.setArg(3, reduc_buffers[i + 1]);

        RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
            k_reduction, cl::NullRange, cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1)));
        update_reduction_status(length, num_groups, local_size, global_size, device_max_local_size);
    }

    k_reduction.setArg(0, reduc_buffers.back());
    k_reduction.setArg(1, static_cast<unsigned int>(length));
    k_reduction.setArg(2, cl::Local(local_size * sizeof(CType)));
    k_reduction.setArg(3, res);
    RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
        k_reduction, cl::NullRange, cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1)));
    return res;
}

template <typename ArrowType, typename Reduction>
void OpenCLConfig::reduction_cols_offset(
    const cl::Buffer& input_mat, int input_rows, int input_cols, cl::Buffer& output_vec, int output_offset) {
    using CType = typename ArrowType::c_type;

    auto reduc_buffers = create_reduction_mat_buffers<ArrowType>(input_rows, input_cols, Reduction::reduction_mat);

    auto k_local_size = kernel_local_size(Reduction::reduction_mat);
    auto k_local_memory = kernel_local_memory(Reduction::reduction_mat);
    auto free_local_memory = m_max_local_memory_bytes - k_local_memory;
    auto device_max_local_size = std::min(static_cast<int>(free_local_memory / static_cast<double>(sizeof(CType))),
                                          static_cast<int>(k_local_size));
    auto length = input_rows;
    auto local_size = std::min(length, device_max_local_size);
    auto num_groups = static_cast<int>(std::ceil(static_cast<double>(length) / static_cast<double>(local_size)));
    auto global_size = local_size * num_groups;

    if (num_groups == 1) {
        auto k_reduction = kernel(Reduction::reduction_mat);
        k_reduction.setArg(0, input_mat);
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size * sizeof(CType)));
        k_reduction.setArg(3, output_vec);
        k_reduction.setArg(4, static_cast<unsigned int>(output_offset));
        RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
            k_reduction, cl::NullRange, cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1)));
    } else {
        auto k_reduction = kernel(Reduction::reduction_mat);
        k_reduction.setArg(0, input_mat);
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size * sizeof(CType)));
        k_reduction.setArg(3, reduc_buffers[0]);
        k_reduction.setArg(4, 0u);

        RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
            k_reduction, cl::NullRange, cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1)));

        update_reduction_status(length, num_groups, local_size, global_size, device_max_local_size);

        for (auto i = 0; length > device_max_local_size; ++i) {
            k_reduction.setArg(0, reduc_buffers[i]);
            k_reduction.setArg(1, static_cast<unsigned int>(length));
            k_reduction.setArg(2, cl::Local(local_size * sizeof(CType)));
            k_reduction.setArg(3, reduc_buffers[i + 1]);

            RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
                k_reduction, cl::NullRange, cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1)));

            update_reduction_status(length, num_groups, local_size, global_size, device_max_local_size);
        }

        k_reduction.setArg(0, reduc_buffers.back());
        k_reduction.setArg(1, static_cast<unsigned int>(length));
        k_reduction.setArg(2, cl::Local(local_size * sizeof(CType)));
        k_reduction.setArg(3, output_vec);
        k_reduction.setArg(4, static_cast<unsigned int>(output_offset));

        RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
            k_reduction, cl::NullRange, cl::NDRange(global_size, input_cols), cl::NDRange(local_size, 1)));
    }
}

template <typename ArrowType>
void OpenCLConfig::logsumexp_cols_offset(
    cl::Buffer& input_mat, int input_rows, int input_cols, cl::Buffer& output_vec, int output_offset) {
    auto max_buffer = amax_cols<ArrowType>(input_mat, input_rows, input_cols);

    auto logsumexp_coeffs = kernel(OpenCL_kernel_traits<ArrowType>::logsumexp_coeffs);
    logsumexp_coeffs.setArg(0, input_mat);
    logsumexp_coeffs.setArg(1, static_cast<unsigned int>(input_rows));
    logsumexp_coeffs.setArg(2, max_buffer);
    RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
        logsumexp_coeffs, cl::NullRange, cl::NDRange(input_rows * input_cols), cl::NullRange));
    sum_cols_offset<ArrowType>(input_mat, input_rows, input_cols, output_vec, static_cast<unsigned int>(output_offset));

    auto finish_lse = kernel(OpenCL_kernel_traits<ArrowType>::finish_lse_offset);
    finish_lse.setArg(0, output_vec);
    finish_lse.setArg(1, static_cast<unsigned int>(output_offset));
    finish_lse.setArg(2, max_buffer);
    RAISE_ENQUEUEKERNEL_ERROR(
        m_queue.enqueueNDRangeKernel(finish_lse, cl::NullRange, cl::NDRange(input_cols), cl::NullRange));
}

template <typename ArrowType>
cl::Buffer OpenCLConfig::accum_sum_cols(cl::Buffer& mat, int input_rows, int input_cols) {
    using CType = typename ArrowType::c_type;
    auto k_local_size = kernel_local_size(OpenCL_kernel_traits<ArrowType>::accum_sum_mat_cols);
    auto k_local_memory = kernel_local_memory(OpenCL_kernel_traits<ArrowType>::accum_sum_mat_cols);
    auto free_local_memory = m_max_local_memory_bytes - k_local_memory;
    auto device_max_local_wg = std::min(
        util::bit_util::previous_power2(static_cast<int>(free_local_memory / static_cast<double>(2 * sizeof(CType)))),
        static_cast<int>(k_local_size));
    auto local_wg = (input_rows > device_max_local_wg) ? device_max_local_wg : util::bit_util::next_power2(input_rows);
    auto num_groups = static_cast<int>(std::ceil(static_cast<double>(input_rows) / static_cast<double>(2 * local_wg)));
    auto global_wg = static_cast<int>(std::ceil(static_cast<double>(num_groups * local_wg)));

    auto& opencl = OpenCLConfig::get();
    auto group_sums = opencl.new_buffer<CType>(num_groups * input_cols);

    auto k_accum_sumexp = kernel(OpenCL_kernel_traits<ArrowType>::accum_sum_mat_cols);
    k_accum_sumexp.setArg(0, mat);
    k_accum_sumexp.setArg(1, static_cast<unsigned int>(input_rows));
    k_accum_sumexp.setArg(2, cl::Local(2 * local_wg * sizeof(CType)));
    k_accum_sumexp.setArg(3, group_sums);
    RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
        k_accum_sumexp, cl::NullRange, cl::NDRange(global_wg, input_cols), cl::NDRange(local_wg, 1)));

    if (num_groups > 1) {
        auto total_sum = accum_sum_cols<ArrowType>(group_sums, num_groups, input_cols);

        auto k_add_accum_sumexp = kernel(OpenCL_kernel_traits<ArrowType>::add_accum_sum_mat_cols);
        k_add_accum_sumexp.setArg(0, mat);
        k_add_accum_sumexp.setArg(1, static_cast<unsigned int>(input_rows));
        k_add_accum_sumexp.setArg(2, static_cast<unsigned int>(2 * local_wg));
        k_add_accum_sumexp.setArg(3, static_cast<unsigned int>(2 * local_wg));
        k_add_accum_sumexp.setArg(4, static_cast<unsigned int>(num_groups));
        k_add_accum_sumexp.setArg(5, group_sums);
        RAISE_ENQUEUEKERNEL_ERROR(m_queue.enqueueNDRangeKernel(
            k_add_accum_sumexp, cl::NullRange, cl::NDRange(input_rows - 2 * local_wg, input_cols), cl::NullRange));

        return total_sum;
    } else {
        return group_sums;
    }
}

}  // namespace opencl

#endif  // PYBNESIAN_OPENCL_OPENCL_CONFIG_HPP