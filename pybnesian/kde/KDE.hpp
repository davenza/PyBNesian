#ifndef PYBNESIAN_KDE_KDE_HPP
#define PYBNESIAN_KDE_KDE_HPP

#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <kde/BandwidthSelector.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <opencl/opencl_config.hpp>
#include <util/math_constants.hpp>
#include <util/pickle.hpp>

using opencl::OpenCLConfig, opencl::OpenCL_kernel_traits;

namespace kde {

/**
 * @brief Class for calculating the Univariate Kernel Density Estimation.

 *
 */
struct UnivariateKDE {
    /**
     * @brief Executes the log-likelihood calculation for a univariate KDE model.
     *
     * @tparam ArrowType Arrow data type.
     * @param training_vec Training data.
     * @param training_length Number of training instances.
     * @param test_vec Test data.
     * @param int Unused.
     * @param test_offset ?
     * @param test_length Number of test instances.
     * @param int Unused.
     * @param cholesky Cholesky decomposition of the bandwidth matrix.
     * @param lognorm_const log-likelihood constant.
     * @param output_mat Output matrix.
     */
    template <typename ArrowType>
    void static execute_logl_mat(const cl::Buffer& training_vec,
                                 const unsigned int training_length,
                                 const cl::Buffer& test_vec,
                                 const unsigned int,
                                 const unsigned int test_offset,
                                 const unsigned int test_length,
                                 const unsigned int,
                                 const cl::Buffer& cholesky,
                                 const typename ArrowType::c_type lognorm_const,
                                 cl::Buffer&,
                                 cl::Buffer& output_mat);

    template <typename ArrowType>
    static void execute_conditional_means(const cl::Buffer& joint_training,
                                          const cl::Buffer&,
                                          const unsigned int training_rows,
                                          const cl::Buffer& evidence_test,
                                          const unsigned int test_physical_rows,
                                          const unsigned int test_offset,
                                          const unsigned int test_length,
                                          const unsigned int,
                                          const cl::Buffer& transform_mean,
                                          cl::Buffer&,
                                          cl::Buffer& output_mat);
};

/**
 * @brief Executes the log-likelihood calculation for a univariate KDE model for each variable.
 *
 * @tparam ArrowType Arrow data type.
 * @param training_vec Training data.
 * @param training_length Number of training instances.
 * @param test_vec Test data.
 * @param int Unused.
 * @param test_offset
 * @param test_length Number of test instances.
 * @param int Unused.
 * @param cholesky Cholesky decomposition of the bandwidth matrix.
 * @param lognorm_const log-likelihood constant.
 * @param output_mat Output matrix.
 */
template <typename ArrowType>
void UnivariateKDE::execute_logl_mat(const cl::Buffer& training_vec,
                                     const unsigned int training_length,
                                     const cl::Buffer& test_vec,
                                     const unsigned int,
                                     const unsigned int test_offset,
                                     const unsigned int test_length,
                                     const unsigned int,
                                     const cl::Buffer& cholesky,
                                     const typename ArrowType::c_type lognorm_const,
                                     cl::Buffer&,
                                     cl::Buffer& output_mat) {
    auto& opencl = OpenCLConfig::get();
    // TODO: This is the kernel that is executed, might be wrong?
    // OpenCL kernel for calculating the log-likelihood values for each test instance
    //     __kernel void logl_values_1d_mat_double(__global double *restrict train_vector,
    //                                       __private uint train_rows,
    //                                       __global double *restrict test_vector,
    //                                       __private uint test_offset,
    //                                       __constant double *standard_deviation,
    //                                       __private double lognorm_factor,
    //                                       __global double *restrict result) {
    //     int i = get_global_id(0);
    //     int train_idx = ROW(i, train_rows);
    //     int test_idx = COL(i, train_rows);
    //     double d = (train_vector[train_idx] - test_vector[test_offset + test_idx]) / standard_deviation[0];

    //     result[i] = (-0.5*d*d) + lognorm_factor;
    // }
    auto& k_logl_values_1d_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logl_values_1d_mat);
    k_logl_values_1d_mat.setArg(0, training_vec);
    k_logl_values_1d_mat.setArg(1, training_length);
    k_logl_values_1d_mat.setArg(2, test_vec);
    k_logl_values_1d_mat.setArg(3, test_offset);
    k_logl_values_1d_mat.setArg(4, cholesky);
    k_logl_values_1d_mat.setArg(5, lognorm_const);
    k_logl_values_1d_mat.setArg(6, output_mat);

    auto& queue = opencl.queue();
    // ? Calculates the log-likelihood values for each test instance
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
        k_logl_values_1d_mat, cl::NullRange, cl::NDRange(training_length * test_length), cl::NullRange));
}

// Computes conditional mu.
template <typename ArrowType>
void UnivariateKDE::execute_conditional_means(const cl::Buffer& joint_training,
                                              const cl::Buffer&,
                                              const unsigned int training_rows,
                                              const cl::Buffer& evidence_test,
                                              const unsigned int test_physical_rows,
                                              const unsigned int test_offset,
                                              const unsigned int test_length,
                                              const unsigned int,
                                              const cl::Buffer& transform_mean,
                                              cl::Buffer&,
                                              cl::Buffer& output_mat) {
    auto& opencl = OpenCLConfig::get();
    auto& k_conditional_means_1d = opencl.kernel(OpenCL_kernel_traits<ArrowType>::conditional_means_1d);
    k_conditional_means_1d.setArg(0, joint_training);
    k_conditional_means_1d.setArg(1, training_rows);
    k_conditional_means_1d.setArg(2, evidence_test);
    k_conditional_means_1d.setArg(3, test_physical_rows);
    k_conditional_means_1d.setArg(4, test_offset);
    k_conditional_means_1d.setArg(5, transform_mean);
    k_conditional_means_1d.setArg(6, output_mat);
    auto& queue = opencl.queue();

    // ? Calculates the log-likelihood values for each test instance
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
        k_conditional_means_1d, cl::NullRange, cl::NDRange(training_rows * test_length), cl::NullRange));
}

struct MultivariateKDE {
    template <typename ArrowType>
    static void execute_logl_mat(const cl::Buffer& training_mat,
                                 const unsigned int training_rows,
                                 const cl::Buffer& test_mat,
                                 const unsigned int test_physical_rows,
                                 const unsigned int test_offset,
                                 const unsigned int test_length,
                                 const unsigned int matrices_cols,
                                 const cl::Buffer& cholesky,
                                 const typename ArrowType::c_type lognorm_const,
                                 cl::Buffer& tmp_mat,
                                 cl::Buffer& output_mat);

    template <typename ArrowType>
    static void execute_conditional_means(const cl::Buffer& joint_training,
                                          const cl::Buffer& marg_training,
                                          const unsigned int training_rows,
                                          const cl::Buffer& evidence_test,
                                          const unsigned int test_physical_rows,
                                          const unsigned int test_offset,
                                          const unsigned int test_length,
                                          const unsigned int evidence_cols,
                                          const cl::Buffer& transform_mean,
                                          cl::Buffer& tmp_mat,
                                          cl::Buffer& output_mat);
};
/**
 * @brief Executes the log-likelihood calculation for a multivariate KDE model for each variable.
 *
 * @tparam ArrowType Arrow data type.
 * @param training_mat Training data.
 * @param training_rows Number of training instances.
 * @param test_mat Test data.
 * @param test_physical_rows Number of test instances.
 * @param test_offset ?
 * @param test_length Number of test instances.
 * @param matrices_cols Number of columns of the matrices.
 * @param cholesky Cholesky decomposition of the bandwidth matrix.
 * @param lognorm_const log-likelihood constant.
 * @param tmp_mat Temporary matrix.
 * @param output_mat Output matrix.
 */
template <typename ArrowType>
void MultivariateKDE::execute_logl_mat(const cl::Buffer& training_mat,
                                       const unsigned int training_rows,
                                       const cl::Buffer& test_mat,
                                       const unsigned int test_physical_rows,
                                       const unsigned int test_offset,
                                       const unsigned int test_length,
                                       const unsigned int matrices_cols,
                                       const cl::Buffer& cholesky,
                                       const typename ArrowType::c_type lognorm_const,
                                       cl::Buffer& tmp_mat,
                                       cl::Buffer& output_mat) {
    auto& opencl = OpenCLConfig::get();

    // __kernel void substract_double(__global double* restrict training_matrix,
    //                                __private uint training_physical_rows,
    //                                __private uint training_offset,
    //                                __private uint training_rows,
    //                                __global double* restrict test_matrix,
    //                                __private uint test_physical_rows,
    //                                __private uint test_offset,
    //                                __private uint test_row_idx,
    //                                __global double* restrict res) {
    //     uint i = get_global_id(0);
    //     uint r = ROW(i, training_rows) + training_offset;
    //     uint c = COL(i, training_rows);
    //     res[i] = test_matrix[IDX(test_offset + test_row_idx, c, test_physical_rows)] -
    //              training_matrix[IDX(r, c, training_physical_rows)];
    // }
    auto& k_substract = opencl.kernel(OpenCL_kernel_traits<ArrowType>::substract);

    // __kernel void solve_double(__global double* restrict diff_matrix,
    //                            __private uint diff_matrix_rows,
    //                            __private uint matrices_cols,
    //                            __global double* restrict cholesky_matrix) {
    //     uint r = get_global_id(0);

    //     for (uint c = 0; c < matrices_cols; c++) {
    //         for (uint i = 0; i < c; i++) {
    //             diff_matrix[IDX(r, c, diff_matrix_rows)] -=
    //                 cholesky_matrix[IDX(c, i, matrices_cols)] * diff_matrix[IDX(r, i, diff_matrix_rows)];
    //         }
    //         diff_matrix[IDX(r, c, diff_matrix_rows)] /= cholesky_matrix[IDX(c, c, matrices_cols)];
    //     }
    // }
    auto& k_solve = opencl.kernel(OpenCL_kernel_traits<ArrowType>::solve);
    k_solve.setArg(0, tmp_mat);
    k_solve.setArg(2, matrices_cols);
    k_solve.setArg(3, cholesky);

    // __kernel void square_double(__global double* restrict m) {
    //     uint idx = get_global_id(0);
    //     double d = m[idx];
    //     m[idx] = d * d;
    // }
    auto& k_square = opencl.kernel(OpenCL_kernel_traits<ArrowType>::square);
    k_square.setArg(0, tmp_mat);

    auto& queue = opencl.queue();

    if (training_rows >
        test_length) {  // When the number of training instances is greater than the number of test instances
        // Test Matrix - Training Matrix
        k_substract.setArg(0, training_mat);
        k_substract.setArg(1, training_rows);
        k_substract.setArg(2, 0u);
        k_substract.setArg(3, training_rows);
        k_substract.setArg(4, test_mat);
        k_substract.setArg(5, test_physical_rows);
        k_substract.setArg(6, test_offset);
        k_substract.setArg(8, tmp_mat);

        k_solve.setArg(1, training_rows);

        auto& k_logl_values_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logl_values_mat_column);
        k_logl_values_mat.setArg(0, tmp_mat);
        k_logl_values_mat.setArg(1, matrices_cols);
        k_logl_values_mat.setArg(2, output_mat);
        k_logl_values_mat.setArg(3, training_rows);
        k_logl_values_mat.setArg(5, lognorm_const);

        // NOTE: Calculates the log-likelihood values for each test instance
        for (unsigned int i = 0; i < test_length; ++i) {
            k_substract.setArg(7, i);
            RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
                k_substract, cl::NullRange, cl::NDRange(training_rows * matrices_cols), cl::NullRange));

            RAISE_ENQUEUEKERNEL_ERROR(
                queue.enqueueNDRangeKernel(k_solve, cl::NullRange, cl::NDRange(training_rows), cl::NullRange));

            RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
                k_square, cl::NullRange, cl::NDRange(training_rows * matrices_cols), cl::NullRange));

            k_logl_values_mat.setArg(4, i);
            RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
                k_logl_values_mat, cl::NullRange, cl::NDRange(training_rows), cl::NullRange));
        }
    } else {
        k_substract.setArg(0, test_mat);
        k_substract.setArg(1, test_physical_rows);
        k_substract.setArg(2, test_offset);
        k_substract.setArg(3, test_length);
        k_substract.setArg(4, training_mat);
        k_substract.setArg(5, training_rows);
        k_substract.setArg(6, 0);
        k_substract.setArg(8, tmp_mat);

        k_solve.setArg(1, test_length);

        auto& k_logl_values_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::logl_values_mat_row);
        k_logl_values_mat.setArg(0, tmp_mat);
        k_logl_values_mat.setArg(1, matrices_cols);
        k_logl_values_mat.setArg(2, output_mat);
        k_logl_values_mat.setArg(3, training_rows);
        k_logl_values_mat.setArg(5, lognorm_const);

        // ? Calculates the log-likelihood values for each test instance
        for (unsigned int i = 0; i < training_rows; ++i) {
            k_substract.setArg(7, i);
            RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
                k_substract, cl::NullRange, cl::NDRange(test_length * matrices_cols), cl::NullRange));
            RAISE_ENQUEUEKERNEL_ERROR(
                queue.enqueueNDRangeKernel(k_solve, cl::NullRange, cl::NDRange(test_length), cl::NullRange));
            RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
                k_square, cl::NullRange, cl::NDRange(test_length * matrices_cols), cl::NullRange));
            k_logl_values_mat.setArg(4, i);
            RAISE_ENQUEUEKERNEL_ERROR(
                queue.enqueueNDRangeKernel(k_logl_values_mat, cl::NullRange, cl::NDRange(test_length), cl::NullRange));
        }
    }
}
// Computes conditional mu.
template <typename ArrowType>
void MultivariateKDE::execute_conditional_means(const cl::Buffer& joint_training,
                                                const cl::Buffer& marg_training,
                                                const unsigned int training_rows,
                                                const cl::Buffer& evidence_test,
                                                const unsigned int test_physical_rows,
                                                const unsigned int test_offset,
                                                const unsigned int test_length,
                                                const unsigned int evidence_cols,
                                                const cl::Buffer& transform_mean,
                                                cl::Buffer& tmp_mat,
                                                cl::Buffer& output_mat) {
    auto& opencl = OpenCLConfig::get();

    auto& k_substract = opencl.kernel(OpenCL_kernel_traits<ArrowType>::substract);
    auto& queue = opencl.queue();

    if (training_rows > test_length) {
        k_substract.setArg(0, marg_training);
        k_substract.setArg(1, training_rows);
        k_substract.setArg(2, 0u);
        k_substract.setArg(3, training_rows);
        k_substract.setArg(4, evidence_test);
        k_substract.setArg(5, test_physical_rows);
        k_substract.setArg(6, test_offset);
        k_substract.setArg(8, tmp_mat);

        auto& k_conditional_means = opencl.kernel(OpenCL_kernel_traits<ArrowType>::conditional_means_column);

        k_conditional_means.setArg(0, joint_training);
        k_conditional_means.setArg(1, static_cast<unsigned int>(training_rows));
        k_conditional_means.setArg(2, tmp_mat);
        k_conditional_means.setArg(3, static_cast<unsigned int>(training_rows));
        k_conditional_means.setArg(4, transform_mean);
        k_conditional_means.setArg(5, static_cast<unsigned int>(evidence_cols));
        k_conditional_means.setArg(6, output_mat);
        k_conditional_means.setArg(8, static_cast<unsigned int>(training_rows));

        for (unsigned int i = 0; i < test_length; ++i) {
            k_substract.setArg(7, i);
            RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
                k_substract, cl::NullRange, cl::NDRange(training_rows * evidence_cols), cl::NullRange));
            k_conditional_means.setArg(7, i);
            RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
                k_conditional_means, cl::NullRange, cl::NDRange(training_rows), cl::NullRange));
        }
    } else {
        k_substract.setArg(0, evidence_test);
        k_substract.setArg(1, test_physical_rows);
        k_substract.setArg(2, test_offset);
        k_substract.setArg(3, test_length);
        k_substract.setArg(4, marg_training);
        k_substract.setArg(5, training_rows);
        k_substract.setArg(6, 0);
        k_substract.setArg(8, tmp_mat);

        auto& k_conditional_means = opencl.kernel(OpenCL_kernel_traits<ArrowType>::conditional_means_row);

        k_conditional_means.setArg(0, joint_training);
        k_conditional_means.setArg(1, training_rows);
        k_conditional_means.setArg(2, tmp_mat);
        k_conditional_means.setArg(3, test_length);
        k_conditional_means.setArg(4, transform_mean);
        k_conditional_means.setArg(5, evidence_cols);
        k_conditional_means.setArg(6, output_mat);
        k_conditional_means.setArg(8, training_rows);

        for (unsigned int i = 0; i < training_rows; ++i) {
            k_substract.setArg(7, i);
            RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
                k_substract, cl::NullRange, cl::NDRange(test_length * evidence_cols), cl::NullRange));
            k_conditional_means.setArg(7, i);
            RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
                k_conditional_means, cl::NullRange, cl::NDRange(test_length), cl::NullRange));
        }
    }
}

class KDE {
public:
    KDE()
        : m_variables(),
          m_fitted(false),
          m_bselector(std::make_shared<NormalReferenceRule>()),
          m_bandwidth(),
          m_H_cholesky(),
          m_training(),
          m_lognorm_const(0),
          N(0),
          m_training_type(arrow::float64()) {}

    KDE(std::vector<std::string> variables) : KDE(variables, std::make_shared<NormalReferenceRule>()) {}

    KDE(std::vector<std::string> variables, std::shared_ptr<BandwidthSelector> b_selector)
        : m_variables(variables),
          m_fitted(false),
          m_bselector(b_selector),
          m_bandwidth(),
          m_H_cholesky(),
          m_training(),
          m_lognorm_const(0),
          N(0),
          m_training_type(arrow::float64()) {
        if (b_selector == nullptr) throw std::runtime_error("Bandwidth selector procedure must be non-null.");

        if (m_variables.empty()) {
            throw std::invalid_argument("Cannot create a KDE model with 0 variables");
        }
    }

    const std::vector<std::string>& variables() const { return m_variables; }
    void fit(const DataFrame& df);

    template <typename ArrowType, typename EigenMatrix>
    void fit(EigenMatrix bandwidth,
             cl::Buffer training_data,
             std::shared_ptr<arrow::DataType> training_type,
             int training_instances);

    const MatrixXd& bandwidth() const { return m_bandwidth; }
    void setBandwidth(MatrixXd& new_bandwidth) {
        if (new_bandwidth.rows() != new_bandwidth.cols() ||
            static_cast<size_t>(new_bandwidth.rows()) != m_variables.size())
            throw std::invalid_argument(
                "The bandwidth matrix must be a square matrix with shape "
                "(" +
                std::to_string(m_variables.size()) + ", " + std::to_string(m_variables.size()) + ")");

        m_bandwidth = new_bandwidth;
        if (m_bandwidth.rows() > 0) copy_bandwidth_opencl();
    }

    cl::Buffer& training_buffer() { return m_training; }
    const cl::Buffer& training_buffer() const { return m_training; }

    cl::Buffer& cholesky_buffer() { return m_H_cholesky; }
    const cl::Buffer& cholesky_buffer() const { return m_H_cholesky; }

    double lognorm_const() const { return m_lognorm_const; }

    DataFrame training_data() const;

    int num_instances() const {
        check_fitted();
        return N;
    }
    int num_variables() const { return m_variables.size(); }
    bool fitted() const { return m_fitted; }

    std::shared_ptr<arrow::DataType> data_type() const {
        check_fitted();
        return m_training_type;
    }

    std::shared_ptr<BandwidthSelector> bandwidth_type() const { return m_bselector; }

    VectorXd logl(const DataFrame& df) const;

    template <typename ArrowType>
    cl::Buffer logl_buffer(const DataFrame& df) const;
    template <typename ArrowType>
    cl::Buffer logl_buffer(const DataFrame& df, Buffer_ptr& bitmap) const;

    double slogl(const DataFrame& df) const;

    void save(const std::string name) { util::save_object(*this, name); }

    py::tuple __getstate__() const;
    static KDE __setstate__(py::tuple& t);
    static KDE __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    void check_fitted() const {
        if (!fitted()) throw std::invalid_argument("KDE factor not fitted.");
    }
    template <typename ArrowType>
    DataFrame _training_data() const;

    template <typename ArrowType, bool contains_null>
    void _fit(const DataFrame& df);

    template <typename ArrowType>
    VectorXd _logl(const DataFrame& df) const;
    template <typename ArrowType>
    double _slogl(const DataFrame& df) const;

    template <typename ArrowType, typename KDEType>
    cl::Buffer _logl_impl(cl::Buffer& test_buffer, int m) const;

    void copy_bandwidth_opencl();

    template <typename ArrowType>
    py::tuple __getstate__() const;

    std::vector<std::string> m_variables;
    bool m_fitted;
    std::shared_ptr<BandwidthSelector> m_bselector;
    MatrixXd m_bandwidth;
    cl::Buffer m_H_cholesky;
    cl::Buffer m_training;
    double m_lognorm_const;
    size_t N;
    std::shared_ptr<arrow::DataType> m_training_type;
};

template <typename ArrowType>
DataFrame KDE::_training_data() const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    arrow::NumericBuilder<ArrowType> builder;

    auto& opencl = OpenCLConfig::get();
    VectorType tmp_buffer(N * m_variables.size());
    opencl.read_from_buffer(tmp_buffer.data(), m_training, N * m_variables.size());

    std::vector<Array_ptr> columns;
    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);
    for (size_t i = 0; i < m_variables.size(); ++i) {
        auto status = builder.Resize(N);
        RAISE_STATUS_ERROR(builder.AppendValues(tmp_buffer.data() + i * N, N));

        Array_ptr out;
        RAISE_STATUS_ERROR(builder.Finish(&out));

        columns.push_back(out);
        builder.Reset();

        auto f = arrow::field(m_variables[i], out->type());
        RAISE_STATUS_ERROR(b.AddField(f));
    }

    RAISE_RESULT_ERROR(auto schema, b.Finish())

    auto rb = arrow::RecordBatch::Make(schema, N, columns);
    return DataFrame(rb);
}
/**
 * @brief Private function to learn the KDE parameters given the training data.
 * Used in the public function fit in KDE.cpp.
 *
 * @tparam ArrowType Arrow data type.
 * @tparam contains_null Boolean indicating if the training data contains null values.
 * @param df Training data.
 */
template <typename ArrowType, bool contains_null>
void KDE::_fit(const DataFrame& df) {
    using CType = typename ArrowType::c_type;

    auto d = m_variables.size();
    // NOTE: Here the positive definiteness of the bandwidth is checked
    try {
        m_bandwidth = m_bselector->bandwidth(df, m_variables);
    } catch (util::singular_covariance_data& e) {
        std::cerr << "KDE::_fit:\t" << e.what() << std::endl;
        throw e;
    }

    // Calculates the LLT decomposition matrix of the bandwidth matrix
    auto llt_cov = m_bandwidth.llt();
    auto cholesky = llt_cov.matrixLLT();

    auto& opencl = OpenCLConfig::get();

    if constexpr (std::is_same_v<CType, double>) {
        m_H_cholesky = opencl.copy_to_buffer(cholesky.data(), d * d);
    } else {
        using MatrixType = Matrix<CType, Dynamic, Dynamic>;
        MatrixType casted_cholesky = cholesky.template cast<CType>();
        m_H_cholesky = opencl.copy_to_buffer(casted_cholesky.data(), d * d);
    }

    auto training_data = df.to_eigen<false, ArrowType, contains_null>(m_variables);
    N = training_data->rows();
    m_training = opencl.copy_to_buffer(training_data->data(), N * d);

    // NOTE: The determinant of the bandwidth matrix is the product of the diagonal elements of the cholesky
    // - log(|h|) - 1/2 * d * log(2 * pi) - log(N)
    m_lognorm_const = -cholesky.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<double>) - std::log(N);
}

/**
 * @brief Learns the KDE parameters given the bandwidth matrix, the training data, the training type (?) and the number
 * of training instances.
 *
 * @tparam ArrowType Arrow data type.
 * @tparam EigenMatrix Eigen matrix type.
 * @param bandwidth Bandwidth matrix.
 * @param training_data Training data.
 * @param training_type Training type.
 * @param training_instances Number of training instances.
 */
template <typename ArrowType, typename EigenMatrix>
void KDE::fit(EigenMatrix bandwidth,
              cl::Buffer training_data,
              std::shared_ptr<arrow::DataType> training_type,
              int training_instances) {
    using CType = typename ArrowType::c_type;

    if ((bandwidth.rows() != bandwidth.cols()) || (static_cast<size_t>(bandwidth.rows()) != m_variables.size())) {
        throw std::invalid_argument("Bandwidth matrix must be a square matrix with dimensionality " +
                                    std::to_string(m_variables.size()));
    }

    m_bandwidth = bandwidth;
    auto d = m_variables.size();
    auto llt_cov = bandwidth.llt();
    auto cholesky = llt_cov.matrixLLT();
    auto& opencl = OpenCLConfig::get();

    if constexpr (std::is_same_v<CType, double>) {
        m_H_cholesky = opencl.copy_to_buffer(cholesky.data(), d * d);
    } else {
        using MatrixType = Matrix<CType, Dynamic, Dynamic>;
        MatrixType casted_cholesky = cholesky.template cast<CType>();
        m_H_cholesky = opencl.copy_to_buffer(casted_cholesky.data(), d * d);
    }

    m_training = training_data;
    m_training_type = training_type;
    N = training_instances;

    // NOTE: The determinant of the bandwidth matrix is the product of the diagonal elements of the cholesky
    m_lognorm_const = -cholesky.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<double>) - std::log(N);
    m_fitted = true;
}

/**
 * @brief Calculates Log-likelihood of the given data with OpenCL.
 *
 * @tparam ArrowType Arrow data type.
 * @param df Data.
 * @return VectorXd Log-likelihood values.
 */
template <typename ArrowType>
VectorXd KDE::_logl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto logl_buff = logl_buffer<ArrowType>(df);
    auto& opencl = OpenCLConfig::get();
    if (df.null_count(m_variables) == 0) {  // No null variables -> Returns the data?
        VectorType read_data(df->num_rows());
        opencl.read_from_buffer(read_data.data(), logl_buff, df->num_rows());
        if constexpr (!std::is_same_v<CType, double>)
            return read_data.template cast<double>();
        else
            return read_data;
    } else {  // Null variables -> Returns the data without nulls
        auto m = df.valid_rows(m_variables);
        VectorType read_data(m);
        auto bitmap = df.combined_bitmap(m_variables);
        auto bitmap_data = bitmap->data();

        opencl.read_from_buffer(read_data.data(), logl_buff, m);

        VectorXd res(df->num_rows());

        for (int i = 0, k = 0; i < df->num_rows(); ++i) {
            if (util::bit_util::GetBit(bitmap_data, i)) {
                res(i) = static_cast<double>(read_data[k++]);
            } else {
                res(i) = util::nan<double>;
            }
        }

        return res;
    }
}

template <typename ArrowType>
double KDE::_slogl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;

    auto logl_buff = logl_buffer<ArrowType>(df);
    auto m = df.valid_rows(m_variables);

    auto& opencl = OpenCLConfig::get();
    auto buffer_sum = opencl.sum1d<ArrowType>(logl_buff, m);

    CType result = 0;
    opencl.read_from_buffer(&result, buffer_sum, 1);
    return static_cast<double>(result);
}
/**
 * @brief Calculates the log-likelihood of the given data using _logl_impl.
 *
 * @tparam ArrowType Arrow data type.
 * @param df Data.
 * @return cl::Buffer Log-likelihood values.
 */
template <typename ArrowType>
cl::Buffer KDE::logl_buffer(const DataFrame& df) const {
    auto& opencl = OpenCLConfig::get();

    auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
    auto m = test_matrix->rows();
    auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m * m_variables.size());

    if (m_variables.size() == 1)
        return _logl_impl<ArrowType, UnivariateKDE>(test_buffer, m);
    else
        return _logl_impl<ArrowType, MultivariateKDE>(test_buffer, m);
}

/**
 * @brief Calculates the log-likelihood of the given data using _logl_impl.
 *
 * @tparam ArrowType Arrow data type.
 * @param df Data.
 * @param bitmap Bitmap.
 * @return cl::Buffer Log-likelihood values.
 */
template <typename ArrowType>
cl::Buffer KDE::logl_buffer(const DataFrame& df, Buffer_ptr& bitmap) const {
    auto& opencl = OpenCLConfig::get();

    auto test_matrix = df.to_eigen<false, ArrowType>(bitmap, m_variables);
    auto m = test_matrix->rows();
    auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m * m_variables.size());

    if (m_variables.size() == 1)
        return _logl_impl<ArrowType, UnivariateKDE>(test_buffer, m);
    else
        return _logl_impl<ArrowType, MultivariateKDE>(test_buffer, m);
}

/**
 * @brief Function where the log-likelihood are calculated with OpenCL?.
 *
 * @tparam ArrowType Arrow data type.
 * @tparam KDEType KDE type.
 * @param test_buffer Test data.
 * @param m Number of test instances.
 * @return cl::Buffer Log-likelihood values.
 */
template <typename ArrowType, typename KDEType>
cl::Buffer KDE::_logl_impl(cl::Buffer& test_buffer, int m) const {
    using CType = typename ArrowType::c_type;
    auto d = m_variables.size();
    auto& opencl = OpenCLConfig::get();
    auto res = opencl.new_buffer<CType>(m);

    auto [mat_logls, allocated_m] = opencl.allocate_temp_mat<ArrowType>(N, m);
    auto iterations = static_cast<int>(std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m)));

    cl::Buffer tmp_mat_buffer;
    if constexpr (std::is_same_v<KDEType, MultivariateKDE>) {
        if (N > allocated_m)
            tmp_mat_buffer = opencl.new_buffer<CType>(N * m_variables.size());
        else
            tmp_mat_buffer = opencl.new_buffer<CType>(allocated_m * m_variables.size());
    }

    for (auto i = 0; i < (iterations - 1); ++i) {
        KDEType::template execute_logl_mat<ArrowType>(m_training,
                                                      N,
                                                      test_buffer,
                                                      m,
                                                      i * allocated_m,
                                                      allocated_m,
                                                      d,
                                                      m_H_cholesky,
                                                      m_lognorm_const,
                                                      tmp_mat_buffer,
                                                      mat_logls);
        // Calculates the log-likelihood values for each test instance
        opencl.logsumexp_cols_offset<ArrowType>(mat_logls, N, allocated_m, res, i * allocated_m);
    }
    auto remaining_m = m - (iterations - 1) * allocated_m;

    KDEType::template execute_logl_mat<ArrowType>(m_training,
                                                  N,
                                                  test_buffer,
                                                  m,
                                                  m - remaining_m,
                                                  remaining_m,
                                                  d,
                                                  m_H_cholesky,
                                                  m_lognorm_const,
                                                  tmp_mat_buffer,
                                                  mat_logls);
    // Calculates the log-likelihood values for each test instance
    opencl.logsumexp_cols_offset<ArrowType>(mat_logls, N, remaining_m, res, (iterations - 1) * allocated_m);

    return res;
}

template <typename ArrowType>
py::tuple KDE::__getstate__() const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    MatrixXd bw;
    VectorType training_data;
    double lognorm_const = -1;
    int N_export = -1;
    int training_type = -1;

    if (m_fitted) {
        auto& opencl = OpenCLConfig::get();
        training_data = VectorType(N * m_variables.size());
        opencl.read_from_buffer(training_data.data(), m_training, N * m_variables.size());

        lognorm_const = m_lognorm_const;
        training_type = static_cast<int>(m_training_type->id());
        N_export = N;
        bw = m_bandwidth;
    }

    return py::make_tuple(
        m_variables, m_fitted, m_bselector, bw, training_data, lognorm_const, N_export, training_type);
}

}  // namespace kde

#endif  // PYBNESIAN_KDE_KDE_HPP