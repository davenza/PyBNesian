#include <kde/UCV.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <util/math_constants.hpp>
#include <util/vech_ops.hpp>
#include <nlopt.hpp>

using Eigen::LLT;
using opencl::OpenCLConfig, opencl::OpenCL_kernel_traits;

namespace kde {

class UnivariateUCVScore {
public:
    template <typename ArrowType>
    static void sum_triangular_scores(const cl::Buffer& training_data,
                                      const unsigned int,
                                      const unsigned int,
                                      const unsigned int index_offset,
                                      const unsigned int length,
                                      const cl::Buffer& cholesky,
                                      typename ArrowType::c_type lognorm_2H,
                                      typename ArrowType::c_type lognorm_H,
                                      cl::Buffer&,
                                      cl::Buffer& output_2h,
                                      cl::Buffer& output_h);
};

template <typename ArrowType>
void UnivariateUCVScore::sum_triangular_scores(const cl::Buffer& training_data,
                                               const unsigned int,
                                               const unsigned int,
                                               const unsigned int index_offset,
                                               const unsigned int length,
                                               const cl::Buffer& cholesky,
                                               typename ArrowType::c_type lognorm_2H,
                                               typename ArrowType::c_type lognorm_H,
                                               cl::Buffer&,
                                               cl::Buffer& output_2h,
                                               cl::Buffer& output_h) {
    auto& opencl = OpenCLConfig::get();
    auto& k_sum_ucv_1d = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_ucv_1d);
    k_sum_ucv_1d.setArg(0, training_data);
    k_sum_ucv_1d.setArg(1, index_offset);
    k_sum_ucv_1d.setArg(2, cholesky);
    k_sum_ucv_1d.setArg(3, lognorm_2H);
    k_sum_ucv_1d.setArg(4, lognorm_H);
    k_sum_ucv_1d.setArg(5, output_2h);
    k_sum_ucv_1d.setArg(6, output_h);

    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_sum_ucv_1d, cl::NullRange, cl::NDRange(length), cl::NullRange));
}

class ProductUCVScore {
public:
    template <typename ArrowType>
    static void sum_triangular_scores(const cl::Buffer& training_data,
                                      const unsigned int,
                                      const unsigned int,
                                      const unsigned int index_offset,
                                      const unsigned int length,
                                      const cl::Buffer& cholesky,
                                      typename ArrowType::c_type lognorm_2H,
                                      typename ArrowType::c_type lognorm_H,
                                      cl::Buffer&,
                                      cl::Buffer& output_2h,
                                      cl::Buffer& output_h);
};

template <typename ArrowType>
void ProductUCVScore::sum_triangular_scores(const cl::Buffer& training_data,
                                            const unsigned int training_rows,
                                            const unsigned int training_cols,
                                            const unsigned int index_offset,
                                            const unsigned int length,
                                            const cl::Buffer& h_vector,
                                            typename ArrowType::c_type lognorm_2H,
                                            typename ArrowType::c_type lognorm_H,
                                            cl::Buffer& tmp_h,
                                            cl::Buffer& output_2h,
                                            cl::Buffer& output_h) {
    auto& opencl = OpenCLConfig::get();
    auto& k_ucv_diag = opencl.kernel(OpenCL_kernel_traits<ArrowType>::ucv_diag);
    k_ucv_diag.setArg(0, training_data);
    k_ucv_diag.setArg(1, index_offset);
    k_ucv_diag.setArg(2, h_vector);
    k_ucv_diag.setArg(3, tmp_h);
    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_ucv_diag, cl::NullRange, cl::NDRange(length), cl::NullRange));

    auto& k_sum_ucv_diag = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_ucv_diag);
    k_sum_ucv_diag.setArg(0, training_data);
    k_sum_ucv_diag.setArg(1, training_rows);
    k_sum_ucv_diag.setArg(2, index_offset);
    k_sum_ucv_diag.setArg(3, h_vector);
    k_sum_ucv_diag.setArg(5, tmp_h);

    for (unsigned int i = 1; i < training_cols; ++i) {
        k_sum_ucv_diag.setArg(4, i);
        RAISE_ENQUEUEKERNEL_ERROR(
            queue.enqueueNDRangeKernel(k_sum_ucv_diag, cl::NullRange, cl::NDRange(length), cl::NullRange));
    }

    auto& k_copy_ucv_diag = opencl.kernel(OpenCL_kernel_traits<ArrowType>::copy_ucv_diag);
    k_copy_ucv_diag.setArg(0, tmp_h);
    k_copy_ucv_diag.setArg(1, lognorm_2H);
    k_copy_ucv_diag.setArg(2, lognorm_H);
    k_copy_ucv_diag.setArg(3, output_2h);
    k_copy_ucv_diag.setArg(4, output_h);
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_copy_ucv_diag, cl::NullRange, cl::NDRange(length), cl::NullRange));
}

class MultivariateUCVScore {
public:
    template <typename ArrowType>
    static void sum_triangular_scores(const cl::Buffer& training_data,
                                      const unsigned int training_rows,
                                      const unsigned int training_cols,
                                      const unsigned int index_offset,
                                      const unsigned int length,
                                      const cl::Buffer& cholesky,
                                      typename ArrowType::c_type lognorm_2H,
                                      typename ArrowType::c_type lognorm_H,
                                      cl::Buffer& tmp_diff_mat,
                                      cl::Buffer& output_2h,
                                      cl::Buffer& output_h);
};

template <typename ArrowType>
void MultivariateUCVScore::sum_triangular_scores(const cl::Buffer& training_data,
                                                 const unsigned int training_rows,
                                                 const unsigned int training_cols,
                                                 const unsigned int index_offset,
                                                 const unsigned int length,
                                                 const cl::Buffer& cholesky,
                                                 typename ArrowType::c_type lognorm_2H,
                                                 typename ArrowType::c_type lognorm_H,
                                                 cl::Buffer& tmp_diff_mat,
                                                 cl::Buffer& output_2h,
                                                 cl::Buffer& output_h) {
    auto& opencl = OpenCLConfig::get();
    auto& k_triangular_substract_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::triangular_substract_mat);
    k_triangular_substract_mat.setArg(0, training_data);
    k_triangular_substract_mat.setArg(1, training_rows);
    k_triangular_substract_mat.setArg(2, training_cols);
    k_triangular_substract_mat.setArg(3, index_offset);
    k_triangular_substract_mat.setArg(4, length);
    k_triangular_substract_mat.setArg(5, tmp_diff_mat);
    auto& queue = opencl.queue();
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(
        k_triangular_substract_mat, cl::NullRange, cl::NDRange(length * training_cols), cl::NullRange));

    auto& k_solve = opencl.kernel(OpenCL_kernel_traits<ArrowType>::solve);
    k_solve.setArg(0, tmp_diff_mat);
    k_solve.setArg(1, length);
    k_solve.setArg(2, training_cols);
    k_solve.setArg(3, cholesky);
    RAISE_ENQUEUEKERNEL_ERROR(queue.enqueueNDRangeKernel(k_solve, cl::NullRange, cl::NDRange(length), cl::NullRange));

    auto& k_square = opencl.kernel(OpenCL_kernel_traits<ArrowType>::square);
    k_square.setArg(0, tmp_diff_mat);
    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_square, cl::NullRange, cl::NDRange(length * training_cols), cl::NullRange));

    auto& k_sum_ucv_mat = opencl.kernel(OpenCL_kernel_traits<ArrowType>::sum_ucv_mat);
    k_sum_ucv_mat.setArg(0, tmp_diff_mat);
    k_sum_ucv_mat.setArg(1, training_cols);
    k_sum_ucv_mat.setArg(2, lognorm_2H);
    k_sum_ucv_mat.setArg(3, lognorm_H);
    k_sum_ucv_mat.setArg(4, output_2h);
    k_sum_ucv_mat.setArg(5, output_h);

    RAISE_ENQUEUEKERNEL_ERROR(
        queue.enqueueNDRangeKernel(k_sum_ucv_mat, cl::NullRange, cl::NDRange(length), cl::NullRange));
}

template <typename ArrowType, bool contains_null>
cl::Buffer UCVScorer::_copy_training_data(const DataFrame& df, const std::vector<std::string>& variables) const {
    auto training_data = df.to_eigen<false, ArrowType, contains_null>(variables);
    auto& opencl = OpenCLConfig::get();
    return opencl.copy_to_buffer(training_data->data(), training_data->rows() * variables.size());
}

cl::Buffer UCVScorer::_copy_training_data(const DataFrame& df, const std::vector<std::string>& variables) const {
    bool contains_null = df.null_count(variables) > 0;
    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (contains_null)
                return _copy_training_data<arrow::DoubleType, true>(df, variables);
            else
                return _copy_training_data<arrow::DoubleType, false>(df, variables);
            break;
        }
        case Type::FLOAT: {
            if (contains_null)
                return _copy_training_data<arrow::FloatType, true>(df, variables);
            else
                return _copy_training_data<arrow::FloatType, true>(df, variables);
            break;
        }
        default:
            throw std::invalid_argument("Wrong data type to score UCV. [double] or [float] data is expected.");
    }
}

template <typename ArrowType>
std::pair<cl::Buffer, typename ArrowType::c_type> UCVScorer::copy_diagonal_bandwidth(
    const Matrix<typename ArrowType::c_type, Dynamic, 1>& diagonal_sqrt_bandwidth) const {
    using CType = typename ArrowType::c_type;
    auto& opencl = OpenCLConfig::get();
    auto bw = opencl.copy_to_buffer(diagonal_sqrt_bandwidth.data(), d);

    auto lognorm_const = -diagonal_sqrt_bandwidth.array().log().sum() - 0.5 * d * std::log(2 * util::pi<CType>);
    return std::make_pair(bw, lognorm_const);
}

template <typename ArrowType>
std::pair<cl::Buffer, typename ArrowType::c_type> UCVScorer::copy_unconstrained_bandwidth(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth) const {
    using CType = typename ArrowType::c_type;
    auto llt_cov = bandwidth.llt();
    auto llt_matrix = llt_cov.matrixLLT();

    auto& opencl = OpenCLConfig::get();
    auto cholesky = opencl.copy_to_buffer(llt_matrix.data(), d * d);

    auto lognorm_const = -llt_matrix.diagonal().array().log().sum() - 0.5 * d * std::log(2 * util::pi<CType>);

    return std::make_pair(cholesky, lognorm_const);
}

template <typename ArrowType>
double UCVScorer::score_diagonal_impl(
    const Matrix<typename ArrowType::c_type, Dynamic, 1>& diagonal_sqrt_bandwidth) const {
    using CType = typename ArrowType::c_type;
    auto [bw, lognorm_H] = copy_diagonal_bandwidth<ArrowType>(diagonal_sqrt_bandwidth);
    auto lognorm_2H = lognorm_H - 0.5 * d * std::log(2.);

    auto& opencl = OpenCLConfig::get();

    auto n_distances = N * (N - 1) / 2;

    auto instances_per_iteration = std::min(static_cast<size_t>(1000000), n_distances);
    auto iterations =
        static_cast<int>(std::ceil(static_cast<double>(n_distances) / static_cast<double>(instances_per_iteration)));

    cl::Buffer sum2h = opencl.new_buffer<CType>(instances_per_iteration);
    opencl.fill_buffer<CType>(sum2h, 0., instances_per_iteration);
    cl::Buffer sumh = opencl.new_buffer<CType>(instances_per_iteration);
    opencl.fill_buffer<CType>(sumh, 0., instances_per_iteration);

    cl::Buffer temp_h = opencl.new_buffer<CType>(instances_per_iteration);

    for (auto i = 0; i < (iterations - 1); ++i) {
        ProductUCVScore::sum_triangular_scores<ArrowType>(m_training,
                                                          N,
                                                          d,
                                                          i * instances_per_iteration,
                                                          instances_per_iteration,
                                                          bw,
                                                          lognorm_2H,
                                                          lognorm_H,
                                                          temp_h,
                                                          sum2h,
                                                          sumh);
    }

    auto remaining = n_distances - (iterations - 1) * instances_per_iteration;

    ProductUCVScore::sum_triangular_scores<ArrowType>(m_training,
                                                      N,
                                                      d,
                                                      (iterations - 1) * instances_per_iteration,
                                                      remaining,
                                                      bw,
                                                      lognorm_2H,
                                                      lognorm_H,
                                                      temp_h,
                                                      sum2h,
                                                      sumh);

    auto b2h = opencl.sum1d<ArrowType>(sum2h, instances_per_iteration);
    auto bh = opencl.sum1d<ArrowType>(sumh, instances_per_iteration);

    CType s2h, sh;
    opencl.read_from_buffer(&s2h, b2h, 1);
    opencl.read_from_buffer(&sh, bh, 1);

    // Returns UCV scaled by N: N * UCV
    return std::exp(lognorm_2H) + 2 * s2h / N - 4 * sh / (N - 1);
}

template <typename ArrowType, typename UCVScore>
double UCVScorer::score_unconstrained_impl(
    const Matrix<typename ArrowType::c_type, Dynamic, Dynamic>& bandwidth) const {
    using CType = typename ArrowType::c_type;
    auto [cholesky, lognorm_H] = copy_unconstrained_bandwidth<ArrowType>(bandwidth);
    auto lognorm_2H = lognorm_H - 0.5 * d * std::log(2.);

    auto& opencl = OpenCLConfig::get();

    auto n_distances = N * (N - 1) / 2;

    auto instances_per_iteration = std::min(static_cast<size_t>(1000000), n_distances);
    auto iterations =
        static_cast<int>(std::ceil(static_cast<double>(n_distances) / static_cast<double>(instances_per_iteration)));

    cl::Buffer sum2h = opencl.new_buffer<CType>(instances_per_iteration);
    opencl.fill_buffer<CType>(sum2h, 0., instances_per_iteration);
    cl::Buffer sumh = opencl.new_buffer<CType>(instances_per_iteration);
    opencl.fill_buffer<CType>(sumh, 0., instances_per_iteration);

    cl::Buffer tmp_mat_buffer;
    if constexpr (std::is_same_v<UCVScore, MultivariateUCVScore>) {
        tmp_mat_buffer = opencl.new_buffer<CType>(instances_per_iteration * d);
    }

    for (auto i = 0; i < (iterations - 1); ++i) {
        UCVScore::template sum_triangular_scores<ArrowType>(m_training,
                                                            N,
                                                            d,
                                                            i * instances_per_iteration,
                                                            instances_per_iteration,
                                                            cholesky,
                                                            lognorm_2H,
                                                            lognorm_H,
                                                            tmp_mat_buffer,
                                                            sum2h,
                                                            sumh);
    }

    auto remaining = n_distances - (iterations - 1) * instances_per_iteration;

    UCVScore::template sum_triangular_scores<ArrowType>(m_training,
                                                        N,
                                                        d,
                                                        (iterations - 1) * instances_per_iteration,
                                                        remaining,
                                                        cholesky,
                                                        lognorm_2H,
                                                        lognorm_H,
                                                        tmp_mat_buffer,
                                                        sum2h,
                                                        sumh);

    auto b2h = opencl.sum1d<ArrowType>(sum2h, instances_per_iteration);
    auto bh = opencl.sum1d<ArrowType>(sumh, instances_per_iteration);

    CType s2h, sh;
    opencl.read_from_buffer(&s2h, b2h, 1);
    opencl.read_from_buffer(&sh, bh, 1);

    // Returns UCV scaled by N: N * UCV
    return std::exp(lognorm_2H) + 2 * s2h / N - 4 * sh / (N - 1);
}

double UCVScorer::score_diagonal(const VectorXd& diagonal_bandwidth) const {
    if (d != static_cast<size_t>(diagonal_bandwidth.rows()))
        throw std::invalid_argument("Wrong dimension for bandwidth vector. it should be a " + std::to_string(d) +
                                    " vector.");

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            return score_diagonal_impl<arrow::DoubleType>(diagonal_bandwidth.cwiseSqrt());
        }
        case Type::FLOAT: {
            return score_diagonal_impl<arrow::FloatType>(diagonal_bandwidth.template cast<float>().cwiseSqrt());
        }
        default:
            throw std::runtime_error("Unreachable code");
    }
}

double UCVScorer::score_unconstrained(const MatrixXd& bandwidth) const {
    if (d != static_cast<size_t>(bandwidth.rows()) && d != static_cast<size_t>(bandwidth.cols()))
        throw std::invalid_argument("Wrong dimension for bandwidth matrix. it should be a " + std::to_string(d) + "x" +
                                    std::to_string(d) + " matrix.");

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (d == 1)
                return score_unconstrained_impl<arrow::DoubleType, UnivariateUCVScore>(bandwidth);
            else
                return score_unconstrained_impl<arrow::DoubleType, MultivariateUCVScore>(bandwidth);
        }
        case Type::FLOAT: {
            if (d == 1)
                return score_unconstrained_impl<arrow::FloatType, UnivariateUCVScore>(bandwidth.template cast<float>());
            else
                return score_unconstrained_impl<arrow::FloatType, MultivariateUCVScore>(
                    bandwidth.template cast<float>());
        }
        default:
            throw std::runtime_error("Unreachable code");
    }
}

struct UCVOptimInfo {
    UCVScorer ucv_scorer;
    double start_score;
    double start_determinant;
};

double wrap_ucv_diag_optim(unsigned n, const double* x, double*, void* my_func_data) {
    using MapType = Eigen::Map<const VectorXd>;
    MapType xm(x, n);

    UCVOptimInfo& optim_info = *reinterpret_cast<UCVOptimInfo*>(my_func_data);

    auto det_sqrt = xm.prod();
    auto det = det_sqrt * det_sqrt;

    if (det <= util::machine_tol || det < 1e-3 * optim_info.start_determinant ||
        det > 1e3 * optim_info.start_determinant)
        return optim_info.start_score + 10e-8;

    auto score = optim_info.ucv_scorer.score_diagonal(xm.array().square().matrix());

    if (std::abs(score) > 1e3 * std::abs(optim_info.start_score)) return optim_info.start_score + 10e-8;

    return score;
}

double wrap_ucv_optim(unsigned n, const double* x, double*, void* my_func_data) {
    using MapType = Eigen::Map<const VectorXd>;
    MapType xm(x, n);

    auto sqrt = util::invvech_triangular(xm);
    auto H = sqrt * sqrt.transpose();

    UCVOptimInfo& optim_info = *reinterpret_cast<UCVOptimInfo*>(my_func_data);

    auto det = std::exp(2 * sqrt.diagonal().array().log().sum());

    // Avoid too small/large determinants returning the start score.
    // Package ks uses 1e10 as constant.
    if (det <= util::machine_tol || det < 1e-3 * optim_info.start_determinant ||
        det > 1e3 * optim_info.start_determinant || std::isnan(det))
        return optim_info.start_score + 10e-8;

    auto score = optim_info.ucv_scorer.score_unconstrained(H);

    // Avoid scores with too much difference.
    if (std::abs(score) > 1e3 * std::abs(optim_info.start_score)) return optim_info.start_score + 10e-8;

    return score;
}

VectorXd UCV::diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
    if (variables.empty()) return VectorXd(0);

    NormalReferenceRule nr;

    auto normal_bandwidth = nr.diag_bandwidth(df, variables);

    UCVScorer ucv_scorer(df, variables);
    auto start_score = ucv_scorer.score_unconstrained(normal_bandwidth);
    auto start_determinant = normal_bandwidth.prod();

    UCVOptimInfo optim_info{/*.ucv_scorer = */ ucv_scorer,
                            /*.start_score = */ start_score,
                            /*.start_determinant = */ start_determinant};

    auto start_bandwidth = normal_bandwidth.cwiseSqrt().eval();

    nlopt::opt opt(nlopt::LN_NELDERMEAD, start_bandwidth.rows());
    opt.set_min_objective(wrap_ucv_diag_optim, &optim_info);
    opt.set_ftol_rel(1e-4);
    opt.set_xtol_rel(1e-4);
    std::vector<double> x(start_bandwidth.rows());
    std::copy(start_bandwidth.data(), start_bandwidth.data() + start_bandwidth.rows(), x.data());
    double minf;

    try {
        opt.optimize(x, minf);
    } catch (std::exception& e) {
        throw std::invalid_argument(std::string("Failed optimizing bandwidth: ") + e.what());
    }

    std::copy(x.data(), x.data() + x.size(), start_bandwidth.data());

    return start_bandwidth.array().square().matrix();
}

MatrixXd UCV::bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
    if (variables.empty()) return MatrixXd(0, 0);

    NormalReferenceRule nr;

    auto normal_bandwidth = nr.bandwidth(df, variables);

    UCVScorer ucv_scorer(df, variables);
    auto start_score = ucv_scorer.score_unconstrained(normal_bandwidth);
    auto start_determinant = normal_bandwidth.determinant();
    UCVOptimInfo optim_info{/*.ucv_scorer = */ ucv_scorer,
                            /*.start_score = */ start_score,
                            /*.start_determinant = */ start_determinant};

    LLT<Eigen::Ref<MatrixXd>> start_sqrt(normal_bandwidth);
    auto start_vech = util::vech(start_sqrt.matrixL());

    nlopt::opt opt(nlopt::LN_NELDERMEAD, start_vech.rows());
    opt.set_min_objective(wrap_ucv_optim, &optim_info);
    opt.set_ftol_rel(1e-4);
    opt.set_xtol_rel(1e-4);
    std::vector<double> x(start_vech.rows());
    std::copy(start_vech.data(), start_vech.data() + start_vech.rows(), x.data());
    double minf;

    try {
        opt.optimize(x, minf);
    } catch (std::exception& e) {
        throw std::invalid_argument(std::string("Failed optimizing bandwidth: ") + e.what());
    }

    std::copy(x.data(), x.data() + x.size(), start_vech.data());

    auto sqrt = util::invvech_triangular(start_vech);
    auto H = sqrt * sqrt.transpose();

    return H;
}

}  // namespace kde