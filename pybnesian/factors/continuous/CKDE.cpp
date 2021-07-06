#include <factors/continuous/CKDE.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <models/BayesianNetwork.hpp>
#include <opencl/opencl_config.hpp>
#include <util/vech_ops.hpp>
#include <util/basic_eigen_ops.hpp>

#include <nlopt.hpp>

using factors::discrete::DiscreteFactorType;
using models::BayesianNetworkBase, models::ConditionalBayesianNetworkBase;
using opencl::OpenCLConfig;

namespace factors::continuous {

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
            throw py::value_error("Wrong data type to score UCV. [double] or [float] data is expected.");
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

    auto instances_per_iteration = std::min(1000000UL, n_distances);
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

    auto instances_per_iteration = std::min(1000000UL, n_distances);
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
            throw py::value_error("Unreachable code");
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
            throw py::value_error("Unreachable code");
    }
}

double wrap_ucv_diag_optim(unsigned n, const double* x, double*, void* my_func_data) {
    using MapType = Eigen::Map<const VectorXd>;
    MapType xm(x, n);

    UCVScorer& ucv_scorer = *reinterpret_cast<UCVScorer*>(my_func_data);
    auto score = ucv_scorer.score_diagonal(xm.array().square().matrix());

    return score;
}

double wrap_ucv_optim(unsigned n, const double* x, double*, void* my_func_data) {
    using MapType = Eigen::Map<const VectorXd>;
    MapType xm(x, n);

    auto sqrt = util::invvech(xm);
    auto H = sqrt * sqrt;

    UCVScorer& ucv_scorer = *reinterpret_cast<UCVScorer*>(my_func_data);
    auto score = ucv_scorer.score_unconstrained(H);

    return score;
}

VectorXd UCV::estimate_diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
    NormalReferenceRule nr;

    auto start_bandwidth = nr.estimate_diag_bandwidth(df, variables).cwiseSqrt().eval();

    UCVScorer ucv_scorer(df, variables);

    nlopt::opt opt(nlopt::LN_SBPLX, start_bandwidth.rows());
    opt.set_min_objective(wrap_ucv_diag_optim, &ucv_scorer);
    // opt.set_ftol_rel(1e-6);
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

MatrixXd UCV::estimate_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
    NormalReferenceRule nr;

    auto normal_bandwidth = nr.estimate_bandwidth(df, variables);

    auto start_sqrt = util::sqrt_matrix(normal_bandwidth);
    auto start_vech = util::vech(start_sqrt);

    UCVScorer ucv_scorer(df, variables);

    nlopt::opt opt(nlopt::LN_SBPLX, start_vech.rows());
    opt.set_min_objective(wrap_ucv_optim, &ucv_scorer);
    // opt.set_ftol_rel(1e-6);
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

    auto sqrt = util::invvech(start_vech);
    auto H = sqrt * sqrt;

    return H;
}

void KDE::copy_bandwidth_opencl() {
    auto d = m_variables.size();
    auto llt_cov = m_bandwidth.llt();
    auto llt_matrix = llt_cov.matrixLLT();

    m_lognorm_const = -llt_matrix.diagonal().array().log().sum() -
                      0.5 * m_variables.size() * std::log(2 * util::pi<double>) - std::log(N);

    auto& opencl = OpenCLConfig::get();

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            m_H_cholesky = opencl.copy_to_buffer(llt_matrix.data(), d * d);
            break;
        }
        case Type::FLOAT: {
            MatrixXf casted_cholesky = llt_matrix.template cast<float>();
            m_H_cholesky = opencl.copy_to_buffer(casted_cholesky.data(), d * d);
            break;
        }
        default:
            throw std::invalid_argument("Unreachable code.");
    }
}

DataFrame KDE::training_data() const {
    check_fitted();
    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return _training_data<arrow::DoubleType>();
        case Type::FLOAT:
            return _training_data<arrow::FloatType>();
        default:
            throw std::invalid_argument("Unreachable code.");
    }
}

void KDE::fit(const DataFrame& df) {
    m_training_type = df.same_type(m_variables);

    bool contains_null = df.null_count(m_variables) > 0;

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (contains_null)
                _fit<arrow::DoubleType, true>(df);
            else
                _fit<arrow::DoubleType, false>(df);
            break;
        }
        case Type::FLOAT: {
            if (contains_null)
                _fit<arrow::FloatType, true>(df);
            else
                _fit<arrow::FloatType, false>(df);
            break;
        }
        default:
            throw py::value_error("Wrong data type to fit KDE. [double] or [float] data is expected.");
    }

    m_fitted = true;
}

VectorXd KDE::logl(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _logl<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _logl<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

double KDE::slogl(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _slogl<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _slogl<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

py::tuple KDE::__getstate__() const {
    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return __getstate__<arrow::DoubleType>();
        case Type::FLOAT:
            return __getstate__<arrow::FloatType>();
        default:
            // Not fitted model.
            return __getstate__<arrow::DoubleType>();
    }
}

KDE KDE::__setstate__(py::tuple& t) {
    if (t.size() != 8) throw std::runtime_error("Not valid KDE.");

    KDE kde(t[0].cast<std::vector<std::string>>());

    kde.m_fitted = t[1].cast<bool>();
    kde.m_bselector = t[2].cast<std::shared_ptr<BandwidthEstimator>>();

    if (kde.m_fitted) {
        kde.m_bandwidth = t[3].cast<MatrixXd>();
        kde.m_lognorm_const = t[5].cast<double>();
        kde.N = static_cast<size_t>(t[6].cast<int>());
        kde.m_training_type = pyarrow::GetPrimitiveType(static_cast<arrow::Type::type>(t[7].cast<int>()));

        auto llt_cov = kde.m_bandwidth.llt();
        auto llt_matrix = llt_cov.matrixLLT();

        auto& opencl = OpenCLConfig::get();

        switch (kde.m_training_type->id()) {
            case Type::DOUBLE: {
                kde.m_H_cholesky =
                    opencl.copy_to_buffer(llt_matrix.data(), kde.m_variables.size() * kde.m_variables.size());

                auto training_data = t[4].cast<VectorXd>();
                kde.m_training = opencl.copy_to_buffer(training_data.data(), kde.N * kde.m_variables.size());
                break;
            }
            case Type::FLOAT: {
                MatrixXf casted_cholesky = llt_matrix.template cast<float>();
                kde.m_H_cholesky =
                    opencl.copy_to_buffer(casted_cholesky.data(), kde.m_variables.size() * kde.m_variables.size());

                auto training_data = t[4].cast<VectorXf>();
                kde.m_training = opencl.copy_to_buffer(training_data.data(), kde.N * kde.m_variables.size());
                break;
            }
            default:
                throw std::runtime_error("Not valid data type in KDE.");
        }
    }

    return kde;
}

void ProductKDE::copy_bandwidth_opencl() {
    m_cl_bandwidth.clear();
    auto& opencl = OpenCLConfig::get();

    for (size_t i = 0; i < m_variables.size(); ++i) {
        switch (m_training_type->id()) {
            case Type::DOUBLE: {
                auto sqrt = std::sqrt(m_bandwidth(i));
                m_cl_bandwidth.push_back(opencl.copy_to_buffer(&sqrt, 1));
                break;
            }
            case Type::FLOAT: {
                auto casted = std::sqrt(static_cast<float>(m_bandwidth(i)));
                m_cl_bandwidth.push_back(opencl.copy_to_buffer(&casted, 1));
                break;
            }
            default:
                throw std::invalid_argument("Unreachable code.");
        }
    }

    m_lognorm_const = -0.5 * m_variables.size() * std::log(2 * util::pi<double>) -
                      0.5 * m_bandwidth.array().log().sum() - std::log(N);
}

DataFrame ProductKDE::training_data() const {
    check_fitted();
    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return _training_data<arrow::DoubleType>();
        case Type::FLOAT:
            return _training_data<arrow::FloatType>();
        default:
            throw std::invalid_argument("Unreachable code.");
    }
}

void ProductKDE::fit(const DataFrame& df) {
    m_training_type = df.same_type(m_variables);

    bool contains_null = df.null_count(m_variables) > 0;

    switch (m_training_type->id()) {
        case Type::DOUBLE: {
            if (contains_null)
                _fit<arrow::DoubleType, true>(df);
            else
                _fit<arrow::DoubleType, false>(df);
            break;
        }
        case Type::FLOAT: {
            if (contains_null)
                _fit<arrow::FloatType, true>(df);
            else
                _fit<arrow::FloatType, false>(df);
            break;
        }
        default:
            throw py::value_error("Wrong data type to fit ProductKDE. [double] or [float] data is expected.");
    }

    m_fitted = true;
}

VectorXd ProductKDE::logl(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _logl<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _logl<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

double ProductKDE::slogl(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _slogl<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _slogl<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

py::tuple ProductKDE::__getstate__() const {
    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return __getstate__<arrow::DoubleType>();
        case Type::FLOAT:
            return __getstate__<arrow::FloatType>();
        default:
            // Not fitted model.
            return __getstate__<arrow::DoubleType>();
    }
}

ProductKDE ProductKDE::__setstate__(py::tuple& t) {
    if (t.size() != 8) throw std::runtime_error("Not valid ProductKDE.");

    ProductKDE kde(t[0].cast<std::vector<std::string>>());

    kde.m_fitted = t[1].cast<bool>();
    kde.m_bselector = t[2].cast<std::shared_ptr<BandwidthEstimator>>();

    if (kde.m_fitted) {
        kde.m_bandwidth = t[3].cast<VectorXd>();
        kde.m_lognorm_const = t[5].cast<double>();
        kde.N = static_cast<size_t>(t[6].cast<int>());
        kde.m_training_type = pyarrow::GetPrimitiveType(static_cast<arrow::Type::type>(t[7].cast<int>()));

        auto& opencl = OpenCLConfig::get();

        switch (kde.m_training_type->id()) {
            case Type::DOUBLE: {
                auto data = t[4].cast<std::vector<VectorXd>>();

                for (size_t i = 0; i < kde.m_variables.size(); ++i) {
                    kde.m_cl_bandwidth.push_back(opencl.copy_to_buffer(&kde.m_bandwidth(i), 1));
                    kde.m_training.push_back(opencl.copy_to_buffer(data[i].data(), kde.N));
                }

                break;
            }
            case Type::FLOAT: {
                auto data = t[4].cast<std::vector<VectorXf>>();

                for (size_t i = 0; i < kde.m_variables.size(); ++i) {
                    auto casted_bw = static_cast<float>(kde.m_bandwidth(i));

                    kde.m_cl_bandwidth.push_back(opencl.copy_to_buffer(&casted_bw, 1));
                    kde.m_training.push_back(opencl.copy_to_buffer(data[i].data(), kde.N));
                }

                break;
            }
            default:
                throw std::runtime_error("Not valid data type in ProductKDE.");
        }
    }

    return kde;
}

std::shared_ptr<Factor> CKDEType::new_factor(const BayesianNetworkBase& m,
                                             const std::string& variable,
                                             const std::vector<std::string>& evidence) const {
    for (const auto& e : evidence) {
        if (m.node_type(e) == DiscreteFactorType::get()) {
            return std::make_shared<DCKDE>(variable, evidence);
        }
    }

    return std::make_shared<CKDE>(variable, evidence);
}

std::shared_ptr<Factor> CKDEType::new_factor(const ConditionalBayesianNetworkBase& m,
                                             const std::string& variable,
                                             const std::vector<std::string>& evidence) const {
    for (const auto& e : evidence) {
        if (m.node_type(e) == DiscreteFactorType::get()) {
            return std::make_shared<DCKDE>(variable, evidence);
        }
    }

    return std::make_shared<CKDE>(variable, evidence);
}

void CKDE::fit(const DataFrame& df) {
    auto type = df.same_type(m_variables);

    m_training_type = type;
    switch (type->id()) {
        case Type::DOUBLE:
            _fit<arrow::DoubleType>(df);
            break;
        case Type::FLOAT:
            _fit<arrow::FloatType>(df);
            break;
        default:
            throw std::invalid_argument("Wrong data type to fit KDE. [double] or [float] data is expected.");
    }

    m_fitted = true;
}

VectorXd CKDE::logl(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _logl<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _logl<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

double CKDE::slogl(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _slogl<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _slogl<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

Array_ptr CKDE::sample(int n, const DataFrame& evidence_values, unsigned int seed) const {
    if (n < 0) {
        throw std::invalid_argument("n should be a non-negative number");
    }

    check_fitted();
    if (!this->evidence().empty()) {
        auto type = evidence_values.same_type(this->evidence());

        if (type->id() != m_training_type->id()) {
            throw std::invalid_argument("Data type of evidence values (" + type->name() +
                                        ") is different from CKDE training data (" + m_training_type->name() + ").");
        }
    }

    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return _sample<arrow::DoubleType>(n, evidence_values, seed);
        case Type::FLOAT:
            return _sample<arrow::FloatType>(n, evidence_values, seed);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

VectorXd CKDE::cdf(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _cdf<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _cdf<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

std::string CKDE::ToString() const {
    std::stringstream stream;
    const auto& e = this->evidence();
    if (!e.empty()) {
        stream << "[CKDE] P(" << this->variable() << " | " << e[0];

        for (size_t i = 1; i < e.size(); ++i) {
            stream << ", " << e[i];
        }

        if (m_fitted)
            stream << ") = CKDE with " << N << " instances";
        else
            stream << ") not fitted";
        return stream.str();
    } else {
        if (m_fitted)
            stream << "[CKDE] P(" << this->variable() << ") = CKDE with " << N << " instances";
        else
            stream << "[CKDE] P(" << this->variable() << ") not fitted";
        return stream.str();
    }
}

py::tuple CKDE::__getstate__() const {
    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return __getstate__<arrow::DoubleType>();
        case Type::FLOAT:
            return __getstate__<arrow::FloatType>();
        default:
            // Not fitted model.
            return __getstate__<arrow::DoubleType>();
    }
}

CKDE CKDE::__setstate__(py::tuple& t) {
    if (t.size() != 4) throw std::runtime_error("Not valid CKDE.");

    CKDE ckde(t[0].cast<std::string>(), t[1].cast<std::vector<std::string>>());

    ckde.m_fitted = t[2].cast<bool>();

    if (ckde.m_fitted) {
        auto joint_tuple = t[3].cast<py::tuple>();
        auto kde_joint = KDE::__setstate__(joint_tuple);
        ckde.m_bselector = kde_joint.bandwidth_type();
        ckde.m_training_type = kde_joint.data_type();
        ckde.N = kde_joint.num_instances();
        ckde.m_joint = std::move(kde_joint);

        if (!ckde.evidence().empty()) {
            auto& joint_bandwidth = ckde.m_joint.bandwidth();
            auto d = ckde.m_variables.size();
            auto marg_bandwidth = joint_bandwidth.bottomRightCorner(d - 1, d - 1);

            cl::Buffer& training_buffer = ckde.m_joint.training_buffer();

            auto& opencl = OpenCLConfig::get();

            switch (ckde.m_training_type->id()) {
                case Type::DOUBLE: {
                    auto marg_buffer = opencl.copy_buffer<double>(training_buffer, ckde.N, ckde.N * (d - 1));
                    ckde.m_marg.fit<arrow::DoubleType>(marg_bandwidth, marg_buffer, ckde.m_joint.data_type(), ckde.N);
                    break;
                }
                case Type::FLOAT: {
                    auto marg_buffer = opencl.copy_buffer<float>(training_buffer, ckde.N, ckde.N * (d - 1));
                    ckde.m_marg.fit<arrow::FloatType>(marg_bandwidth, marg_buffer, ckde.m_joint.data_type(), ckde.N);
                    break;
                }
                default:
                    throw std::invalid_argument("Wrong data type in CKDE.");
            }
        }
    }

    return ckde;
}

}  // namespace factors::continuous