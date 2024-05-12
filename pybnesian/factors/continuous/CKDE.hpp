#ifndef PYBNESIAN_FACTORS_CONTINUOUS_CKDE_HPP
#define PYBNESIAN_FACTORS_CONTINUOUS_CKDE_HPP

#include <random>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <dataset/dataset.hpp>
#include <factors/factors.hpp>
#include <factors/discrete/DiscreteAdaptator.hpp>
#include <kde/BandwidthSelector.hpp>
#include <kde/NormalReferenceRule.hpp>
#include <kde/KDE.hpp>
#include <opencl/opencl_config.hpp>
#include <util/math_constants.hpp>

namespace py = pybind11;

using dataset::DataFrame;
using Eigen::VectorXd, Eigen::VectorXi;
using factors::FactorType, factors::discrete::DiscreteAdaptator;
using kde::KDE, kde::BandwidthSelector, kde::NormalReferenceRule, kde::UnivariateKDE, kde::MultivariateKDE;
using opencl::OpenCLConfig, opencl::OpenCL_kernel_traits;

namespace factors::continuous {

class CKDEType : public FactorType {
public:
    CKDEType(const CKDEType&) = delete;
    void operator=(const CKDEType&) = delete;

    static std::shared_ptr<CKDEType> get() {
        static std::shared_ptr<CKDEType> singleton = std::shared_ptr<CKDEType>(new CKDEType);
        return singleton;
    }

    static CKDEType& get_ref() {
        static CKDEType& ref = *CKDEType::get();
        return ref;
    }

    std::shared_ptr<Factor> new_factor(const BayesianNetworkBase&,
                                       const std::string&,
                                       const std::vector<std::string>&,
                                       py::args = py::args{},
                                       py::kwargs = py::kwargs{}) const override;
    std::shared_ptr<Factor> new_factor(const ConditionalBayesianNetworkBase&,
                                       const std::string&,
                                       const std::vector<std::string>&,
                                       py::args = py::args{},
                                       py::kwargs = py::kwargs{}) const override;

    std::string ToString() const override { return "CKDEFactor"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<CKDEType> __setstate__(py::tuple&) { return CKDEType::get(); }

private:
    CKDEType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

class CKDE : public Factor {
public:
    using FactorTypeClass = CKDEType;

    CKDE() = default;
    CKDE(std::string variable, std::vector<std::string> evidence)
        : CKDE(variable, evidence, std::make_shared<NormalReferenceRule>()) {}
    CKDE(std::string variable, std::vector<std::string> evidence, std::shared_ptr<BandwidthSelector> b_selector)
        : Factor(variable, evidence),
          m_variables(),
          m_fitted(false),
          m_bselector(b_selector),
          m_training_type(arrow::float64()),
          m_joint(),
          m_marg() {
        if (b_selector == nullptr) throw std::runtime_error("Bandwidth selector procedure must be non-null.");

        m_variables.reserve(evidence.size() + 1);
        m_variables.push_back(variable);
        for (auto it = evidence.begin(); it != evidence.end(); ++it) {
            m_variables.push_back(*it);
        }

        m_joint = KDE(m_variables, b_selector);
        if (!this->evidence().empty()) {
            m_marg = KDE(this->evidence(), b_selector);
        }
    }

    std::shared_ptr<FactorType> type() const override { return CKDEType::get(); }

    FactorType& type_ref() const override { return CKDEType::get_ref(); }

    std::shared_ptr<arrow::DataType> data_type() const override {
        check_fitted();
        return m_training_type;
    }

    int num_instances() const {
        check_fitted();
        return N;
    }

    KDE& kde_joint() {
        check_fitted();
        return m_joint;
    }
    KDE& kde_marg() {
        check_fitted();
        return m_marg;
    }

    bool fitted() const override { return m_fitted; }

    std::shared_ptr<BandwidthSelector> bandwidth_type() const { return m_bselector; }

    void fit(const DataFrame& df) override;
    VectorXd logl(const DataFrame& df) const override;
    double slogl(const DataFrame& df) const override;

    Array_ptr sample(int n,
                     const DataFrame& evidence_values,
                     unsigned int seed = std::random_device{}()) const override;

    VectorXd cdf(const DataFrame& df) const;

    std::string ToString() const override;

    py::tuple __getstate__() const override;
    static CKDE __setstate__(py::tuple& t);
    static CKDE __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    void check_fitted() const {
        if (!fitted()) throw std::invalid_argument("CKDE factor not fitted.");
    }
    template <typename ArrowType>
    void _fit(const DataFrame& df);

    template <typename ArrowType>
    VectorXd _logl(const DataFrame& df) const;

    template <typename ArrowType>
    double _slogl(const DataFrame& df) const;

    template <typename ArrowType>
    Array_ptr _sample(int n, const DataFrame& evidence_values, unsigned int seed) const;

    template <typename ArrowType>
    Array_ptr _sample_multivariate(int n, const DataFrame& evidence_values, unsigned int seed) const;

    template <typename ArrowType>
    VectorXi _sample_indices_multivariate(Matrix<typename ArrowType::c_type, Dynamic, 1>& random_prob,
                                          const DataFrame& evidence_values,
                                          int n) const;

    template <typename ArrowType, typename KDEType>
    cl::Buffer _sample_indices_from_weights(cl::Buffer& random_prob, cl::Buffer& test_buffer, int n) const;

    template <typename ArrowType>
    VectorXd _cdf(const DataFrame& df) const;

    template <typename ArrowType>
    cl::Buffer _cdf_univariate(cl::Buffer& test_buffer, int m) const;

    template <typename ArrowType, typename KDEType>
    cl::Buffer _cdf_multivariate(cl::Buffer& variable_test_buffer, cl::Buffer& evidence_test_buffer, int m) const;

    template <typename ArrowType>
    py::tuple __getstate__() const;

    std::vector<std::string> m_variables;
    bool m_fitted;
    std::shared_ptr<BandwidthSelector> m_bselector;
    std::shared_ptr<arrow::DataType> m_training_type;
    size_t N;
    KDE m_joint;
    KDE m_marg;
};

template <typename ArrowType>
void CKDE::_fit(const DataFrame& df) {
    m_joint.fit(df);
    N = m_joint.num_instances();

    if (!this->evidence().empty()) {
        auto& joint_bandwidth = m_joint.bandwidth();
        auto d = m_variables.size();
        auto marg_bandwidth = joint_bandwidth.bottomRightCorner(d - 1, d - 1);

        cl::Buffer& training_buffer = m_joint.training_buffer();

        auto& opencl = OpenCLConfig::get();
        using CType = typename ArrowType::c_type;
        auto marg_buffer = opencl.copy_buffer<CType>(training_buffer, N, N * (d - 1));

        m_marg.fit<ArrowType>(marg_bandwidth, marg_buffer, m_joint.data_type(), N);
    }
}

template <typename ArrowType>
VectorXd CKDE::_logl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    auto logl_joint = m_joint.logl_buffer<ArrowType>(df);

    auto combined_bitmap = df.combined_bitmap(m_variables);
    auto m = df->num_rows();
    if (combined_bitmap) m = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

    auto& opencl = OpenCLConfig::get();
    if (!this->evidence().empty()) {
        cl::Buffer logl_marg;
        if (combined_bitmap)
            logl_marg = m_marg.logl_buffer<ArrowType>(df, combined_bitmap);
        else
            logl_marg = m_marg.logl_buffer<ArrowType>(df);

        auto& k_substract = opencl.kernel(OpenCL_kernel_traits<ArrowType>::substract_vectors);
        k_substract.setArg(0, logl_joint);
        k_substract.setArg(1, logl_marg);
        auto& queue = opencl.queue();
        RAISE_ENQUEUEKERNEL_ERROR(
            queue.enqueueNDRangeKernel(k_substract, cl::NullRange, cl::NDRange(m), cl::NullRange));
    }

    if (combined_bitmap) {
        VectorType read_data(m);
        auto bitmap_data = combined_bitmap->data();

        opencl.read_from_buffer(read_data.data(), logl_joint, m);

        VectorXd res(df->num_rows());

        for (int i = 0, k = 0; i < df->num_rows(); ++i) {
            if (util::bit_util::GetBit(bitmap_data, i)) {
                res(i) = static_cast<double>(read_data[k++]);
            } else {
                res(i) = util::nan<double>;
            }
        }

        return res;
    } else {
        VectorType read_data(df->num_rows());
        opencl.read_from_buffer(read_data.data(), logl_joint, df->num_rows());
        if constexpr (!std::is_same_v<CType, double>)
            return read_data.template cast<double>();
        else
            return read_data;
    }
}

template <typename ArrowType>
double CKDE::_slogl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;

    auto logl_joint = m_joint.logl_buffer<ArrowType>(df);

    auto combined_bitmap = df.combined_bitmap(m_variables);
    auto m = df->num_rows();
    if (combined_bitmap) m = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

    auto& opencl = OpenCLConfig::get();
    if (!this->evidence().empty()) {
        cl::Buffer logl_marg;
        if (combined_bitmap)
            logl_marg = m_marg.logl_buffer<ArrowType>(df, combined_bitmap);
        else
            logl_marg = m_marg.logl_buffer<ArrowType>(df);

        auto& k_substract = opencl.kernel(OpenCL_kernel_traits<ArrowType>::substract_vectors);
        k_substract.setArg(0, logl_joint);
        k_substract.setArg(1, logl_marg);
        auto& queue = opencl.queue();
        RAISE_ENQUEUEKERNEL_ERROR(
            queue.enqueueNDRangeKernel(k_substract, cl::NullRange, cl::NDRange(m), cl::NullRange));
    }

    auto buffer_sum = opencl.sum1d<ArrowType>(logl_joint, m);

    CType result = 0;
    opencl.read_from_buffer(&result, buffer_sum, 1);
    return static_cast<double>(result);
}

template <typename ArrowType>
Array_ptr CKDE::_sample(int n, const DataFrame& evidence_values, unsigned int seed) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;

    if (this->evidence().empty()) {
        arrow::NumericBuilder<ArrowType> builder;
        RAISE_STATUS_ERROR(builder.Resize(n));
        std::mt19937 rng{seed};
        std::uniform_int_distribution<> uniform(0, N - 1);

        std::normal_distribution<CType> normal(0, std::sqrt(m_joint.bandwidth()(0, 0)));
        VectorType training_data(N);
        const auto& training_buffer = m_joint.training_buffer();
        auto& opencl = OpenCLConfig::get();
        opencl.read_from_buffer(training_data.data(), training_buffer, N);

        for (auto i = 0; i < n; ++i) {
            auto index = uniform(rng);
            builder.UnsafeAppend(training_data(index) + normal(rng));
        }

        Array_ptr out;
        RAISE_STATUS_ERROR(builder.Finish(&out));
        return out;
    } else {
        return _sample_multivariate<ArrowType>(n, evidence_values, seed);
    }
}

template <typename ArrowType>
Array_ptr CKDE::_sample_multivariate(int n, const DataFrame& evidence_values, unsigned int seed) const {
    using CType = typename ArrowType::c_type;
    using ArrowArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using VectorType = Matrix<CType, Dynamic, 1>;
    using MatrixType = Matrix<CType, Dynamic, Dynamic>;

    const auto& e = this->evidence();

    if (!evidence_values.has_columns(e)) throw std::domain_error("Evidence values not present for sampling.");

    VectorType random_prob(n);
    std::mt19937 rng{seed};
    std::uniform_real_distribution<CType> uniform(0, 1);
    for (auto i = 0; i < n; ++i) {
        random_prob(i) = uniform(rng);
    }

    VectorXi sample_indices = _sample_indices_multivariate<ArrowType>(random_prob, evidence_values, n);

    const auto& bandwidth = m_joint.bandwidth();
    const auto& marg_bandwidth = m_marg.bandwidth();

    auto cholesky = marg_bandwidth.llt();
    auto matrixL = cholesky.matrixL();

    auto d = e.size();
    MatrixXd inverseL = MatrixXd::Identity(d, d);

    // Solves and saves the result in inverseL
    matrixL.solveInPlace(inverseL);
    auto R = inverseL * bandwidth.bottomLeftCorner(d, 1);
    auto cond_var = bandwidth(0, 0) - R.squaredNorm();
    auto transform = (R.transpose() * inverseL).transpose().template cast<CType>();

    MatrixType training_dataset(N, m_variables.size());
    auto& opencl = OpenCLConfig::get();
    opencl.read_from_buffer(training_dataset.data(), m_joint.training_buffer(), N * m_variables.size());

    MatrixType evidence_substract(n, e.size());
    for (size_t j = 0; j < e.size(); ++j) {
        auto evidence = evidence_values->GetColumnByName(e[j]);
        auto dwn_evidence = std::static_pointer_cast<ArrowArrayType>(evidence);
        auto raw_values = dwn_evidence->raw_values();
        for (auto i = 0; i < n; ++i) {
            evidence_substract(i, j) = raw_values[i] - training_dataset(sample_indices(i), j + 1);
        }
    }

    auto cond_mean = (evidence_substract * transform).eval();

    std::normal_distribution<CType> normal(0, std::sqrt(cond_var));
    arrow::NumericBuilder<ArrowType> builder;
    RAISE_STATUS_ERROR(builder.Resize(n));

    for (auto i = 0; i < n; ++i) {
        cond_mean(i) += training_dataset(sample_indices(i), 0) + normal(rng);
    }

    RAISE_STATUS_ERROR(builder.AppendValues(cond_mean.data(), n));

    Array_ptr out;
    RAISE_STATUS_ERROR(builder.Finish(&out));

    return out;
}

template <typename ArrowType>
VectorXi CKDE::_sample_indices_multivariate(Matrix<typename ArrowType::c_type, Dynamic, 1>& random_prob,
                                            const DataFrame& evidence_values,
                                            int n) const {
    using CType = typename ArrowType::c_type;
    using ArrowArray = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using MatrixType = Matrix<CType, Dynamic, Dynamic>;

    const auto& e = this->evidence();
    MatrixType test_matrix(n, e.size());

    for (size_t i = 0; i < e.size(); ++i) {
        auto evidence = evidence_values->GetColumnByName(e[i]);

        auto dwn_evidence = std::static_pointer_cast<ArrowArray>(evidence);
        auto raw_evidence = dwn_evidence->raw_values();

        std::memcpy(test_matrix.data() + i * n, raw_evidence, sizeof(CType) * n);
    }

    auto& opencl = OpenCLConfig::get();
    auto test_buffer = opencl.copy_to_buffer(test_matrix.data(), n * e.size());
    auto buff_random_prob = opencl.copy_to_buffer(random_prob.data(), n);

    cl::Buffer indices_buffer;
    if (e.size() == 1)
        indices_buffer = _sample_indices_from_weights<ArrowType, UnivariateKDE>(buff_random_prob, test_buffer, n);
    else
        indices_buffer = _sample_indices_from_weights<ArrowType, MultivariateKDE>(buff_random_prob, test_buffer, n);

    VectorXi res(n);
    opencl.read_from_buffer(res.data(), indices_buffer, n);
    return res;
}

template <typename ArrowType, typename KDEType>
cl::Buffer CKDE::_sample_indices_from_weights(cl::Buffer& random_prob, cl::Buffer& test_buffer, int n) const {
    using CType = typename ArrowType::c_type;

    auto& opencl = OpenCLConfig::get();
    auto res = opencl.new_buffer<int>(n);
    opencl.fill_buffer<int>(res, N - 1, n);

    auto [mat_logls, allocated_m] = opencl.allocate_temp_mat<ArrowType>(N, n);
    auto iterations = static_cast<int>(std::ceil(static_cast<double>(n) / static_cast<double>(allocated_m)));

    cl::Buffer tmp_mat_buffer;
    if constexpr (std::is_same_v<KDEType, MultivariateKDE>) {
        if (N > allocated_m)
            tmp_mat_buffer = opencl.new_buffer<CType>(N * this->evidence().size());
        else
            tmp_mat_buffer = opencl.new_buffer<CType>(allocated_m * this->evidence().size());
    }

    auto& k_exp = opencl.kernel(OpenCL_kernel_traits<ArrowType>::exp_elementwise);
    k_exp.setArg(0, mat_logls);

    auto& k_normalize_accum_sumexp = opencl.kernel(OpenCL_kernel_traits<ArrowType>::normalize_accum_sum_mat_cols);
    k_normalize_accum_sumexp.setArg(0, mat_logls);
    k_normalize_accum_sumexp.setArg(1, static_cast<unsigned int>(N));

    auto& k_find_random_indices = opencl.kernel(OpenCL_kernel_traits<ArrowType>::find_random_indices);
    k_find_random_indices.setArg(0, mat_logls);
    k_find_random_indices.setArg(1, static_cast<unsigned int>(N));
    k_find_random_indices.setArg(3, random_prob);
    k_find_random_indices.setArg(4, res);

    for (auto i = 0; i < (iterations - 1); ++i) {
        KDEType::template execute_logl_mat<ArrowType>(m_marg.training_buffer(),
                                                      N,
                                                      test_buffer,
                                                      n,
                                                      i * allocated_m,
                                                      allocated_m,
                                                      this->evidence().size(),
                                                      m_marg.cholesky_buffer(),
                                                      m_marg.lognorm_const(),
                                                      tmp_mat_buffer,
                                                      mat_logls);

        RAISE_ENQUEUEKERNEL_ERROR(
            opencl.queue().enqueueNDRangeKernel(k_exp, cl::NullRange, cl::NDRange(N * allocated_m), cl::NullRange));

        auto total_sum = opencl.accum_sum_cols<ArrowType>(mat_logls, N, allocated_m);

        k_normalize_accum_sumexp.setArg(2, total_sum);
        RAISE_ENQUEUEKERNEL_ERROR(opencl.queue().enqueueNDRangeKernel(
            k_normalize_accum_sumexp, cl::NullRange, cl::NDRange(N - 1, allocated_m), cl::NullRange));

        k_find_random_indices.setArg(2, static_cast<unsigned int>(i * allocated_m));
        RAISE_ENQUEUEKERNEL_ERROR(opencl.queue().enqueueNDRangeKernel(
            k_find_random_indices, cl::NullRange, cl::NDRange(N - 1, allocated_m), cl::NullRange));
    }
    auto offset = (iterations - 1) * allocated_m;
    auto remaining_m = n - offset;
    KDEType::template execute_logl_mat<ArrowType>(m_marg.training_buffer(),
                                                  N,
                                                  test_buffer,
                                                  n,
                                                  offset,
                                                  remaining_m,
                                                  this->evidence().size(),
                                                  m_marg.cholesky_buffer(),
                                                  m_marg.lognorm_const(),
                                                  tmp_mat_buffer,
                                                  mat_logls);

    RAISE_ENQUEUEKERNEL_ERROR(
        opencl.queue().enqueueNDRangeKernel(k_exp, cl::NullRange, cl::NDRange(N * remaining_m), cl::NullRange));

    auto total_sum = opencl.accum_sum_cols<ArrowType>(mat_logls, N, remaining_m);

    k_normalize_accum_sumexp.setArg(2, total_sum);
    RAISE_ENQUEUEKERNEL_ERROR(opencl.queue().enqueueNDRangeKernel(
        k_normalize_accum_sumexp, cl::NullRange, cl::NDRange(N - 1, remaining_m), cl::NullRange));

    k_find_random_indices.setArg(2, static_cast<unsigned int>(offset));
    RAISE_ENQUEUEKERNEL_ERROR(opencl.queue().enqueueNDRangeKernel(
        k_find_random_indices, cl::NullRange, cl::NDRange(N - 1, remaining_m), cl::NullRange));

    return res;
}

template <typename ArrowType>
VectorXd CKDE::_cdf(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    auto& opencl = OpenCLConfig::get();

    cl::Buffer res_buffer;
    auto test_matrix = df.to_eigen<false, ArrowType>(m_variables);
    auto m = test_matrix->rows();

    auto test_buffer = opencl.copy_to_buffer(test_matrix->data(), m);
    if (this->evidence().empty()) {
        res_buffer = _cdf_univariate<ArrowType>(test_buffer, m);
    } else {
        auto evidence_test_buffer = opencl.copy_to_buffer(test_matrix->data() + m, m * this->evidence().size());
        if (this->evidence().size() == 1) {
            res_buffer = _cdf_multivariate<ArrowType, UnivariateKDE>(test_buffer, evidence_test_buffer, m);
        } else {
            res_buffer = _cdf_multivariate<ArrowType, MultivariateKDE>(test_buffer, evidence_test_buffer, m);
        }
    }

    if (df.null_count(m_variables) == 0) {
        VectorType read_data(df->num_rows());
        opencl.read_from_buffer(read_data.data(), res_buffer, df->num_rows());
        if constexpr (!std::is_same_v<CType, double>)
            return read_data.template cast<double>();
        else
            return read_data;
    } else {
        auto valid = df.valid_rows(m_variables);
        VectorType read_data(valid);
        auto bitmap = df.combined_bitmap(m_variables);
        auto bitmap_data = bitmap->data();

        opencl.read_from_buffer(read_data.data(), res_buffer, valid);

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
cl::Buffer CKDE::_cdf_univariate(cl::Buffer& test_buffer, int m) const {
    using CType = typename ArrowType::c_type;
    auto& opencl = OpenCLConfig::get();
    auto res = opencl.new_buffer<CType>(m);

    auto [mu, allocated_m] = opencl.allocate_temp_mat<ArrowType>(N, m);
    auto iterations = std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m));

    auto& k_cdf = opencl.kernel(OpenCL_kernel_traits<ArrowType>::univariate_normal_cdf);
    k_cdf.setArg(0, m_joint.training_buffer());
    k_cdf.setArg(1, static_cast<unsigned int>(N));
    k_cdf.setArg(2, test_buffer);
    k_cdf.setArg(4, static_cast<CType>(1.0 / std::sqrt(m_joint.bandwidth()(0, 0))));
    k_cdf.setArg(5, static_cast<CType>(1.0 / N));
    k_cdf.setArg(6, mu);

    for (auto i = 0; i < (iterations - 1); ++i) {
        k_cdf.setArg(3, static_cast<unsigned int>(i * allocated_m));
        RAISE_ENQUEUEKERNEL_ERROR(
            opencl.queue().enqueueNDRangeKernel(k_cdf, cl::NullRange, cl::NDRange(N * allocated_m), cl::NullRange));
        opencl.sum_cols_offset<ArrowType>(mu, N, allocated_m, res, i * allocated_m);
    }
    auto offset = (iterations - 1) * allocated_m;
    auto remaining_m = m - offset;

    k_cdf.setArg(3, static_cast<unsigned int>(offset));
    RAISE_ENQUEUEKERNEL_ERROR(
        opencl.queue().enqueueNDRangeKernel(k_cdf, cl::NullRange, cl::NDRange(N * remaining_m), cl::NullRange));

    opencl.sum_cols_offset<ArrowType>(mu, N, remaining_m, res, offset);

    return res;
}

template <typename ArrowType, typename KDEType>
cl::Buffer CKDE::_cdf_multivariate(cl::Buffer& variable_test_buffer, cl::Buffer& evidence_test_buffer, int m) const {
    using CType = typename ArrowType::c_type;

    const auto& bandwidth = m_joint.bandwidth();
    const auto& marg_bandwidth = m_marg.bandwidth();

    auto cholesky = marg_bandwidth.llt();
    auto matrixL = cholesky.matrixL();

    auto d = this->evidence().size();
    MatrixXd inverseL = MatrixXd::Identity(d, d);

    // Solves and saves the result in inverseL
    matrixL.solveInPlace(inverseL);
    auto R = inverseL * bandwidth.bottomLeftCorner(d, 1);
    auto cond_var = bandwidth(0, 0) - R.squaredNorm();
    auto transform = (R.transpose() * inverseL).template cast<CType>().eval();

    auto& opencl = OpenCLConfig::get();
    auto transform_buffer = opencl.copy_to_buffer(transform.data(), this->evidence().size());

    auto res = opencl.new_buffer<CType>(m);

    auto [mu, allocated_m] = opencl.allocate_temp_mat<ArrowType>(N, m);
    auto W = opencl.new_buffer<CType>(N * allocated_m);
    auto sum_W = opencl.new_buffer<CType>(allocated_m);

    auto iterations = static_cast<int>(std::ceil(static_cast<double>(m) / static_cast<double>(allocated_m)));

    cl::Buffer tmp_mat_buffer;
    if constexpr (std::is_same_v<KDEType, MultivariateKDE>) {
        if (N > allocated_m)
            tmp_mat_buffer = opencl.new_buffer<CType>(N * this->evidence().size());
        else
            tmp_mat_buffer = opencl.new_buffer<CType>(allocated_m * this->evidence().size());
    }

    auto& k_exp = opencl.kernel(OpenCL_kernel_traits<ArrowType>::exp_elementwise);
    k_exp.setArg(0, W);

    auto& k_normal_cdf = opencl.kernel(OpenCL_kernel_traits<ArrowType>::normal_cdf);
    k_normal_cdf.setArg(0, mu);
    k_normal_cdf.setArg(1, static_cast<unsigned int>(N));
    k_normal_cdf.setArg(2, variable_test_buffer);
    k_normal_cdf.setArg(4, static_cast<CType>(1.0 / std::sqrt(cond_var)));

    auto& k_product = opencl.kernel(OpenCL_kernel_traits<ArrowType>::product_elementwise);
    k_product.setArg(0, mu);
    k_product.setArg(1, W);

    auto& k_divide = opencl.kernel(OpenCL_kernel_traits<ArrowType>::division_elementwise);
    k_divide.setArg(0, res);
    k_divide.setArg(2, sum_W);

    auto new_lognorm_marg = m_marg.lognorm_const() + std::log(N);

    for (auto i = 0; i < (iterations - 1); ++i) {
        // Computes Weigths
        KDEType::template execute_logl_mat<ArrowType>(m_marg.training_buffer(),
                                                      N,
                                                      evidence_test_buffer,
                                                      m,
                                                      i * allocated_m,
                                                      allocated_m,
                                                      this->evidence().size(),
                                                      m_marg.cholesky_buffer(),
                                                      new_lognorm_marg,
                                                      tmp_mat_buffer,
                                                      W);

        RAISE_ENQUEUEKERNEL_ERROR(
            opencl.queue().enqueueNDRangeKernel(k_exp, cl::NullRange, cl::NDRange(N * allocated_m), cl::NullRange));
        opencl.sum_cols_offset<ArrowType>(W, N, allocated_m, sum_W, 0);

        // Computes conditional mu.
        KDEType::template execute_conditional_means<ArrowType>(m_joint.training_buffer(),
                                                               m_marg.training_buffer(),
                                                               N,
                                                               evidence_test_buffer,
                                                               m,
                                                               i * allocated_m,
                                                               allocated_m,
                                                               this->evidence().size(),
                                                               transform_buffer,
                                                               tmp_mat_buffer,
                                                               mu);
        k_normal_cdf.setArg(3, static_cast<unsigned int>(i * allocated_m));
        RAISE_ENQUEUEKERNEL_ERROR(opencl.queue().enqueueNDRangeKernel(
            k_normal_cdf, cl::NullRange, cl::NDRange(N * allocated_m), cl::NullRange));
        RAISE_ENQUEUEKERNEL_ERROR(
            opencl.queue().enqueueNDRangeKernel(k_product, cl::NullRange, cl::NDRange(N * allocated_m), cl::NullRange));
        opencl.sum_cols_offset<ArrowType>(mu, N, allocated_m, res, i * allocated_m);
        k_divide.setArg(1, static_cast<unsigned int>(i * allocated_m));
        RAISE_ENQUEUEKERNEL_ERROR(
            opencl.queue().enqueueNDRangeKernel(k_divide, cl::NullRange, cl::NDRange(allocated_m), cl::NullRange));
    }
    auto offset = (iterations - 1) * allocated_m;
    auto remaining_m = m - offset;
    // Computes Weigths
    KDEType::template execute_logl_mat<ArrowType>(m_marg.training_buffer(),
                                                  N,
                                                  evidence_test_buffer,
                                                  m,
                                                  offset,
                                                  remaining_m,
                                                  this->evidence().size(),
                                                  m_marg.cholesky_buffer(),
                                                  new_lognorm_marg,
                                                  tmp_mat_buffer,
                                                  W);

    RAISE_ENQUEUEKERNEL_ERROR(
        opencl.queue().enqueueNDRangeKernel(k_exp, cl::NullRange, cl::NDRange(N * remaining_m), cl::NullRange));
    opencl.sum_cols_offset<ArrowType>(W, N, remaining_m, sum_W, 0);

    // Computes conditional mu.
    KDEType::template execute_conditional_means<ArrowType>(m_joint.training_buffer(),
                                                           m_marg.training_buffer(),
                                                           N,
                                                           evidence_test_buffer,
                                                           m,
                                                           offset,
                                                           remaining_m,
                                                           this->evidence().size(),
                                                           transform_buffer,
                                                           tmp_mat_buffer,
                                                           mu);

    k_normal_cdf.setArg(3, static_cast<unsigned int>(offset));
    RAISE_ENQUEUEKERNEL_ERROR(
        opencl.queue().enqueueNDRangeKernel(k_normal_cdf, cl::NullRange, cl::NDRange(N * remaining_m), cl::NullRange));
    RAISE_ENQUEUEKERNEL_ERROR(
        opencl.queue().enqueueNDRangeKernel(k_product, cl::NullRange, cl::NDRange(N * remaining_m), cl::NullRange));
    opencl.sum_cols_offset<ArrowType>(mu, N, remaining_m, res, offset);
    k_divide.setArg(1, static_cast<unsigned int>(offset));
    RAISE_ENQUEUEKERNEL_ERROR(
        opencl.queue().enqueueNDRangeKernel(k_divide, cl::NullRange, cl::NDRange(remaining_m), cl::NullRange));

    return res;
}

template <typename ArrowType>
py::tuple CKDE::__getstate__() const {
    py::tuple joint_tuple;
    if (m_fitted) {
        joint_tuple = m_joint.__getstate__();
    }

    return py::make_tuple(this->variable(), this->evidence(), m_fitted, joint_tuple);
}

// Fix const name: https://stackoverflow.com/a/15862594
struct HCKDEName {
    inline constexpr static auto* str = "HCKDE";
};

struct CKDEFitter {
    static bool fit(const std::shared_ptr<Factor>& factor, const DataFrame& df) {
        try {
            factor->fit(df);
            return true;
        } catch (util::singular_covariance_data& e) {
            return false;
        } catch (py::error_already_set& e) {
            auto t = py::module_::import("pybnesian").attr("SingularCovarianceData");
            if (e.matches(t)) {
                return false;
            } else {
                throw;
            }
        }
    }
};

using HCKDE = DiscreteAdaptator<CKDE, CKDEFitter, HCKDEName>;

}  // namespace factors::continuous

#endif  // PYBNESIAN_FACTORS_CONTINUOUS_CKDE_HPP