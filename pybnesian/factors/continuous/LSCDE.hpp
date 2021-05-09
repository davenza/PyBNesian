#ifndef PYBNESIAN_FACTORS_CONTINUOUS_LSCDE_HPP
#define PYBNESIAN_FACTORS_CONTINUOUS_LSCDE_HPP

#include <iostream>
#include <algorithm>
#include <arrow/compute/api.h>
#include <dataset/crossvalidation_adaptator.hpp>
#include <factors/factors.hpp>
#include <util/basic_eigen_ops.hpp>
#include <util/math_constants.hpp>

using dataset::CrossValidation;
using Eigen::Ref, Eigen::PartialPivLU, Eigen::LLT;

namespace factors::continuous {

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    if (v.empty()) {
        out << "[]";
    } else {
        out << "[" << v[0];

        for (auto it = ++v.begin(); it != v.end(); ++it) {
            out << ", " << *it;
        }

        out << "]";
    }

    return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::unordered_set<T>& s) {
    if (s.empty()) {
        out << "{}";
    } else {
        std::vector<T> v(s.begin(), s.end());
        out << "{" << v[0];

        for (auto it = ++v.begin(); it != v.end(); ++it) {
            out << ", " << *it;
        }

        out << "}";
    }

    return out;
}

class LSCDEType : public FactorType {
public:
    LSCDEType(const LSCDEType&) = delete;
    void operator=(const LSCDEType&) = delete;

    static std::shared_ptr<LSCDEType> get() {
        static std::shared_ptr<LSCDEType> singleton = std::shared_ptr<LSCDEType>(new LSCDEType);
        return singleton;
    }

    static LSCDEType& get_ref() {
        static LSCDEType& ref = *LSCDEType::get();
        return ref;
    }

    std::shared_ptr<Factor> new_factor(const BayesianNetworkBase&,
                                       const std::string&,
                                       const std::vector<std::string>&) const override;
    std::shared_ptr<Factor> new_factor(const ConditionalBayesianNetworkBase&,
                                       const std::string&,
                                       const std::vector<std::string>&) const override;

    std::shared_ptr<FactorType> opposite_semiparametric() const override;

    std::string ToString() const override { return "LinearGaussianFactor"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<LSCDEType> __setstate__(py::tuple&) { return LSCDEType::get(); }

private:
    LSCDEType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

class LSCDE : public Factor {
public:
    LSCDE() = default;
    LSCDE(std::string variable, std::vector<std::string> evidence, int b = 100)
        : Factor(variable, evidence), m_fitted(false), m_b(b), m_alpha(), m_sigma(), m_u(), m_v() {}

    std::shared_ptr<FactorType> type() const override { return LSCDEType::get(); }
    FactorType& type_ref() const override { return LSCDEType::get_ref(); }

    std::shared_ptr<arrow::DataType> data_type() const override { return arrow::float64(); }

    bool fitted() const override { return m_fitted; }
    void fit(const DataFrame& df) override;
    VectorXd logl(const DataFrame& df) const override;
    double slogl(const DataFrame& df) const override;

    std::string ToString() const override { return "LSCDE"; }

    Array_ptr sample(int n,
                     const DataFrame& evidence_values,
                     unsigned int seed = std::random_device{}()) const override {
        arrow::NumericBuilder<arrow::Int64Type> builder;
        builder.Append(1);

        Array_ptr out;
        RAISE_STATUS_ERROR(builder.Finish(&out));
    }

    py::tuple __getstate__() const override { return py::make_tuple(); }

private:
    std::pair<DataFrame, DataFrame> get_uv(const DataFrame& df);
    template <typename ArrowType>
    void train_cv(const DataFrame& df);

    template <typename ArrowType>
    void _fit(const DataFrame& df);

    template <typename ArrowType>
    VectorXd _logl(const DataFrame& df) const;

    bool m_fitted;
    int m_b;
    VectorXd m_alpha;
    double m_sigma;
    DataFrame m_u;
    DataFrame m_v;
};

// Get (x - x_i)**2 for a given variable
template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, Dynamic> log_kern_dist(const DataFrame& basis_df,
                                                                   const DataFrame& df,
                                                                   const std::string& variable) {
    using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, Dynamic>;
    auto up = basis_df.to_eigen<false, ArrowType, false>(variable);
    const auto& u = *up;
    auto xp = df.to_eigen<false, ArrowType, false>(variable);
    const auto& x = *xp;

    MatrixType res(u.rows(), x.rows());

    for (int i = 0, i_end = u.rows(); i < i_end; ++i) {
        for (int j = 0, j_end = x.rows(); j < j_end; ++j) {
            res(i, j) = u(i) - x(j);
        }
    }

    return res.array().square().matrix();
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, Dynamic> log_kern_dist(const DataFrame& basis_df,
                                                                   const DataFrame& df,
                                                                   const std::vector<std::string>& variables) {
    if (variables.empty()) {
        throw std::invalid_argument("This code do not implement marginal probability");
    } else {
        auto xu = log_kern_dist<ArrowType>(basis_df, df, variables[0]);

        for (int i = 1, end = variables.size(); i < end; ++i) {
            xu.noalias() += log_kern_dist<ArrowType>(basis_df, df, variables[i]);
        }

        return xu;
    }
}

template <typename ArrowType>
void LSCDE::train_cv(const DataFrame& df) {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    using MatrixType = Matrix<CType, Dynamic, Dynamic>;
    VectorType sigma_list = util::logspace<CType>(-1.5, 1.5, 20);
    VectorType lambda_list = util::logspace<CType>(-3, 1, 5);

    int K = 5;

    CrossValidation cv(df, K, 0);

    auto [u, v] = get_uv(df);
    auto b = u->num_rows();

    auto logxu = log_kern_dist<ArrowType>(u, df, evidence());
    auto logyv = log_kern_dist<ArrowType>(v, df, variable());
    auto logzw = logxu + logyv;
    auto logvv = log_kern_dist<ArrowType>(v, v, variable());
    std::vector<MatrixType> cv_phibar(K);
    std::vector<int> num_instances(K);

    MatrixType cv_score(sigma_list.rows(), lambda_list.rows());
    for (int i = 0, i_end = sigma_list.rows(); i < i_end; ++i) {
        auto mult_sigma = 1. / (2 * sigma_list(i) * sigma_list(i));
        auto mult_sigma2 = 1. / (4 * sigma_list(i) * sigma_list(i));

        auto phi_xu = (-mult_sigma * logxu).array().exp().matrix().eval();
        auto phi_zw = (-mult_sigma * logzw).array().exp().matrix().eval();
        auto phi_vv = (-mult_sigma2 * logvv).array().exp().matrix();

        int k = 0;

        auto constant = std::sqrt(util::pi<CType>) * sigma_list(i);
        for (auto [_, test_indices] : cv.indices()) {
            auto xu_cv = util::filter_cols(phi_xu, test_indices);

            cv_phibar[k] = constant * phi_vv.cwiseProduct(xu_cv * xu_cv.transpose());
            num_instances[k] = test_indices.size();
            ++k;
        }

        auto sum_phibar = cv_phibar[0];
        for (int k = 1; k < K; ++k) {
            sum_phibar += cv_phibar[k];
        }

        VectorType sum_h = phi_zw.rowwise().sum();

        for (int j = 0, j_end = lambda_list.rows(); j < j_end; ++j) {
            int k = 0;
            CType score = 0;
            for (auto [train_indices, test_indices] : cv.indices()) {
                auto tmp_phibar = ((sum_phibar - cv_phibar[k]) / (df->num_rows() - num_instances[k])).eval();

                tmp_phibar += VectorType::Constant(b, lambda_list(j)).asDiagonal();

                auto test_xu = util::filter_cols(phi_xu, test_indices);
                auto test_zw = util::filter_cols(phi_zw, test_indices);

                VectorType h = (sum_h - test_zw.rowwise().sum()) * (1. / train_indices.size());

                LLT<Ref<MatrixType>> llt(tmp_phibar);
                auto alpha = llt.solve(h).array().max(0).matrix();

                auto normalization = (util::root_two<CType> * constant * alpha.transpose() * test_xu)
                                         .array()
                                         .max(util::machine_tol<CType>)
                                         .matrix();

                auto ph = (alpha.transpose() * test_zw).cwiseProduct(normalization.cwiseInverse());
                score -= (ph.array() + util::machine_tol<CType>).log().mean();

                ++k;
            }

            cv_score(i, j) = score;
        }
    }

    int best_sigma = 0;
    int best_lambda = 0;
    cv_score.minCoeff(&best_sigma, &best_lambda);

    m_sigma = sigma_list(best_sigma);
    auto lambda = lambda_list(best_lambda);

    auto mult_sigma = 1. / (2 * m_sigma * m_sigma);
    auto mult_sigma2 = 1. / (4 * m_sigma * m_sigma);

    auto phi_xu = (-mult_sigma * logxu).array().exp().matrix().eval();
    auto phi_vv = (-mult_sigma2 * logvv).array().exp().matrix();

    auto constant = std::sqrt(util::pi<CType>) * m_sigma;
    MatrixType phibar = constant * phi_vv.cwiseProduct(phi_xu * phi_xu.transpose());
    phibar += VectorType::Constant(b, lambda).asDiagonal();

    VectorType h = (-mult_sigma * (logzw)).array().exp().rowwise().mean().transpose().matrix();

    LLT<Ref<MatrixType>> llt(phibar);
    VectorType alpha = llt.solve(h).array().max(0).matrix();

    std::vector<int64_t> positive_alpha;

    for (auto i = 0; i < alpha.rows(); ++i) {
        if (alpha(i) > 0) positive_alpha.push_back(i);
    }

    if (positive_alpha.size() == b) {
        if constexpr (std::is_same_v<CType, double>) {
            m_alpha = alpha;
        } else {
            m_alpha = alpha.template cast<double>();
        }
        m_u = u;
        m_v = v;
    } else {
        if (positive_alpha.empty()) throw std::invalid_argument("Valid least squares solution could not be found.");

        m_alpha = VectorXd(positive_alpha.size());
        for (int i = 0; i < positive_alpha.size(); ++i) {
            m_alpha(i) = alpha(positive_alpha[i]);
        }

        arrow::AdaptiveIntBuilder builder;
        RAISE_STATUS_ERROR(builder.AppendValues(positive_alpha.data(), positive_alpha.size()));
        Array_ptr positive_ind;
        RAISE_STATUS_ERROR(builder.Finish(&positive_ind));

        auto fu = arrow::compute::Take(u.record_batch(), positive_ind, arrow::compute::TakeOptions::NoBoundsCheck());
        auto fv = arrow::compute::Take(v.record_batch(), positive_ind, arrow::compute::TakeOptions::NoBoundsCheck());

        m_u = DataFrame(std::move(fu).ValueOrDie().record_batch());
        m_v = DataFrame(std::move(fv).ValueOrDie().record_batch());
    }
}

template <typename ArrowType>
VectorXd LSCDE::_logl(const DataFrame& df) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<CType, Dynamic, 1>;
    using MatrixType = Matrix<CType, Dynamic, Dynamic>;

    auto mult_sigma = 1 / (2 * m_sigma * m_sigma);

    auto logxu = log_kern_dist<ArrowType>(m_u, df, evidence());
    auto logyv = log_kern_dist<ArrowType>(m_v, df, variable());

    auto phi_xu = (-mult_sigma * logxu).array().exp().matrix().eval();
    auto phi_zw = (-mult_sigma * (logxu + logyv)).array().exp().matrix().eval();

    auto constant = std::sqrt(2 * util::pi<CType>) * m_sigma;

    if constexpr (std::is_same_v<CType, double>) {
        auto normalization = (constant * m_alpha.transpose() * phi_xu).array().max(util::machine_tol<CType>).matrix();
        return (m_alpha.transpose() * phi_zw).cwiseProduct(normalization.cwiseInverse()).array().log();
    } else {
        auto falpha = m_alpha.template cast<CType>();
        auto normalization = (constant * falpha.transpose() * phi_xu).array().max(util::machine_tol<CType>).matrix();
        return (falpha.transpose() * phi_zw)
            .cwiseProduct(normalization.cwiseInverse())
            .array()
            .log()
            .template cast<double>();
    }

    /********
     * This is an logSumExp implementation. It is much slower than the normal domain implementation.
     *
     ********/

    // auto mult_sigma = -1. / (2 * m_sigma * m_sigma);

    // auto logxu = log_kern_dist<ArrowType>(m_u, df, evidence());
    // logxu *= mult_sigma;

    // auto logyv = log_kern_dist<ArrowType>(m_v, df, variable()).eval();
    // logyv *= mult_sigma;

    // if constexpr (std::is_same_v<CType, double>) {
    //     auto dll = logxu.colwise() + m_logalpha;
    //     auto max_dll = dll.colwise().maxCoeff();
    //     auto normalization = 0.5 * std::log(2 * util::pi<CType>) + std::log(m_sigma) + max_dll.array() +
    //                          (dll.rowwise() - max_dll).array().exp().colwise().sum().log();

    //     auto nll = dll + logyv;
    //     auto max_nll = dll.colwise().maxCoeff();
    //     auto numerator = max_nll.array() + (nll.rowwise() - max_nll).array().exp().colwise().sum().log();

    //     return (numerator - normalization).array().matrix();
    // } else {
    //     auto logalpha = m_logalpha.template cast<CType>();

    //     auto dll = logxu.colwise() + logalpha;
    //     auto max_dll = dll.colwise().maxCoeff();
    //     auto normalization = 0.5 * std::log(2 * util::pi<CType>) + std::log(m_sigma) + max_dll.array() +
    //                          (dll.rowwise() - max_dll).array().exp().colwise().sum().log();

    //     auto nll = dll + logyv;
    //     auto max_nll = dll.colwise().maxCoeff();
    //     auto numerator = max_nll.array() + (nll.rowwise() - max_nll).array().exp().colwise().sum().log();

    //     return (numerator - normalization).array().matrix().template cast<double>();
    // }
}

}  // namespace factors::continuous

#endif  // PYBNESIAN_FACTORS_CONTINUOUS_LSCDE_HPP