#include <Python.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <arrow/api.h>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <models/BayesianNetwork.hpp>
#include <dataset/dataset.hpp>
#include <util/bit_util.hpp>
#include <util/math_constants.hpp>
#include <util/arrow_macros.hpp>
#include <Eigen/Dense>
#include <learning/parameters/mle_base.hpp>
#include <boost/math/distributions/normal.hpp>

namespace py = pybind11;

using arrow::Type;
using Eigen::Matrix, Eigen::Dynamic, Eigen::Array;

using boost::math::normal_distribution;
using dataset::DataFrame;
using factors::discrete::DiscreteFactorType;
using learning::parameters::MLE;
using models::BayesianNetworkBase, models::ConditionalBayesianNetworkBase;
using util::pi, util::one_div_root_two;

typedef std::shared_ptr<arrow::Array> Array_ptr;

namespace factors::continuous {

std::shared_ptr<Factor> LinearGaussianCPDType::new_factor(const BayesianNetworkBase& m,
                                                          const std::string& variable,
                                                          const std::vector<std::string>& evidence,
                                                          py::args args,
                                                          py::kwargs kwargs) const {
    for (const auto& e : evidence) {
        if (m.node_type(e) == DiscreteFactorType::get()) {
            return generic_new_factor<CLinearGaussianCPD>(variable, evidence, args, kwargs);
        }
    }

    return generic_new_factor<LinearGaussianCPD>(variable, evidence, args, kwargs);
}

std::shared_ptr<Factor> LinearGaussianCPDType::new_factor(const ConditionalBayesianNetworkBase& m,
                                                          const std::string& variable,
                                                          const std::vector<std::string>& evidence,
                                                          py::args args,
                                                          py::kwargs kwargs) const {
    for (const auto& e : evidence) {
        if (m.node_type(e) == DiscreteFactorType::get()) {
            return generic_new_factor<CLinearGaussianCPD>(variable, evidence, args, kwargs);
        }
    }

    return generic_new_factor<LinearGaussianCPD>(variable, evidence, args, kwargs);
}

LinearGaussianCPD::LinearGaussianCPD(std::string variable, std::vector<std::string> evidence)
    : Factor(variable, evidence), m_fitted(false), m_beta(), m_variance(-1){};

LinearGaussianCPD::LinearGaussianCPD(std::string variable,
                                     std::vector<std::string> evidence,
                                     VectorXd beta,
                                     double variance)
    : Factor(variable, evidence), m_fitted(true), m_variance(variance) {
    if (static_cast<size_t>(beta.rows()) != (evidence.size() + 1)) {
        throw std::invalid_argument(
            "Wrong number of beta parameters. Beta vector size: " + std::to_string(beta.size()) +
            ". Expected beta vector size: " + std::to_string(evidence.size() + 1) + ".");
    }

    if (variance <= 0) {
        throw std::invalid_argument("Variance must be a positive value.");
    }

    m_beta = beta;
};

void LinearGaussianCPD::fit(const DataFrame& df) {
    MLE<LinearGaussianCPD> mle;

    auto params = mle.estimate(df, this->variable(), this->evidence());

    m_beta = params.beta;
    m_variance = params.variance;
    m_fitted = true;
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> logl_impl(const DataFrame& df,
                                                         const VectorXd& beta,
                                                         double variance,
                                                         const std::string& var,
                                                         const std::vector<std::string>& evidence) {
    using CType = typename ArrowType::c_type;
    using ArrayVecType = Array<CType, Dynamic, 1>;

    double inv_std = 1 / std::sqrt(variance);

    ArrayVecType logl;
    auto var_array = df.to_eigen<false, ArrowType, false>(var);
    if (evidence.empty()) {
        logl = -0.5 * (inv_std * (var_array->array() - beta(0))).square();
    } else {
        ArrayVecType means = ArrayVecType::Constant(df->num_rows(), beta(0));
        int idx = 1;
        for (auto it = evidence.begin(); it != evidence.end(); ++it, ++idx) {
            auto ev_array = df.to_eigen<false, ArrowType, false>(*it);
            means += static_cast<CType>(beta[idx]) * ev_array->array();
        }

        logl = -0.5 * (inv_std * (var_array->array() - means)).square();
    }

    logl += -0.5 * std::log(variance) - 0.5 * std::log(2 * pi<CType>);
    return logl.matrix();
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> logl_impl_null(const DataFrame& df,
                                                              const VectorXd& beta,
                                                              double variance,
                                                              const std::string& var,
                                                              const std::vector<std::string>& evidence) {
    using CType = typename ArrowType::c_type;

    auto logl = logl_impl<ArrowType>(df, beta, variance, var, evidence);
    auto combined_bitmap = df.combined_bitmap(var, evidence);
    auto bitmap_data = combined_bitmap->data();
    auto logl_ptr = logl.data();

    for (int i = 0; i < df->num_rows(); ++i) {
        if (!util::bit_util::GetBit(bitmap_data, i)) logl_ptr[i] = util::nan<CType>;
    }

    return logl;
}

template <typename ArrowType>
double slogl_impl(const DataFrame& df,
                  const VectorXd& beta,
                  double variance,
                  const std::string& var,
                  const std::vector<std::string>& evidence) {
    return logl_impl<ArrowType>(df, beta, variance, var, evidence).sum();
}

template <typename ArrowType>
double slogl_impl_null(const DataFrame& df,
                       const VectorXd& beta,
                       double variance,
                       const std::string& var,
                       const std::vector<std::string>& evidence) {
    auto logl = logl_impl<ArrowType>(df, beta, variance, var, evidence);

    auto combined_bitmap = df.combined_bitmap(var, evidence);
    auto bitmap_data = combined_bitmap->data();
    auto logl_ptr = logl.data();

    double accum = 0;
    for (int i = 0; i < df->num_rows(); ++i) {
        if (util::bit_util::GetBit(bitmap_data, i)) accum += logl_ptr[i];
    }

    return accum;
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> cdf_impl(const DataFrame& df,
                                                        const VectorXd& beta,
                                                        double variance,
                                                        const std::string& var,
                                                        const std::vector<std::string>& evidence) {
    using CType = typename ArrowType::c_type;
    using ArrayVecType = Array<CType, Dynamic, 1>;

    CType inv_std = static_cast<CType>(1. / std::sqrt(variance));
    auto var_ptr = df.to_eigen<false, ArrowType, false>(var);
    auto var_array = *var_ptr;

    ArrayVecType t(df->num_rows());
    if (evidence.empty()) {
        for (int64_t i = 0, end = df->num_rows(); i < end; ++i) {
            t(i) = std::erfc((beta(0) - var_array(i)) * inv_std * one_div_root_two<CType>);
        }
    } else {
        ArrayVecType means = ArrayVecType::Constant(df->num_rows(), beta[0]);

        int idx = 1;
        for (auto it = evidence.begin(); it != evidence.end(); ++it, ++idx) {
            auto ev_array = df.to_eigen<false, ArrowType, false>(*it);
            means += static_cast<CType>(beta[idx]) * ev_array->array();
        }

        for (int64_t i = 0, end = df->num_rows(); i < end; ++i) {
            t(i) = std::erfc((means(i) - var_array(i)) * inv_std * one_div_root_two<CType>);
        }
    }

    return (0.5 * t).matrix();
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> cdf_impl_null(const DataFrame& df,
                                                             const VectorXd& beta,
                                                             double variance,
                                                             const std::string& var,
                                                             const std::vector<std::string>& evidence) {
    using CType = typename ArrowType::c_type;
    using ArrayVecType = Array<CType, Dynamic, 1>;

    CType inv_std = static_cast<CType>(1. / std::sqrt(variance));
    auto combined_bitmap = df.combined_bitmap(var, evidence);
    auto bitmap_data = combined_bitmap->data();

    auto var_ptr = df.to_eigen<false, ArrowType, false>(var);
    auto var_array = *var_ptr;

    ArrayVecType t(df->num_rows());
    if (evidence.empty()) {
        for (int64_t i = 0, end = df->num_rows(); i < end; ++i) {
            if (util::bit_util::GetBit(bitmap_data, i))
                t(i) = std::erfc((beta(0) - var_array(i)) * inv_std * one_div_root_two<CType>);
            else
                t(i) = util::nan<CType>;
        }
    } else {
        auto non_null = util::bit_util::non_null_count(combined_bitmap, df->num_rows());
        ArrayVecType means = ArrayVecType::Constant(non_null, beta[0]);

        int idx = 1;
        for (auto it = evidence.begin(); it != evidence.end(); ++it, ++idx) {
            auto ev_array = df.to_eigen<false, ArrowType>(combined_bitmap, *it);
            means += static_cast<CType>(beta[idx]) * ev_array->array();
        }

        for (int64_t i = 0, k = 0, end = df->num_rows(); i < end; ++i) {
            if (util::bit_util::GetBit(bitmap_data, i))
                t(i) = std::erfc((means(k++) - var_array(i)) * inv_std * one_div_root_two<CType>);
            else
                t(i) = util::nan<CType>;
        }
    }

    return (0.5 * t).matrix();
}

VectorXd LinearGaussianCPD::logl(const DataFrame& df) const {
    check_fitted();
    switch (df.same_type(this->variable(), this->evidence())->id()) {
        case Type::DOUBLE: {
            if (df.null_count(this->variable(), this->evidence()) == 0)
                return logl_impl<arrow::DoubleType>(df, m_beta, m_variance, this->variable(), this->evidence());
            else
                return logl_impl_null<arrow::DoubleType>(df, m_beta, m_variance, this->variable(), this->evidence());
        }
        case Type::FLOAT: {
            if (df.null_count(this->variable(), this->evidence()) == 0) {
                auto t = logl_impl<arrow::FloatType>(df, m_beta, m_variance, this->variable(), this->evidence());
                return t.template cast<double>();
            } else {
                auto t = logl_impl_null<arrow::FloatType>(df, m_beta, m_variance, this->variable(), this->evidence());
                return t.template cast<double>();
            }
        }
        default:
            throw std::invalid_argument("Wrong data type to compute logl. [double] or [float] data is expected.");
    }
}

double LinearGaussianCPD::slogl(const DataFrame& df) const {
    check_fitted();
    switch (df.same_type(this->variable(), this->evidence())->id()) {
        case Type::DOUBLE: {
            if (df.null_count(this->variable(), this->evidence()) == 0)
                return slogl_impl<arrow::DoubleType>(df, m_beta, m_variance, this->variable(), this->evidence());
            else
                return slogl_impl_null<arrow::DoubleType>(df, m_beta, m_variance, this->variable(), this->evidence());
        }
        case Type::FLOAT: {
            if (df.null_count(this->variable(), this->evidence()) == 0)
                return slogl_impl<arrow::FloatType>(df, m_beta, m_variance, this->variable(), this->evidence());
            else
                return slogl_impl_null<arrow::FloatType>(df, m_beta, m_variance, this->variable(), this->evidence());
        }
        default:
            throw std::invalid_argument("Wrong data type to compute slogl. [double] or [float] data is expected.");
    }
}

VectorXd LinearGaussianCPD::cdf(const DataFrame& df) const {
    check_fitted();
    switch (df.same_type(this->variable(), this->evidence())->id()) {
        case Type::DOUBLE: {
            if (df.null_count(this->variable(), this->evidence()) == 0)
                return cdf_impl<arrow::DoubleType>(df, m_beta, m_variance, this->variable(), this->evidence());
            else
                return cdf_impl_null<arrow::DoubleType>(df, m_beta, m_variance, this->variable(), this->evidence());
        }
        case Type::FLOAT: {
            if (df.null_count(this->variable(), this->evidence()) == 0) {
                auto t = cdf_impl<arrow::FloatType>(df, m_beta, m_variance, this->variable(), this->evidence());
                return t.template cast<double>();
            } else {
                auto t = cdf_impl_null<arrow::FloatType>(df, m_beta, m_variance, this->variable(), this->evidence());
                return t.template cast<double>();
            }
        }
        default:
            throw std::invalid_argument("Wrong data type to compute cdf. [double] or [float] data is expected.");
    }
}

Array_ptr LinearGaussianCPD::sample(int n, const DataFrame& evidence_values, unsigned int seed) const {
    if (n < 0) {
        throw std::invalid_argument("n should be a non-negative number");
    }

    check_fitted();
    arrow::NumericBuilder<arrow::DoubleType> builder;
    RAISE_STATUS_ERROR(builder.Resize(n));

    std::mt19937 rng{seed};
    std::normal_distribution<> normal(m_beta(0), std::sqrt(m_variance));

    for (auto i = 0; i < n; ++i) {
        builder.UnsafeAppend(normal(rng));
    }

    std::shared_ptr<arrow::DoubleArray> out;
    RAISE_STATUS_ERROR(builder.Finish(&out));

    if (!this->evidence().empty()) {
        if (!evidence_values.has_columns(this->evidence()))
            throw std::domain_error("Evidence values not present for sampling.");

        auto out_values = reinterpret_cast<double*>(out->values()->mutable_data());
        const auto& e = this->evidence();
        for (size_t j = 0; j < e.size(); ++j) {
            auto evidence = evidence_values->GetColumnByName(e[j]);

            switch (evidence->type_id()) {
                case Type::DOUBLE: {
                    auto dwn_evidence = std::static_pointer_cast<arrow::DoubleArray>(evidence);
                    auto raw_evidence = dwn_evidence->raw_values();

                    for (auto i = 0; i < n; ++i) {
                        out_values[i] += m_beta(j + 1) * raw_evidence[i];
                    }
                    break;
                }
                case Type::FLOAT: {
                    auto dwn_evidence = std::static_pointer_cast<arrow::FloatArray>(evidence);
                    auto raw_evidence = dwn_evidence->raw_values();

                    for (auto i = 0; i < n; ++i) {
                        out_values[i] += m_beta(j + 1) * raw_evidence[i];
                    }
                    break;
                }
                default: {
                    throw std::invalid_argument("Wrong data type \"" + evidence->type()->ToString() +
                                                "\" for LinearGaussianCPD parent data.");
                }
            }
        }
    }

    return out;
}

std::string LinearGaussianCPD::ToString() const {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(3);
    if (!this->evidence().empty()) {
        const auto& e = this->evidence();
        stream << "[LinearGaussianCPD] P(" << this->variable() << " | " << e[0];
        for (size_t i = 1; i < e.size(); ++i) {
            stream << ", " << e[i];
        }

        if (m_fitted) {
            stream << ") = N(" << m_beta(0);
            for (size_t i = 0; i < e.size(); ++i) {
                stream << " + " << m_beta(i + 1) << "*" << e[i];
            }
            stream << ", " << m_variance << ")";
        } else {
            stream << ") not fitted";
        }
    } else {
        if (m_fitted)
            stream << "[LinearGaussianCPD] P(" << this->variable() << ") = N(" << m_beta(0) << ", " << m_variance
                   << ")";
        else
            stream << "[LinearGaussianCPD] P(" << this->variable() << ") not fitted";
    }
    return stream.str();
}

py::tuple LinearGaussianCPD::__getstate__() const {
    return py::make_tuple(this->variable(), this->evidence(), m_fitted, m_beta, m_variance);
}

LinearGaussianCPD LinearGaussianCPD::__setstate__(py::tuple& t) {
    if (t.size() != 5) throw std::runtime_error("Not valid LinearGaussianCPD.");

    LinearGaussianCPD cpd(t[0].cast<std::string>(), t[1].cast<std::vector<std::string>>());

    cpd.m_fitted = t[2].cast<bool>();
    cpd.m_beta = t[3].cast<VectorXd>();
    cpd.m_variance = t[4].cast<double>();

    return cpd;
}

}  // namespace factors::continuous
