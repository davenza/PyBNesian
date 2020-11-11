#include <Python.h>
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>
#include <util/bit_util.hpp>
#include <util/math_constants.hpp>
#include <util/arrow_macros.hpp>
#include <Eigen/Dense>
#include <learning/parameters/mle_LinearGaussianCPD.hpp>
#include <boost/math/distributions/normal.hpp>

namespace py = pybind11;

using arrow::Type;
using Eigen::Matrix, Eigen::Dynamic, Eigen::Array;

using dataset::DataFrame;

using learning::parameters::MLE;
using util::pi, util::one_div_root_two;
using boost::math::normal_distribution;

typedef std::shared_ptr<arrow::Array> Array_ptr;

namespace factors::continuous {


    LinearGaussianCPD::LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence) :
                                                                        m_variable(variable),
                                                                        m_evidence(evidence),
                                                                        m_fitted(false)
    {
        m_beta = VectorXd(evidence.size() + 1);
    };

    LinearGaussianCPD::LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence,
                                         const std::vector<double> beta, double variance) :
    m_variable(variable),
    m_evidence(evidence),
    m_fitted(true),
    m_variance(variance) {
        if (beta.size() != (evidence.size() + 1)) {
            throw py::value_error("Wrong number of beta parameters. Beta vector size: " + std::to_string(beta.size()) + 
                                    ". Expected beta vector size: " + std::to_string(evidence.size() + 1) + ".");
        }
        
        if (variance <= 0) {
            throw py::value_error("Variance must be a positive value.");
        }

        m_beta = VectorXd(beta.size());
        auto m_ptr = m_beta.data();
        auto vec_ptr = beta.data();
        std::memcpy(m_ptr, vec_ptr, sizeof(double) * beta.size());
    };

    void LinearGaussianCPD::fit(const DataFrame& df) {
        MLE<LinearGaussianCPD> mle;

        auto params = mle.estimate(df, m_variable, m_evidence.begin(), m_evidence.end());
        
        m_beta = params.beta;
        m_variance = params.variance;
        m_fitted = true;
    }


    template<typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> 
    logl_impl(const DataFrame& df, const VectorXd& beta, double variance, 
                            const std::string& var, const std::vector<std::string>& evidence) {
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
        
        logl += -0.5*std::log(variance) - 0.5*std::log(2*pi<CType>);
        return logl.matrix();
    }

    template<typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> 
    logl_impl_null(const DataFrame& df, const VectorXd& beta, double variance, 
                            const std::string& var, const std::vector<std::string>& evidence) {
        using CType = typename ArrowType::c_type;

        auto logl = logl_impl<ArrowType>(df, beta, variance, var, evidence);
        auto combined_bitmap = df.combined_bitmap(var, evidence);
        auto bitmap_data = combined_bitmap->data();
        auto logl_ptr = logl.data();

        for (int i = 0; i < df->num_rows(); ++i) {
            if (!arrow::BitUtil::GetBit(bitmap_data, i))
                logl_ptr[i] = util::nan<CType>;
        }

        return logl;
    }

    template<typename ArrowType>
    double slogl_impl(const DataFrame& df, const VectorXd& beta, double variance, 
                            const std::string& var, const std::vector<std::string>& evidence) {
        return logl_impl<ArrowType>(df, beta, variance, var, evidence).sum();
    }

    template<typename ArrowType>
    double slogl_impl_null(const DataFrame& df, const VectorXd& beta, double variance, 
                            const std::string& var, const std::vector<std::string>& evidence) {
        auto logl = logl_impl<ArrowType>(df, beta, variance, var, evidence);

        auto combined_bitmap = df.combined_bitmap(var, evidence);
        auto bitmap_data = combined_bitmap->data();
        auto logl_ptr = logl.data();

        double accum = 0;
        for (int i = 0; i < df->num_rows(); ++i) {
            if (arrow::BitUtil::GetBit(bitmap_data, i))
                accum += logl_ptr[i];
        }

        return accum;
    }

    template<typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> 
    cdf_impl(const DataFrame& df, const VectorXd& beta, double variance, 
                            const std::string& var, const std::vector<std::string>& evidence) {
        using CType = typename ArrowType::c_type;
        using ArrayVecType = Array<CType, Dynamic, 1>;

        CType inv_std = static_cast<CType>(1. / std::sqrt(variance));
        auto var_ptr = df.to_eigen<false, ArrowType, false>(var);
        auto var_array = *var_ptr;

        ArrayVecType t(df->num_rows());
        if (evidence.empty()) {
            for (int64_t i = 0, end = df->num_rows(); i < end; ++i) {
                t(i) = std::erfc((beta(0) - var_array(i))*inv_std*one_div_root_two<CType>);
            }
        } else {
            ArrayVecType means = ArrayVecType::Constant(df->num_rows(), beta[0]);

            int idx = 1;
            for (auto it = evidence.begin(); it != evidence.end(); ++it, ++idx) {
                auto ev_array = df.to_eigen<false, ArrowType, false>(*it);
                means += static_cast<CType>(beta[idx]) * ev_array->array();
            }

            for (int64_t i = 0, end = df->num_rows(); i < end; ++i) {
                t(i) = std::erfc((means(i) - var_array(i))*inv_std*one_div_root_two<CType>);
            }
        }

        return (0.5*t).matrix();
    }

    template<typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> 
    cdf_impl_null(const DataFrame& df, const VectorXd& beta, double variance, 
                            const std::string& var, const std::vector<std::string>& evidence) {
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
                if (arrow::BitUtil::GetBit(bitmap_data, i))
                    t(i) = std::erfc((beta(0) - var_array(i))*inv_std*one_div_root_two<CType>);
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
                if (arrow::BitUtil::GetBit(bitmap_data, i))
                    t(i) = std::erfc((means(k++) - var_array(i))*inv_std*one_div_root_two<CType>);
                else
                    t(i) = util::nan<CType>;
            }
        }

        return (0.5*t).matrix();
    }

    VectorXd LinearGaussianCPD::logl(const DataFrame& df) const {
        switch(df.same_type(m_variable, m_evidence)) {
            case Type::DOUBLE: {
                if(df.null_count(m_variable, m_evidence) == 0)
                    return logl_impl<arrow::DoubleType>(df, m_beta, m_variance, m_variable, m_evidence);
                else
                    return logl_impl_null<arrow::DoubleType>(df, m_beta, m_variance, m_variable, m_evidence);
            }
            case Type::FLOAT: {
                if(df.null_count(m_variable, m_evidence) == 0) {
                    auto t = logl_impl<arrow::FloatType>(df, m_beta, m_variance, m_variable, m_evidence);
                    return t.template cast<double>();
                }
                else {
                    auto t = logl_impl_null<arrow::FloatType>(df, m_beta, m_variance, m_variable, m_evidence);
                    return t.template cast<double>();
                }
            }
            default:
                throw py::value_error("Wrong data type to compute logl. (double) or (float) data is expected.");
        }
    }

    double LinearGaussianCPD::slogl(const DataFrame& df) const {
        switch(df.same_type(m_variable, m_evidence)) {
            case Type::DOUBLE: {
                if(df.null_count(m_variable, m_evidence) == 0)
                    return slogl_impl<arrow::DoubleType>(df, m_beta, m_variance, m_variable, m_evidence);
                else
                    return slogl_impl_null<arrow::DoubleType>(df, m_beta, m_variance, m_variable, m_evidence);
            }
            case Type::FLOAT: {
                if(df.null_count(m_variable, m_evidence) == 0)
                    return slogl_impl<arrow::FloatType>(df, m_beta, m_variance, m_variable, m_evidence);
                else
                    return slogl_impl_null<arrow::FloatType>(df, m_beta, m_variance, m_variable, m_evidence);
            }
            default:
                throw py::value_error("Wrong data type to compute slogl. (double) or (float) data is expected.");
        }
    }

    VectorXd LinearGaussianCPD::cdf(const DataFrame& df) const {
        switch(df.same_type(m_variable, m_evidence)) {
            case Type::DOUBLE: {
                if(df.null_count(m_variable, m_evidence) == 0)
                    return cdf_impl<arrow::DoubleType>(df, m_beta, m_variance, m_variable, m_evidence);
                else
                    return cdf_impl_null<arrow::DoubleType>(df, m_beta, m_variance, m_variable, m_evidence);
            }
            case Type::FLOAT: {
                if(df.null_count(m_variable, m_evidence) == 0) {
                    auto t = cdf_impl<arrow::FloatType>(df, m_beta, m_variance, m_variable, m_evidence);
                    return t.template cast<double>();
                }
                else {
                    auto t = cdf_impl_null<arrow::FloatType>(df, m_beta, m_variance, m_variable, m_evidence);
                    return t.template cast<double>();
                }
            }
            default:
                throw py::value_error("Wrong data type to compute cdf. (double) or (float) data is expected.");
        }
    }

    Array_ptr LinearGaussianCPD::sample(int n, 
                                        const DataFrame& evidence_values, 
                                        long unsigned int seed) const {
        arrow::NumericBuilder<arrow::DoubleType> builder;
        RAISE_STATUS_ERROR(builder.Resize(n));


        std::mt19937 rng{seed};
        std::normal_distribution<> normal(m_beta(0), std::sqrt(m_variance));

        for(auto i = 0; i < n; ++i) {
            builder.UnsafeAppend(normal(rng));
        }

        std::shared_ptr<arrow::DoubleArray> out;
        RAISE_STATUS_ERROR(builder.Finish(&out));

        if (!m_evidence.empty()) {
            try {
                evidence_values.has_columns(m_evidence);
            } catch (const std::domain_error& ex) {
                throw std::domain_error(std::string("Evidence values not present for sampling:\n") + ex.what());
            }

            auto out_values = reinterpret_cast<double*>(out->values()->mutable_data());
            for (size_t j = 0; j < m_evidence.size(); ++j) {
                auto evidence = evidence_values->GetColumnByName(m_evidence[j]);

                switch (evidence->type_id()) {
                    case Type::DOUBLE: {
                        auto dwn_evidence = std::static_pointer_cast<arrow::DoubleArray>(evidence);
                        auto raw_evidence = dwn_evidence->raw_values();

                        for (auto i = 0; i < n; ++i) {
                            out_values[i] += m_beta(j+1)*raw_evidence[i];
                        }
                        break;
                    }
                    case Type::FLOAT: {
                        auto dwn_evidence = std::static_pointer_cast<arrow::FloatArray>(evidence);
                        auto raw_evidence = dwn_evidence->raw_values();

                        for (auto i = 0; i < n; ++i) {
                            out_values[i] += m_beta(j+1)*raw_evidence[i];
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
        stream << std::setprecision(3);
        if (!m_evidence.empty()) {
            stream << "[LinearGaussianCPD] P(" << m_variable << " | " << m_evidence[0];
            for (size_t i = 1; i < m_evidence.size(); ++i) {
                stream << ", " << m_evidence[i];
            }

            if (m_fitted) {
                stream << ") = N(" << m_beta(0);
                for (size_t i = 1; i < m_evidence.size(); ++i) {
                    stream << " + " << m_beta(i) << "*" << m_evidence[i];
                }
                stream << ", " << m_variance << ")";
            } else {
                stream << ") not fitted";
            }
        } else {
            if (m_fitted)
                stream << "[LinearGaussianCPD] P(" << m_variable << ") = N(" 
                                           << m_beta(0) << ", " << m_variance << ")";
            else
                stream << "[LinearGaussianCPD] P(" << m_variable << ") not fitted";
        }
        return stream.str();
    }

    py::tuple LinearGaussianCPD::__getstate__() const {
        return py::make_tuple(m_variable, m_evidence, m_fitted, m_beta, m_variance);
    }

    LinearGaussianCPD LinearGaussianCPD::__setstate__(py::tuple& t) {
        if (t.size() != 5)
            throw std::runtime_error("Not valid DirectedGraph.");

        LinearGaussianCPD cpd(t[0].cast<std::string>(),
                              t[1].cast<std::vector<std::string>>());

        cpd.m_fitted = t[2].cast<bool>();
        cpd.m_beta = t[3].cast<VectorXd>();
        cpd.m_variance = t[4].cast<double>();

        return cpd;
    }
}
