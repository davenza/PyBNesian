#include <Python.h>
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>
#include <util/bit_util.hpp>
#include <util/math_constants.hpp>
#include <Eigen/Dense>
#include <learning/parameters/mle_LinearGaussianCPD.hpp>


namespace py = pybind11;

using arrow::Type;
using Eigen::Matrix, Eigen::Array, Eigen::Dynamic, Eigen::Map, Eigen::MatrixBase;

using dataset::DataFrame;

using learning::parameters::MLE;
using util::pi;

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
    m_variance(variance)
//    TODO: Error checking: Length of vectors
    {
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
        
        ArrayVecType means = ArrayVecType::Constant(df->num_rows(), beta[0]);

        int idx = 1;
        for (auto it = evidence.begin(); it != evidence.end(); ++it, ++idx) {
            auto ev_array = df.to_eigen<false, ArrowType, false>(*it);
            means += static_cast<CType>(beta[idx]) * ev_array->array();
        }

        auto var_array = df.to_eigen<false, ArrowType, false>(var);

        double inv_std = 1 / std::sqrt(variance);
        ArrayVecType logl = -0.5 * (inv_std * (var_array->array() - means)).square();

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

    VectorXd LinearGaussianCPD::logl(const DataFrame& df) const {
        switch(df.col(m_variable)->type_id()) {
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
        switch(df.col(m_variable)->type_id()) {
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
                throw py::value_error("Wrong data type to compute logl. (double) or (float) data is expected.");
        }
    }

    Array_ptr LinearGaussianCPD::sample(int n, 
                                        std::unordered_map<std::string, Array_ptr>& parent_values, 
                                        long unsigned int seed) const {
        arrow::NumericBuilder<arrow::DoubleType> builder;
        builder.Resize(n);

        std::mt19937 rng{seed};
        std::normal_distribution<> normal(m_beta(0), std::sqrt(m_variance));

        for(auto i = 0; i < n; ++i) {
            builder.UnsafeAppend(normal(rng));
        }

        std::shared_ptr<arrow::DoubleArray> out;
        auto status = builder.Finish(&out);
        if (!status.ok()) {
            throw std::runtime_error("New array could not be created. Error status: " + status.ToString());
        }

        if (!m_evidence.empty()) {
            auto out_values = reinterpret_cast<double*>(out->values()->mutable_data());
            for (auto j = 0; j < m_evidence.size(); ++j) {
                auto found = parent_values.find(m_evidence[j]);
                auto evidence = found->second;

                switch (evidence->type_id()) {
                    case Type::DOUBLE: {
                        auto dwn_evidence = std::static_pointer_cast<arrow::DoubleArray>(evidence);
                        auto raw_evidence = dwn_evidence->raw_values();

                        for (auto i = 0; i < n; ++i) {
                            out_values[i] += m_beta(j+1)*raw_evidence[i];
                        }
                    }
                    case Type::FLOAT: {
                        auto dwn_evidence = std::static_pointer_cast<arrow::FloatArray>(evidence);
                        auto raw_evidence = dwn_evidence->raw_values();

                        for (auto i = 0; i < n; ++i) {
                            out_values[i] += m_beta(j+1)*raw_evidence[i];
                        }
                    }
                    default:
                        throw std::invalid_argument("Wrong data type for LinearGaussianCPD parent data.");
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
}
