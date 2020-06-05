#include <iostream>
#include <Python.h>
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>
#include <util/bit_util.hpp>
#include <util/math_constants.hpp>
#include <Eigen/Dense>
#include <learning/parameter/mle.hpp>

#include <arrow/compute/kernels/cast.h>
#include <arrow/compute/context.h>
#include <chrono>
#include <iomanip>

namespace py = pybind11;
namespace pyarrow = arrow::py;

using arrow::Type;
using Eigen::Matrix, Eigen::Array, Eigen::Dynamic, Eigen::Map, Eigen::MatrixBase;

using dataset::DataFrame;

using learning::parameter::MLE;
using util::pi;

typedef std::shared_ptr<arrow::Array> Array_ptr;


namespace factors::continuous {


    LinearGaussianCPD::LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence) :
    m_variable(variable),
    m_evidence(evidence)
    {
        m_beta = VectorXd(evidence.size() + 1);
    };

    LinearGaussianCPD::LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence,
                                         const std::vector<double> beta, double variance) :
    m_variable(variable),
    m_evidence(evidence),
    m_variance(variance)
//    TODO: Error checking: Length of vectors
    {
        m_beta = VectorXd(beta.size());
        auto m_ptr = m_beta.data();
        auto vec_ptr = beta.data();
        std::memcpy(m_ptr, vec_ptr, sizeof(double) * beta.size());
    };

    void LinearGaussianCPD::fit(py::handle dataset) {
        auto rb = dataset::to_record_batch(dataset);
        auto df = DataFrame(rb);
        fit(df);
    }

    void LinearGaussianCPD::fit(const DataFrame& df) {
        MLE<LinearGaussianCPD> mle;

        auto params = mle.estimate(df, m_variable, m_evidence);
        
        m_beta = params.beta;
        m_variance = params.variance;
    }


    template<typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> 
    logpdf_impl(const DataFrame& df, const VectorXd& beta, double variance, 
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

        double inv_variance = 1 / variance;
        ArrayVecType logl = (inv_variance * (var_array->array() - means)).square();

        logl += -0.5*std::log(variance) - 0.5*std::log(2*pi<CType>);

        return logl.matrix();
    }

    template<typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1> 
    logpdf_impl_null(const DataFrame& df, const VectorXd& beta, double variance, 
                            const std::string& var, const std::vector<std::string>& evidence) {
        using CType = typename ArrowType::c_type;

        auto logl = logpdf_impl<ArrowType>(df, beta, variance, var, evidence);
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
    double slogpdf_impl(const DataFrame& df, const VectorXd& beta, double variance, 
                            const std::string& var, const std::vector<std::string>& evidence) {
        return logpdf_impl<ArrowType>(df, beta, variance, var, evidence).sum();
    }

    template<typename ArrowType>
    double slogpdf_impl_null(const DataFrame& df, const VectorXd& beta, double variance, 
                            const std::string& var, const std::vector<std::string>& evidence) {
        auto logl = logpdf_impl<ArrowType>(df, beta, variance, var, evidence);

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

    VectorXd LinearGaussianCPD::logpdf(py::handle dataset) const {
        auto rb = dataset::to_record_batch(dataset);
        auto df = DataFrame(rb);

        return logpdf(df);
    }

    VectorXd LinearGaussianCPD::logpdf(const DataFrame& df) const {
        switch(df.col(m_variable)->type_id()) {
            case Type::DOUBLE: {
                if(df.null_count(m_variable, m_evidence) == 0)
                    return logpdf_impl<arrow::DoubleType>(df, m_beta, m_variance, m_variable, m_evidence);
                else
                    return logpdf_impl_null<arrow::DoubleType>(df, m_beta, m_variance, m_variable, m_evidence);
            }
            case Type::FLOAT: {
                if(df.null_count(m_variable, m_evidence) == 0) {
                    auto t = logpdf_impl<arrow::FloatType>(df, m_beta, m_variance, m_variable, m_evidence);
                    return t.template cast<double>();
                }
                else {
                    auto t = logpdf_impl_null<arrow::FloatType>(df, m_beta, m_variance, m_variable, m_evidence);
                    return t.template cast<double>();
                }
            }
            default:
                throw py::value_error("Wrong data type to compute logpdf. (double) or (float) data is expected.");
        }
    }


    double LinearGaussianCPD::slogpdf(py::handle dataset) const {
        auto rb = dataset::to_record_batch(dataset);
        auto df = DataFrame(rb);

        return slogpdf(df);
    }

    double LinearGaussianCPD::slogpdf(const DataFrame& df) const {
        switch(df.col(m_variable)->type_id()) {
            case Type::DOUBLE: {
                if(df.null_count(m_variable, m_evidence) == 0)
                    return slogpdf_impl<arrow::DoubleType>(df, m_beta, m_variance, m_variable, m_evidence);
                else
                    return slogpdf_impl_null<arrow::DoubleType>(df, m_beta, m_variance, m_variable, m_evidence);
            }
            case Type::FLOAT: {
                if(df.null_count(m_variable, m_evidence) == 0)
                    return slogpdf_impl<arrow::FloatType>(df, m_beta, m_variance, m_variable, m_evidence);
                else
                    return slogpdf_impl_null<arrow::FloatType>(df, m_beta, m_variance, m_variable, m_evidence);
            }
            default:
                throw py::value_error("Wrong data type to compute logpdf. (double) or (float) data is expected.");
        }
    }
}