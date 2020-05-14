#include <iostream>
#include <Python.h>
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>
#include <util/bit_util.hpp>
#include <Eigen/Dense>
#include <learning/parameter/mle.hpp>

#include <arrow/compute/kernels/cast.h>
#include <arrow/compute/context.h>
#include <chrono>
#include <iomanip>

namespace py = pybind11;
namespace pyarrow = arrow::py;

using arrow::Type;
using Eigen::Matrix, Eigen::Dynamic, Eigen::Map, Eigen::MatrixBase;

using dataset::DataFrame;

using learning::parameter::MLE;

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

        MLE<LinearGaussianCPD> mle;

        auto params = mle.estimate(df, m_variable, m_evidence);
        
        m_beta = params.beta;
        m_variance = params.variance;
    }
}