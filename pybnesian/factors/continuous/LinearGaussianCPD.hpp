#ifndef PYBNESIAN_FACTORS_CONTINUOUS_LINEARGAUSSIANCPD_HPP
#define PYBNESIAN_FACTORS_CONTINUOUS_LINEARGAUSSIANCPD_HPP

#include <random>
#include <factors/factors.hpp>
#include <factors/discrete/DiscreteAdaptator.hpp>
#include <dataset/dataset.hpp>

using dataset::DataFrame;
using Eigen::VectorXd;
using factors::Factor;
using factors::discrete::DiscreteAdaptator;

namespace py = pybind11;

namespace factors::continuous {

class LinearGaussianCPDType : public FactorType {
public:
    LinearGaussianCPDType(const LinearGaussianCPDType&) = delete;
    void operator=(const LinearGaussianCPDType&) = delete;

    static std::shared_ptr<LinearGaussianCPDType> get() {
        static std::shared_ptr<LinearGaussianCPDType> singleton =
            std::shared_ptr<LinearGaussianCPDType>(new LinearGaussianCPDType);
        return singleton;
    }

    static LinearGaussianCPDType& get_ref() {
        static LinearGaussianCPDType& ref = *LinearGaussianCPDType::get();
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

    std::string ToString() const override { return "LinearGaussianFactor"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<LinearGaussianCPDType> __setstate__(py::tuple&) { return LinearGaussianCPDType::get(); }

private:
    LinearGaussianCPDType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

struct LinearGaussianCPD_Params {
    VectorXd beta;
    double variance;
};

class LinearGaussianCPD : public Factor {
public:
    using ParamsClass = LinearGaussianCPD_Params;
    using FactorTypeClass = LinearGaussianCPDType;

    LinearGaussianCPD() = default;
    LinearGaussianCPD(std::string variable, std::vector<std::string> evidence);
    LinearGaussianCPD(std::string variable, std::vector<std::string> evidence, VectorXd beta, double variance);

    std::shared_ptr<FactorType> type() const override { return LinearGaussianCPDType::get(); }

    FactorType& type_ref() const override { return LinearGaussianCPDType::get_ref(); }

    std::shared_ptr<arrow::DataType> data_type() const override { return arrow::float64(); }

    bool fitted() const override { return m_fitted; }
    void fit(const DataFrame& df) override;
    VectorXd logl(const DataFrame& df) const override;
    double slogl(const DataFrame& df) const override;
    VectorXd cdf(const DataFrame& df) const;

    std::string ToString() const override;

    const VectorXd& beta() const { return m_beta; }
    void set_beta(const VectorXd& new_beta) {
        if (static_cast<size_t>(new_beta.rows()) != (evidence().size() + 1))
            throw std::invalid_argument(
                "Wrong number of elements for the beta vector: " + std::to_string(new_beta.rows()) +
                ". Expected size: " + std::to_string((evidence().size() + 1)));
        m_beta = new_beta;

        if (m_variance > 0) m_fitted = true;
    }

    double variance() const { return m_variance; }
    void set_variance(double v) {
        if (v <= 0) {
            throw std::invalid_argument("Variance must be a positive value.");
        }

        m_variance = v;
        if (m_beta.rows() == static_cast<int>(evidence().size() + 1)) m_fitted = true;
    }

    Array_ptr sample(int n,
                     const DataFrame& evidence_values,
                     unsigned int seed = std::random_device{}()) const override;

    py::tuple __getstate__() const override;
    static LinearGaussianCPD __setstate__(py::tuple& t);
    static LinearGaussianCPD __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    void check_fitted() const {
        if (!fitted()) throw std::invalid_argument("LinearGaussianCPD factor not fitted.");
    }

    bool m_fitted;
    VectorXd m_beta;
    double m_variance;
};

// Fix const name: https://stackoverflow.com/a/15862594
struct CLinearGaussianCPDName {
    inline constexpr static auto* str = "CLinearGaussianCPD";
};

struct LinearGaussianFitter {
    static bool fit(const std::shared_ptr<Factor>& factor, const DataFrame& df) {
        factor->fit(df);
        auto dwn = std::static_pointer_cast<LinearGaussianCPD>(factor);

        if (dwn->variance() < util::machine_tol || std::isinf(dwn->variance())) {
            return false;
        }

        return true;
    }
};

using CLinearGaussianCPD = DiscreteAdaptator<LinearGaussianCPD, LinearGaussianFitter, CLinearGaussianCPDName>;

}  // namespace factors::continuous

#endif  // PYBNESIAN_FACTORS_CONTINUOUS_LINEARGAUSSIANCPD_HPP
