#ifndef PYBNESIAN_KDE_NORMALREFERENCERULE_HPP
#define PYBNESIAN_KDE_NORMALREFERENCERULE_HPP

#include <kde/BandwidthSelector.hpp>

namespace kde {

class NormalReferenceRule : public BandwidthSelector {
public:
    VectorXd diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override {
        switch (df.same_type(variables)->id()) {
            case Type::DOUBLE:
                return diag_bandwidth<arrow::DoubleType>(df, variables);
            case Type::FLOAT:
                return diag_bandwidth<arrow::FloatType>(df, variables);
            default:
                throw py::value_error("Wrong data type to fit bandwidth. [double] or [float] data is expected.");
        }
    }

    MatrixXd bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override {
        switch (df.same_type(variables)->id()) {
            case Type::DOUBLE:
                return bandwidth<arrow::DoubleType>(df, variables);
            case Type::FLOAT:
                return bandwidth<arrow::FloatType>(df, variables);
            default:
                throw py::value_error("Wrong data type to fit bandwidth. [double] or [float] data is expected.");
        }
    }

    std::string ToString() const override { return "NormalReferenceRule"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<NormalReferenceRule> __setstate__(py::tuple&) {
        return std::make_shared<NormalReferenceRule>();
    }

private:
    template <typename ArrowType>
    VectorXd diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
        using CType = typename ArrowType::c_type;

        auto cov_ptr = df.cov<ArrowType>(variables);
        auto& cov = *cov_ptr;
        auto diag = cov.diagonal();
        auto delta = (cov.array().colwise() * diag.cwiseInverse().array()).matrix();
        auto delta_inv = delta.inverse();

        auto N = static_cast<CType>(df.valid_rows(variables));
        auto d = static_cast<CType>(variables.size());

        auto delta_inv_trace = delta_inv.trace();

        // Estimate bandwidth using Equation (3.4) of Chacon and Duong (2018)
        auto k = 4 * d * std::sqrt(delta.determinant()) /
                 (2 * (delta_inv * delta_inv).trace() + delta_inv_trace * delta_inv_trace);

        if constexpr (std::is_same_v<ArrowType, arrow::DoubleType>) {
            return std::pow(k / N, 2. / (d + 4.)) * diag;
        } else {
            return (std::pow(k / N, 2. / (d + 4.)) * diag).template cast<double>();
        }
    }

    template <typename ArrowType>
    MatrixXd bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
        using CType = typename ArrowType::c_type;

        auto cov = df.cov<ArrowType>(variables);
        auto N = static_cast<CType>(df.valid_rows(variables));
        auto d = static_cast<CType>(variables.size());

        auto k = std::pow(4. / (N * (d + 2.)), 2. / (d + 4));

        if constexpr (std::is_same_v<ArrowType, arrow::DoubleType>) {
            return k * (*cov);
        } else {
            return k * cov->template cast<double>();
        }
    }
};

}  // namespace kde

#endif  // PYBNESIAN_KDE_NORMALREFERENCERULE_HPP