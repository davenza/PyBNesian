#ifndef PYBNESIAN_KDE_NORMALREFERENCERULE_HPP
#define PYBNESIAN_KDE_NORMALREFERENCERULE_HPP

#include <kde/BandwidthSelector.hpp>
#include <util/basic_eigen_ops.hpp>
#include <util/exceptions.hpp>

namespace kde {

class NormalReferenceRule : public BandwidthSelector {
public:
    /**
     * @brief Public function for calculating the diagonal bandwidth matrix using the Normal Reference Rule given the
     * data and variables.
     *
     * @param df Dataframe.
     * @param variables Variables.
     * @return VectorXd Diagonal bandwidth vector.
     */
    VectorXd diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override {
        if (variables.empty()) return VectorXd(0);

        size_t valid_rows = df.valid_rows(variables);
        if (valid_rows <= variables.size()) {  // If the number of (valid) rows is less than the number of variables
            std::stringstream ss;
            ss << "NormalReferenceRule::diag_bandwidth -> Diagonal bandwidth matrix of "
               << std::to_string(variables.size()) << " variables [" << variables[0];
            for (size_t i = 1; i < variables.size(); ++i) {
                ss << ", " << variables[i];
            }
            ss << "] cannot be estimated with " << std::to_string(valid_rows) << " instances";

            throw util::singular_covariance_data(ss.str());
        }

        switch (df.same_type(variables)->id()) {
            case Type::DOUBLE:
                return diag_bandwidth<arrow::DoubleType>(df, variables);
            case Type::FLOAT:
                return diag_bandwidth<arrow::FloatType>(df, variables);
            default:
                throw std::invalid_argument(
                    "NormalReferenceRule::diag_bandwidth -> Wrong data type to fit bandwidth. [double] or [float] data "
                    "is expected.");
        }
    }
    /**
     * @brief Public function for calculating the bandwidth matrix using the Normal Reference Rule given the data and
     * variables.
     *
     * @param df Data
     * @param variables Variables.
     * @return MatrixXd Bandwidth matrix.
     */
    MatrixXd bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override {
        if (variables.empty()) return MatrixXd(0, 0);

        auto valid_rows = df.valid_rows(variables);
        if (static_cast<size_t>(valid_rows) <=
            variables.size()) {  // If the number of (valid) rows is less than the number of variables
            std::stringstream ss;
            ss << "NormalReferenceRule::bandwidth -> Bandwidth matrix of " << std::to_string(variables.size())
               << " variables [" << variables[0];
            for (size_t i = 1; i < variables.size(); ++i) {
                ss << ", " << variables[i];
            }
            ss << "] cannot be estimated with " << std::to_string(valid_rows) << " instances";

            throw util::singular_covariance_data(ss.str());
        }

        switch (df.same_type(variables)->id()) {
            // Here the bandwidth is calculated using the function defined later in the private section.
            case Type::DOUBLE:
                return bandwidth<arrow::DoubleType>(df, variables);
            case Type::FLOAT:
                return bandwidth<arrow::FloatType>(df, variables);
            default:
                throw std::invalid_argument(
                    "NormalReferenceRule::bandwidth -> Wrong data type to fit bandwidth. [double] or [float] data is "
                    "expected.");
        }
    }

    std::string ToString() const override { return "NormalReferenceRule"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<NormalReferenceRule> __setstate__(py::tuple&) {
        return std::make_shared<NormalReferenceRule>();
    }

private:
    /**
     * @brief Private function to calculate the diagonal bandwidth matrix using the Normal Reference Rule given the data
     * and variables. If the covariance matrix is not positive definite, an exception is thrown.
     *
     * @tparam ArrowType Arrow Data type.
     * @param df Dataframe.
     * @param variables Variables.
     * @return VectorXd Diagonal bandwidth vector.
     */
    template <typename ArrowType>
    VectorXd diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
        using CType = typename ArrowType::c_type;

        auto cov_ptr = df.cov<ArrowType>(variables);
        auto& cov = *cov_ptr;
        // NOTE: UNNECESSARY CHECK
        // if (!util::is_psd(cov)) {
        //     std::stringstream ss;
        //     ss << "NormalReferenceRule::diag_bandwidth -> Covariance matrix for variables [" << variables[0];
        //     for (size_t i = 1; i < variables.size(); ++i) {
        //         ss << ", " << variables[i];
        //     }
        //     ss << "] is not positive-definite.";
        //     throw util::singular_covariance_data(ss.str());
        // }
        // The covariance diagonal is used to calculate the bandwidth
        auto diag = cov.diagonal();
        auto delta = (cov.array().colwise() * diag.cwiseInverse().array()).matrix();  // diag(cov)^ (-1) * cov
        auto delta_inv = delta.inverse();

        auto N = static_cast<CType>(df.valid_rows(variables));
        auto d = static_cast<CType>(variables.size());

        auto delta_inv_trace = delta_inv.trace();

        // NOTE: Estimate bandwidth using Equation (3.4) of Chacon and Duong (2018)
        // [4*d*sqrt(det(delta))] /
        // / [(2*trace(delta^(-1)*delta^(-1)) + trace(delta^(-1))^2) * N]
        auto k = 4 * d * std::sqrt(delta.determinant()) /
                 ((2 * (delta_inv * delta_inv).trace() + delta_inv_trace * delta_inv_trace) * N);
        auto k2 = std::pow(k, 2. / (d + 4.));
        if constexpr (std::is_same_v<ArrowType, arrow::DoubleType>) {
            return k2 * diag;
        } else {
            return (k2 * diag).template cast<double>();
        }
    }
    /**
     * @brief Private function to calculate the bandwidth matrix using the Normal Reference Rule given the data and
     * variables. If the covariance matrix is not positive definite, an exception is thrown.
     *
     * @tparam ArrowType Arrow Data type.
     * @param df Dataframe.
     * @param variables Variables.
     * @return MatrixXd Bandwidth matrix.
     */
    template <typename ArrowType>
    MatrixXd bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
        using CType = typename ArrowType::c_type;

        auto cov_ptr = df.cov<ArrowType>(variables);
        auto& cov = *cov_ptr;

        // NOTE: UNNECESSARY CHECK
        // if (!util::is_psd(cov)) {
        //     std::stringstream ss;
        //     ss << "Covariance matrix for variables [" << variables[0];
        //     for (size_t i = 1; i < variables.size(); ++i) {
        //         ss << ", " << variables[i];
        //     }
        //     ss << "] is not positive-definite.";
        //     throw util::singular_covariance_data(ss.str());
        // }

        // NOTE: We get the diagonal covariance matrix
        for (auto i = 0; i < cov.rows(); ++i) {
            for (auto j = 0; j < cov.cols(); ++j) {
                if (i != j) {
                    cov(i, j) = 0;
                }
            }
        }

        auto N = static_cast<CType>(df.valid_rows(variables));
        auto d = static_cast<CType>(variables.size());
        // Normal Reference Rule formula squared for the bandwidth
        auto k = std::pow(4. / (N * (d + 2.)), 2. / (d + 4));

        if constexpr (std::is_same_v<ArrowType, arrow::DoubleType>) {
            return k * cov;
        } else {
            return (k * cov).template cast<double>();
        }
    }
};

}  // namespace kde

#endif  // PYBNESIAN_KDE_NORMALREFERENCERULE_HPP