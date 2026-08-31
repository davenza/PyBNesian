#ifndef PYBNESIAN_KDE_SCOTTSBANDWIDTH_HPP
#define PYBNESIAN_KDE_SCOTTSBANDWIDTH_HPP

namespace kde {

class ScottsBandwidth : public BandwidthSelector {
public:
    /**
     * @brief Public function for calculating the diagonal bandwidth matrix using Scott's Rule given the data and
     * variables.
     *
     * @param df Dataframe.
     * @param variables Variables.
     * @return VectorXd Diagonal bandwidth vector.
     */
    VectorXd diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override {
        if (variables.empty()) return VectorXd(0);

        size_t valid_rows = df.valid_rows(variables);
        if (valid_rows <= 1) {
            std::stringstream ss;
            ss << "ScottsBandwidth::diag_bandwidth -> Diagonal bandwidth matrix of " << std::to_string(variables.size())
               << " variables [" << variables[0];
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
                    "ScottsBandwidth::diag_bandwidth -> Wrong data type to fit bandwidth. [double] or [float] data is "
                    "expected.");
        }
    }
    /**
     * @brief Public function for calculating the bandwidth matrix using Scott's Rule given the data and variables.
     *
     * @param df Data
     * @param variables Variables.
     * @return MatrixXd Bandwidth matrix.
     */
    MatrixXd bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override {
        if (variables.empty()) return MatrixXd(0, 0);

        size_t valid_rows = df.valid_rows(variables);
        if (valid_rows <= variables.size()) {
            std::stringstream ss;
            ss << "ScottsBandwidth::bandwidth -> Bandwidth matrix of " << std::to_string(variables.size())
               << " variables [" << variables[0];
            for (size_t i = 1; i < variables.size(); ++i) {
                ss << ", " << variables[i];
            }
            ss << "] cannot be estimated with " << std::to_string(valid_rows) << " instances";

            throw util::singular_covariance_data(ss.str());
        }

        switch (df.same_type(variables)->id()) {
            case Type::DOUBLE:
                return bandwidth<arrow::DoubleType>(df, variables);
            case Type::FLOAT:
                return bandwidth<arrow::FloatType>(df, variables);
            default:
                throw std::invalid_argument(
                    "ScottsBandwidth::bandwidth -> Wrong data type to fit bandwidth. [double] or [float] data is "
                    "expected.");
        }
    }

    std::string ToString() const override { return "ScottsBandwidth"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<ScottsBandwidth> __setstate__(py::tuple&) { return std::make_shared<ScottsBandwidth>(); }

private:
    /**
     * @brief Private function for calculating the diagonal bandwidth matrix using Scott's Rule given the data and
     * variables
     *
     * @tparam ArrowType Arrow Data type.
     * @param df Dataframe.
     * @param variables Variables.
     * @return VectorXd Diagonal bandwidth vector.
     */
    template <typename ArrowType>
    VectorXd diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
        using CType = typename ArrowType::c_type;

        auto N = static_cast<CType>(df.valid_rows(variables));
        auto d = static_cast<CType>(variables.size());

        auto k = std::pow(N, -2. / (d + 4.));
        VectorXd bandwidth(variables.size());

        if (df.null_count(variables) > 0) {
            auto combined_bitmap = df.combined_bitmap(variables);

            for (size_t i = 0; i < variables.size(); ++i) {
                bandwidth(i) = k * df.cov<ArrowType>(combined_bitmap, variables[i]);
            }
        } else {
            for (size_t i = 0; i < variables.size(); ++i) {
                bandwidth(i) = k * df.cov<ArrowType, false>(variables[i]);
            }
        }

        return bandwidth;
    }
    /**
     * @brief Private function for calculating the bandwidth matrix using Scott's Rule given the data and variables.
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
        if (!util::is_psd(cov)) {
            std::stringstream ss;
            ss << "Covariance matrix for variables [" << variables[0];
            for (size_t i = 1; i < variables.size(); ++i) {
                ss << ", " << variables[i];
            }
            ss << "] is not positive-definite.";
            throw util::singular_covariance_data(ss.str());
        }

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

        // Scott's Rule formula
        auto k = std::pow(N, -2. / (d + 4));

        if constexpr (std::is_same_v<ArrowType, arrow::DoubleType>) {
            return k * cov;
        } else {
            return (k * cov).template cast<double>();
        }
    }
};

}  // namespace kde

#endif  // PYBNESIAN_KDE_SCOTTSBANDWIDTH_HPP