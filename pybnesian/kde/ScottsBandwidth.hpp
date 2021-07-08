#ifndef PYBNESIAN_KDE_SCOTTSBANDWIDTH_HPP
#define PYBNESIAN_KDE_SCOTTSBANDWIDTH_HPP

namespace kde {

class ScottsBandwidth : public BandwidthEstimator {
public:
    VectorXd estimate_diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override {
        switch (df.same_type(variables)->id()) {
            case Type::DOUBLE:
                return estimate_diag_bandwidth<arrow::DoubleType>(df, variables);
            case Type::FLOAT:
                return estimate_diag_bandwidth<arrow::FloatType>(df, variables);
            default:
                throw py::value_error("Wrong data type to fit bandwidth. [double] or [float] data is expected.");
        }
    }

    MatrixXd estimate_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const override {
        switch (df.same_type(variables)->id()) {
            case Type::DOUBLE:
                return estimate_bandwidth<arrow::DoubleType>(df, variables);
            case Type::FLOAT:
                return estimate_bandwidth<arrow::FloatType>(df, variables);
            default:
                throw py::value_error("Wrong data type to fit bandwidth. [double] or [float] data is expected.");
        }
    }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<ScottsBandwidth> __setstate__(py::tuple&) { return std::make_shared<ScottsBandwidth>(); }

private:
    template <typename ArrowType>
    VectorXd estimate_diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
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

    template <typename ArrowType>
    MatrixXd estimate_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const {
        using CType = typename ArrowType::c_type;

        auto cov = df.cov<ArrowType>(variables);
        auto N = static_cast<CType>(df.valid_rows(variables));
        auto d = static_cast<CType>(variables.size());

        auto k = std::pow(N, -2. / (d + 4));

        if constexpr (std::is_same_v<ArrowType, arrow::DoubleType>) {
            return k * (*cov);
        } else {
            return k * cov->template cast<double>();
        }
    }
};

}  // namespace kde

#endif  // PYBNESIAN_KDE_SCOTTSBANDWIDTH_HPP