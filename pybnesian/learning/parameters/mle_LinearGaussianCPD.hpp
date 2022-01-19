#ifndef PYBNESIAN_LEARNING_PARAMETERS_MLE_LINEARGAUSSIANCPD_HPP
#define PYBNESIAN_LEARNING_PARAMETERS_MLE_LINEARGAUSSIANCPD_HPP

#include <learning/parameters/mle_base.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>

using factors::continuous::LinearGaussianCPD;

namespace learning::parameters {

template <typename ArrowType, bool contains_null>
typename LinearGaussianCPD::ParamsClass _fit_1parent(const DataFrame& df,
                                                     const std::string& variable,
                                                     const std::string& evidence) {
    auto [y, x] = [&df, &variable, &evidence]() {
        if constexpr (contains_null) {
            auto combined_bitmap = df.combined_bitmap(variable, evidence);
            auto y = df.to_eigen<false, ArrowType>(combined_bitmap, variable);
            auto x = df.to_eigen<false, ArrowType>(combined_bitmap, evidence);
            return std::make_tuple(std::move(*y), std::move(*x));
        } else {
            auto y = df.to_eigen<false, ArrowType, contains_null>(variable);
            auto x = df.to_eigen<false, ArrowType, contains_null>(evidence);
            return std::make_tuple(std::move(*y), std::move(*x));
        }
    }();

    auto rows = y.rows();

    auto my = y.mean();
    auto mx = x.mean();

    auto dy = (y.array() - my);
    auto dx = (x.array() - mx);
    auto var_x = dx.matrix().squaredNorm() / (rows - 1);

    if (var_x < util::machine_tol) {
        auto beta = VectorXd(2);
        beta << my, 0;
        auto v = dy.matrix().squaredNorm() / (rows - 2);

        if (rows <= 2) {
            return typename LinearGaussianCPD::ParamsClass{/*.beta = */ beta,
                                                           /*.variance = */ std::numeric_limits<double>::infinity()};
        } else {
            return typename LinearGaussianCPD::ParamsClass{/*.beta = */ beta,
                                                           /*.variance = */ v};
        }
    }

    auto cov_yx = (dy * dx).sum() / (rows - 1);

    auto b = cov_yx / var_x;
    auto a = my - b * mx;

    auto beta = VectorXd(2);
    beta << a, b;

    if (rows <= 2) {
        return typename LinearGaussianCPD::ParamsClass{/*.beta = */ beta,
                                                       /*.variance = */ std::numeric_limits<double>::infinity()};
    }

    auto v = (dy - b * dx).matrix().squaredNorm() / (rows - 2);

    return typename LinearGaussianCPD::ParamsClass{/*.beta = */ beta,
                                                   /*.variance = */ v};
}

template <typename ArrowType, bool contains_null>
typename LinearGaussianCPD::ParamsClass _fit_2parent(const DataFrame& df,
                                                     const std::string& variable,
                                                     const std::vector<std::string>& evidence) {
    auto [y, x1, x2] = [&df, &variable, &evidence]() {
        if constexpr (contains_null) {
            auto combined_bitmap = df.combined_bitmap(variable, evidence);
            auto y = df.to_eigen<false, ArrowType>(combined_bitmap, variable);
            auto x1 = df.to_eigen<false, ArrowType>(combined_bitmap, evidence[0]);
            auto x2 = df.to_eigen<false, ArrowType>(combined_bitmap, evidence[1]);
            return std::make_tuple(std::move(y), std::move(x1), std::move(x2));
        } else {
            auto y = df.to_eigen<false, ArrowType, contains_null>(variable);
            auto x1 = df.to_eigen<false, ArrowType, contains_null>(evidence[0]);
            auto x2 = df.to_eigen<false, ArrowType, contains_null>(evidence[1]);
            return std::make_tuple(std::move(y), std::move(x1), std::move(x2));
        }
    }();

    auto rows = y->rows();

    auto mean_x1 = x1->mean();
    auto dx1 = (x1->array() - mean_x1);
    auto var_x1 = dx1.matrix().squaredNorm() / (rows - 1);
    auto singular1 = var_x1 < util::machine_tol;

    auto mean_x2 = x2->mean();
    auto dx2 = (x2->array() - mean_x2);
    auto var_x2 = dx2.matrix().squaredNorm() / (rows - 1);

    auto cov_xx = (dx1 * dx2).sum() / (rows - 1);

    auto singular2 =
        var_x2 < util::machine_tol || std::abs(cov_xx / std::sqrt(var_x1 * var_x2)) > (1 - util::machine_tol);

    auto mean_y = y->mean();
    auto dy = (y->array() - mean_y);

    VectorXd beta(3);
    double variance = 0;
    if (singular1) {
        if (singular2) {
            beta << mean_y, 0, 0;
            variance = dy.matrix().squaredNorm() / (rows - 3);
        } else {
            auto cov_yx2 = (dy * dx2).sum() / (rows - 1);
            auto b2 = cov_yx2 / var_x2;
            auto a = mean_y - b2 * mean_x2;
            beta << a, 0, b2;
            variance = (dy - b2 * dx2).matrix().squaredNorm() / (rows - 3);
        }
    } else {
        if (singular2) {
            auto cov_yx1 = (dy * dx1).sum() / (rows - 1);
            auto b1 = cov_yx1 / var_x1;
            auto a = mean_y - b1 * mean_x1;
            beta << a, b1, 0;
            variance = (dy - b1 * dx1).matrix().squaredNorm() / (rows - 3);
        } else {
            auto cov_yx1 = (dy * dx1).sum() / (rows - 1);
            auto cov_yx2 = (dy * dx2).sum() / (rows - 1);

            auto den = var_x1 * var_x2 - cov_xx * cov_xx;
            auto b1 = (var_x2 * cov_yx1 - cov_xx * cov_yx2) / den;
            auto b2 = (cov_yx2 - b1 * cov_xx) / var_x2;

            auto a = mean_y - b1 * mean_x1 - b2 * mean_x2;

            beta << a, b1, b2;
            variance = (dy - b1 * dx1 - b2 * dx2).matrix().squaredNorm() / (rows - 3);
        }
    }

    if (rows <= 3) {
        return typename LinearGaussianCPD::ParamsClass{/*.beta = */ beta,
                                                       /*.variance = */ std::numeric_limits<double>::infinity()};
    } else {
        return typename LinearGaussianCPD::ParamsClass{/*.beta = */ beta,
                                                       /*.variance = */ variance};
    }
}

template <typename ArrowType, bool contains_null>
typename LinearGaussianCPD::ParamsClass _fit_nparent(const DataFrame& df,
                                                     const std::string& variable,
                                                     const std::vector<std::string>& evidence) {
    auto [y, X] = [&df, &variable, &evidence]() {
        if constexpr (contains_null) {
            auto combined_bitmap = df.combined_bitmap(variable, evidence);
            auto y = df.to_eigen<false, ArrowType>(combined_bitmap, variable);
            auto X = df.to_eigen<true, ArrowType>(combined_bitmap, evidence);
            return std::make_tuple(std::move(y), std::move(X));
        } else {
            auto y = df.to_eigen<false, ArrowType, contains_null>(variable);
            auto X = df.to_eigen<true, ArrowType, contains_null>(evidence);
            return std::make_tuple(std::move(y), std::move(X));
        }
    }();

    auto rows = y->rows();

    const auto b = X->colPivHouseholderQr().solve(*y).eval();

    if (rows <= b.rows()) {
        if constexpr (std::is_same_v<typename ArrowType::c_type, double>) {
            return typename LinearGaussianCPD::ParamsClass{/*.beta = */ b,
                                                           /*.variance = */ std::numeric_limits<double>::infinity()};
        } else {
            return typename LinearGaussianCPD::ParamsClass{/*.beta = */ b.template cast<double>(),
                                                           /*.variance = */ std::numeric_limits<double>::infinity()};
        }
    }

    auto r = (*X) * b;
    auto v = ((*y) - r).squaredNorm() / (rows - b.rows());

    if constexpr (std::is_same_v<typename ArrowType::c_type, double>) {
        return typename LinearGaussianCPD::ParamsClass{/*.beta = */ b,
                                                       /*.variance = */ v};
    } else {
        return typename LinearGaussianCPD::ParamsClass{/*.beta = */ b.template cast<double>(),
                                                       /*.variance = */ v};
    }
}

template <typename ArrowType, bool contains_null>
typename LinearGaussianCPD::ParamsClass _fit(const DataFrame& df,
                                             const std::string& variable,
                                             const std::vector<std::string>& evidence) {
    if (evidence.size() == 0) {
        auto v = df.to_eigen<false, ArrowType, contains_null>(variable);

        auto mean = v->mean();
        auto b = VectorXd(1);
        b(0) = mean;

        if (v->rows() == 1) {
            return typename LinearGaussianCPD::ParamsClass{/*.beta = */ b,
                                                           /*.variance = */ std::numeric_limits<double>::infinity()};
        }

        auto var = (v->array() - mean).matrix().squaredNorm();
        return typename LinearGaussianCPD::ParamsClass{/*.beta = */ b,
                                                       /*.variance = */ var / (v->rows() - 1)};
    } else if (evidence.size() == 1) {
        return _fit_1parent<ArrowType, contains_null>(df, variable, evidence[0]);
    } else if (evidence.size() == 2) {
        return _fit_2parent<ArrowType, contains_null>(df, variable, evidence);
    } else {
        return _fit_nparent<ArrowType, contains_null>(df, variable, evidence);
    }
}

}  // namespace learning::parameters

#endif  // PYBNESIAN_LEARNING_PARAMETERS_MLE_LINEARGAUSSIANCPD_HPP