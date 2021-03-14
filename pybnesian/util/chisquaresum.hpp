#ifndef PYBNESIAN_UTIL_CHISQUARESUM_HPP
#define PYBNESIAN_UTIL_CHISQUARESUM_HPP

#include <Eigen/Dense>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <util/rpoly.hpp>
#include <util/uniroot.hpp>

using boost::math::binomial_coefficient, boost::math::gamma_distribution, boost::math::cdf, boost::math::complement;
using Eigen::VectorXd, Eigen::Dynamic, Eigen::Matrix;

namespace util {

namespace detail {

template <typename VectorType>
VectorType chisquaresum_moments(VectorType& coeffs, int p) {
    using Scalar = typename VectorType::Scalar;
    VectorType cumulants(2 * p);

    cumulants(0) = coeffs.sum();
    cumulants(1) = 2 * coeffs.squaredNorm();

    // Start loop in r = 3, so 2^(r-1)*(r-1)! = 8
    Scalar fact_const = 8;
    for (int i = 2, end = 2 * p; i < end; ++i) {
        cumulants(i) = fact_const * coeffs.array().pow(i + 1).sum();
        fact_const *= 2 * (i + 1);
    }

    VectorType moments = cumulants;
    moments(1) += moments(0) * moments(0);
    for (int i = 2, end = 2 * p; i < end; ++i) {
        auto offset = cumulants(0) * moments(i - 1) + i * cumulants(1) * moments(i - 2);

        for (int j = 2; j < i; ++j) {
            offset += binomial_coefficient<Scalar>(i, j) * cumulants(j) * moments(i - j - 1);
        }

        moments(i) += offset;
    }

    return moments;
}

template <typename VectorType>
Matrix<typename VectorType::Scalar, Dynamic, Dynamic> delta_matrix_template(VectorType& moments, int size_matrix) {
    using Scalar = typename VectorType::Scalar;
    using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;

    MatrixType t(size_matrix, size_matrix);

    t(0) = 1;
    t(0, 1) = t(1, 0) = moments(0);

    // Fill two first columns first
    for (int i = 2; i < size_matrix; ++i) {
        t(i, 0) = moments(i - 1);
    }

    for (int i = 1; i < size_matrix; ++i) {
        t(i, 1) = moments(i);
    }

    // Fill remaining columns
    for (int j = 2; j < size_matrix; ++j) {
        for (int i = 0; i < size_matrix; ++i) {
            t(i, j) = moments(i + j - 1);
        }
    }

    return t;
}

template <typename Scalar>
Matrix<Scalar, Dynamic, 1> delta_mult_coefficients(Scalar alpha, int size_matrix) {
    using VectorType = Matrix<Scalar, Dynamic, 1>;
    auto max_r = 2 * size_matrix - 2;

    VectorType mult_coefficients(max_r - 1);
    mult_coefficients(0) = 1 + alpha;
    for (int i = 1, end = max_r - 1; i < end; ++i) {
        mult_coefficients(i) = mult_coefficients(i - 1) * (1 + (i + 1) * alpha);
    }

    return mult_coefficients.cwiseInverse();
}

template <typename MatrixType>
void delta_apply_mult_coefficients(MatrixType& delta,
                                   Matrix<typename MatrixType::Scalar, Dynamic, 1>& mult_coefficients) {
    auto p = delta.rows();
    // Divide first two columns
    for (int i = 2; i < p; ++i) {
        delta(i, 0) *= mult_coefficients(i - 2);
    }

    for (int i = 1; i < p; ++i) {
        delta(i, 1) *= mult_coefficients(i - 1);
    }

    // Divide remaining columns.
    for (int j = 2; j < p; ++j) {
        for (int i = 0; i < p; ++i) {
            delta(i, j) *= mult_coefficients(i + j - 2);
        }
    }
}

template <typename Scalar>
struct DeltaMatrixDeterminant {
    using VectorType = Matrix<Scalar, Dynamic, 1>;
    using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
    Scalar operator()(Scalar alpha) {
        MatrixType copy = matrix;

        auto mult_coefficients = delta_mult_coefficients(alpha, matrix.rows());
        delta_apply_mult_coefficients(copy, mult_coefficients);

        return copy.determinant();
    }

    MatrixType matrix;
};

template <typename VectorType>
typename VectorType::Scalar lambda_tilde(VectorType& moments, int p) {
    using Scalar = typename VectorType::Scalar;

    // This is the closed solution for lambda_1
    Scalar last_lambda = moments(1) / (moments(0) * moments(0)) - 1;
    for (auto i = 2; i <= p; ++i) {
        DeltaMatrixDeterminant<Scalar> mdet{/*.matrix = */ delta_matrix_template(moments, i + 1)};
        last_lambda = util::uniroot(mdet, static_cast<Scalar>(0), last_lambda, static_cast<Scalar>(1e-9), 1000);
    }

    return last_lambda;
}

template <typename VectorType>
Matrix<typename VectorType::Scalar, Dynamic, 1> mu_roots(VectorType& moments,
                                                         typename VectorType::Scalar lambda_tilde,
                                                         int p) {
    using Scalar = typename VectorType::Scalar;
    using VecType = Matrix<Scalar, Dynamic, 1>;

    auto M = delta_matrix_template(moments, p + 1);
    auto mult_coefficients = delta_mult_coefficients(lambda_tilde, p + 1);
    delta_apply_mult_coefficients(M, mult_coefficients);

    VecType poly_coeffs(p + 1);

    M.col(p) = VectorType::Zero(p + 1);

    for (int i = p; i >= 0; --i) {
        M(i, p) = 1;
        poly_coeffs(p - i) = M.determinant();
        M(i, p) = 0;
    }

    VecType real_roots = VecType::Zero(p);
    VecType complex_roots = VecType::Zero(p);

    util::RPoly<Scalar> poly_solver;
    poly_solver.findRoots(poly_coeffs.data(), p, real_roots.data(), complex_roots.data());

    return real_roots;
}

template <typename VectorType>
VectorType mixture_proportions(VectorType& mu, VectorType& moments, typename VectorType::Scalar lambda_tilde, int p) {
    using Scalar = typename VectorType::Scalar;
    using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;

    MatrixType vandermonde(p, p);

    vandermonde.row(0) = VectorType::Ones(p);
    vandermonde.row(1) = mu;
    vandermonde.row(2) = mu.cwiseProduct(mu);
    for (int i = 3; i < p; ++i) {
        vandermonde.row(i) = mu.array().pow(i);
    }

    VectorType delta_vec(p);
    delta_vec(0) = 1;
    delta_vec(1) = moments(0);
    delta_vec(2) = moments(1) / (1 + lambda_tilde);
    delta_vec(3) = moments(2) / ((1 + lambda_tilde) * (1 + 2 * lambda_tilde));

    auto mult_coeff = (1 + lambda_tilde) * (1 + 2 * lambda_tilde);
    for (int i = 4; i < p; ++i) {
        mult_coeff *= (1 + (i - 1) * lambda_tilde);
        delta_vec(i) = moments(i - 1) * (1. / mult_coeff);
    }

    return vandermonde.colPivHouseholderQr().solve(delta_vec);
}

template <typename VectorType>
typename VectorType::Scalar lpb4_cdf(VectorType& prop,
                                     VectorType& mu,
                                     typename VectorType::Scalar lambda_tilde,
                                     typename VectorType::Scalar quantile) {
    using Scalar = typename VectorType::Scalar;
    auto k = 1. / lambda_tilde;

    Scalar res = 0;
    for (int i = 0; i < prop.rows(); ++i) {
        auto theta = mu(i) * lambda_tilde;

        if (theta <= 0) {
            throw std::runtime_error("Wrong theta parameter.");
        }

        gamma_distribution<Scalar> gamma(k, theta);
        res += prop(i) * cdf(gamma, quantile);
    }

    return res;
}

template <typename VectorType>
typename VectorType::Scalar lpb4_cdf_complement(VectorType& prop,
                                                VectorType& mu,
                                                typename VectorType::Scalar lambda_tilde,
                                                typename VectorType::Scalar quantile) {
    using Scalar = typename VectorType::Scalar;
    auto k = 1. / lambda_tilde;
    Scalar res = 0;
    for (int i = 0; i < prop.rows(); ++i) {
        auto theta = mu(i) * lambda_tilde;
        gamma_distribution<Scalar> gamma(k, theta);
        res += prop(i) * cdf(complement(gamma, quantile));
    }

    return res;
}

}  // namespace detail

/**
 * A comparison of efficient approximations for a weighted sum of chi-squared random variables
 */
template <typename VectorType>
typename VectorType::Scalar lpb4(VectorType& coeffs, typename VectorType::Scalar quantile) {
    if (coeffs.rows() < 4) {
        throw std::invalid_argument("lbp4 requires at least 4 coefficients.");
    }

    auto p = 4;
    auto moments = detail::chisquaresum_moments(coeffs, p);
    auto ld_tilde = detail::lambda_tilde(moments, p);
    auto mu = detail::mu_roots(moments, ld_tilde, p);
    auto prop = detail::mixture_proportions(mu, moments, ld_tilde, p);
    return detail::lpb4_cdf(prop, mu, ld_tilde, quantile);
}

template <typename VectorType>
typename VectorType::Scalar lpb4_complement(VectorType& coeffs, typename VectorType::Scalar quantile) {
    if (coeffs.rows() < 4) {
        throw std::invalid_argument("lbp4 requires at least 4 coefficients.");
    }

    auto p = 4;
    auto moments = detail::chisquaresum_moments(coeffs, p);
    auto ld_tilde = detail::lambda_tilde(moments, p);
    auto mu = detail::mu_roots(moments, ld_tilde, p);
    auto prop = detail::mixture_proportions(mu, moments, ld_tilde, p);
    return detail::lpb4_cdf_complement(prop, mu, ld_tilde, quantile);
}

template <typename VectorType>
typename VectorType::Scalar hbe(VectorType& coeffs, typename VectorType::Scalar quantile) {
    using Scalar = typename VectorType::Scalar;
    auto k1 = coeffs.sum();
    auto squared = coeffs.array().square().matrix();
    auto k2 = 2 * squared.sum();
    auto k3 = 8 * (coeffs.dot(squared));

    auto nu = 8 * (k2 * k2 * k2) / (k3 * k3);

    auto statistic = std::sqrt(2 * nu / k2) * (quantile - k1) + nu;

    gamma_distribution<Scalar> gamma(nu / 2., 2);

    return cdf(gamma, statistic);
}

template <typename VectorType>
typename VectorType::Scalar hbe_complement(VectorType& coeffs, typename VectorType::Scalar quantile) {
    using Scalar = typename VectorType::Scalar;
    auto k1 = coeffs.sum();
    auto squared = coeffs.array().square().matrix();
    auto k2 = 2 * squared.sum();
    auto k3 = 8 * (coeffs.dot(squared));

    auto nu = 8 * (k2 * k2 * k2) / (k3 * k3);

    auto statistic = std::sqrt(2 * nu / k2) * (quantile - k1) + nu;

    gamma_distribution<Scalar> gamma(nu / 2., 2);

    return cdf(complement(gamma, statistic));
}

}  // namespace util

#endif  // PYBNESIAN_UTIL_CHISQUARESUM_HPP