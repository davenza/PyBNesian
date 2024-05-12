#ifndef PYBNESIAN_UTIL_BASIC_EIGEN_OPS_HPP
#define PYBNESIAN_UTIL_BASIC_EIGEN_OPS_HPP

#include <util/math_constants.hpp>

namespace util {

template <typename Vector>
typename Vector::Scalar sse(const Vector& v) {
    auto mean = v.mean();
    auto diff = v.array() - mean;
    return diff.matrix().squaredNorm();
}

template <typename Vector>
typename Vector::Scalar var(const Vector& v) {
    return sse(v) / (v.rows() - 1);
}

template <typename Vector>
typename Vector::Scalar sd(const Vector& v) {
    return std::sqrt(var(v));
}

template <typename Matrix>
void normalize_cols(Matrix& m) {
    for (int i = 0; i < m.cols(); ++i) {
        auto me = m.col(i).mean();
        auto s = sd(m.col(i));

        if (s != 0)
            m.col(i) = (m.col(i).array() - me) * (1 / s);
        else
            m.col(i).array() = 0;
    }
}

template <typename M>
Matrix<typename M::Scalar, Dynamic, Dynamic> sse_mat(M& m) {
    using Scalar = typename M::Scalar;
    using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
    MatrixType res(m.cols(), m.cols());

    for (auto i = 0; i < m.cols(); ++i) {
        auto d_i = m.col(i);
        res(i, i) = d_i.squaredNorm();

        for (auto j = i + 1; j < m.cols(); ++j) {
            res(i, j) = res(j, i) = d_i.dot(m.col(j));
        }
    }

    return res;
}

template <typename M>
Matrix<typename M::Scalar, Dynamic, Dynamic> sse_mat(M& x, M& y) {
    using Scalar = typename M::Scalar;
    using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
    MatrixType res(x.cols(), y.cols());

    for (auto i = 0; i < x.cols(); ++i) {
        for (auto j = 0; j < y.cols(); ++j) {
            res(i, j) = x.col(i).dot(y.col(j));
        }
    }

    return res;
}

template <typename M>
Matrix<typename M::Scalar, Dynamic, 1> sse_cols(M& m) {
    using Scalar = typename M::Scalar;
    using VectorType = Matrix<Scalar, Dynamic, 1>;
    VectorType res(m.cols());

    for (auto i = 0; i < m.cols(); ++i) {
        res(i) = m.col(i).squaredNorm();
    }

    return res;
}

template <typename M>
Matrix<typename M::Scalar, Dynamic, Dynamic> cov(M& m) {
    using Scalar = typename M::Scalar;
    using VectorType = Matrix<Scalar, Dynamic, 1>;
    // MatrixType res(m.cols(), m.cols());

    VectorType means(m.cols());
    for (auto i = 0; i < m.cols(); ++i) {
        means(i) = m.col(i).mean();
    }

    auto tmp = m.rowwise() - means.transpose();

    Scalar inv_N = 1 / static_cast<Scalar>(m.rows() - 1);
    return sse_mat(tmp) * inv_N;
}

template <typename M>
Matrix<typename M::Scalar, Dynamic, Dynamic> cov(M& x, M& y) {
    using Scalar = typename M::Scalar;
    using VectorType = Matrix<Scalar, Dynamic, 1>;
    VectorType means_x(x.cols());
    VectorType means_y(y.cols());

    for (auto i = 0; i < x.cols(); ++i) {
        means_x(i) = x.col(i).mean();
    }

    for (auto i = 0; i < y.cols(); ++i) {
        means_y(i) = y.col(i).mean();
    }

    Scalar inv_N = 1 / static_cast<Scalar>(x.rows() - 1);

    auto tmp_x = x.rowwise() - means_x.transpose();
    auto tmp_y = y.rowwise() - means_y.transpose();
    return sse_mat(tmp_x, tmp_y) * inv_N;
}

template <typename M>
Matrix<typename M::Scalar, Dynamic, Dynamic> sqrt_matrix(const M& m) {
    using MatrixType = Matrix<typename M::Scalar, Dynamic, Dynamic>;
    if (m.rows() == 1 && m.cols() == 1) {
        return MatrixType::Constant(1, 1, std::sqrt(m(0, 0)));
    }

    Eigen::SelfAdjointEigenSolver<M> svd_solver(m);
    return svd_solver.operatorSqrt();
}

// Checks whether M is positive definite.
template <typename M>
bool is_psd(const M& m) {
    using MatrixType = Matrix<typename M::Scalar, Dynamic, Dynamic>;
    Eigen::SelfAdjointEigenSolver<MatrixType> eigen_solver(m, Eigen::EigenvaluesOnly);

    auto tol = eigen_solver.eigenvalues().maxCoeff() * m.rows() * std::numeric_limits<typename M::Scalar>::epsilon();

    if (eigen_solver.eigenvalues().minCoeff() < tol) {
        return false;
    }

    return true;
}

}  // namespace util

#endif  // PYBNESIAN_UTIL_BASIC_EIGEN_OPS_HPP