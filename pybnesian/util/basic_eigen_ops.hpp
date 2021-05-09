#ifndef PYBNESIAN_UTIL_ARROW_BASIC_EIGEN_OPS_HPP
#define PYBNESIAN_UTIL_ARROW_BASIC_EIGEN_OPS_HPP

#include <cmath>
#include <Eigen/Dense>

using Eigen::Dynamic, Eigen::Matrix, Eigen::VectorXd;

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

template <typename CType>
Matrix<CType, Dynamic, 1> logspace(CType start, CType end, int num) {
    Matrix<CType, Dynamic, 1> res(num);

    res(0) = std::pow(10., start);

    if (num > 1) {
        res(num - 1) = std::pow(10., end);

        CType step = (end - start) / (num - 1);
        CType cur = start + step;
        for (int i = 1, end = num - 1; i < end; ++i, cur += step) {
            res(i) = std::pow(10., cur);
        }
    }

    return res;
}

// This method is needed until Eigen 3.4
template <typename M>
Matrix<typename M::Scalar, Dynamic, Dynamic> filter(const M& m,
                                                    const std::vector<int>& row_indices,
                                                    const std::vector<int>& col_indices) {
    using MatrixType = Matrix<typename M::Scalar, Dynamic, Dynamic>;

    MatrixType out(row_indices.size(), col_indices.size());

    for (size_t i = 0; i < row_indices.size(); ++i) {
        for (size_t j = 0; j < col_indices.size(); ++j) {
            out(i, j) = m(row_indices[i], col_indices[j]);
        }
    }

    return out;
}

// This method is needed until Eigen 3.4
template <typename M>
Matrix<typename M::Scalar, Dynamic, Dynamic> filter_cols(const M& m, const std::vector<int>& indices) {
    using MatrixType = Matrix<typename M::Scalar, Dynamic, Dynamic>;

    MatrixType out(m.rows(), indices.size());

    auto rows = m.rows();
    auto iptr = m.data();
    auto optr = out.data();
    for (auto i : indices) {
        std::memcpy(optr, iptr + rows * i, rows * sizeof(typename M::Scalar));
        optr += rows;
    }

    return out;
}

}  // namespace util

#endif  // PYBNESIAN_UTIL_ARROW_BASIC_EIGEN_OPS_HPP