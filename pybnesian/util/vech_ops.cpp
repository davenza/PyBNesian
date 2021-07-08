#include <algorithm>
#include <util/vech_ops.hpp>

namespace util {

VectorXd vech(const MatrixXd& m) {
    auto d = m.rows();

    auto ns = d * (d + 1) / 2;

    VectorXd v(ns);

    auto offset = 0;
    for (auto i = 0; i < d - 1; ++i) {
        std::copy(m.data() + i * d + i, m.data() + (i + 1) * d, v.data() + offset);

        offset += d - i;
    }

    v(ns - 1) = m(d - 1, d - 1);

    return v;
}

MatrixXd invvech(const VectorXd& v) {
    auto ns = v.rows();
    auto d = static_cast<size_t>(-1 + std::sqrt(8 * ns + 1)) / 2;

    MatrixXd m(d, d);

    auto offset = 0;
    for (size_t i = 0; i < d - 1; ++i) {
        std::copy(v.data() + offset, v.data() + offset + d - i, m.data() + i * d + i);
        offset += d - i;
    }

    m(d - 1, d - 1) = v(ns - 1);

    for (size_t i = 0; i < (d - 1); ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            m(i, j) = m(j, i);
        }
    }

    return m;
}

MatrixXd invvech_triangular(const VectorXd& v) {
    auto ns = v.rows();
    auto d = static_cast<size_t>(-1 + std::sqrt(8 * ns + 1)) / 2;

    MatrixXd m(d, d);

    auto offset = 0;
    for (size_t i = 0; i < d - 1; ++i) {
        std::copy(v.data() + offset, v.data() + offset + d - i, m.data() + i * d + i);
        offset += d - i;
    }

    m(d - 1, d - 1) = v(ns - 1);

    for (size_t i = 0; i < (d - 1); ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            m(i, j) = 0;
        }
    }

    return m;
}

}  // namespace util