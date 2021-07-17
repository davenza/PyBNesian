#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_LINEARCORRELATION_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_LINEARCORRELATION_HPP

#include <algorithm>
#include <dataset/dataset.hpp>
#include <learning/independences/independence.hpp>
#include <util/math_constants.hpp>

using dataset::DataFrame;
using Eigen::LLT, Eigen::Ref;
using learning::independences::IndependenceTest;

namespace learning::independences::continuous {

double cor_pvalue(double cor, int df);

template <typename EigenMat>
double cor_0cond(const EigenMat& cov, int v1, int v2) {
    using CType = typename EigenMat::Scalar;
    if (cov(v1, v1) < util::machine_tol || cov(v2, v2) < util::machine_tol) return 0;

    auto cor = cov(v1, v2) / sqrt(cov(v1, v1) * cov(v2, v2));

    return std::clamp(cor, static_cast<CType>(-1.), static_cast<CType>(1.));
}

template <typename EigenValues, typename EigenVectors>
double cor_svd(const EigenValues& d, const EigenVectors u) {
    double p11 = 0;
    double p12 = 0;
    double p22 = 0;
    double tol = d.rows() * d[d.rows() - 1] * std::numeric_limits<double>::epsilon();
    for (int i = 0; i < d.rows(); ++i) {
        if (d[i] > tol) {
            double inv_d = 1. / d[i];
            p11 += u(0, i) * u(0, i) * inv_d;
            p12 += u(0, i) * u(1, i) * inv_d;
            p22 += u(1, i) * u(1, i) * inv_d;
        }
    }

    if (p11 < util::machine_tol || p22 < util::machine_tol) return 0;

    return std::clamp(-p12 / (sqrt(p11 * p22)), -1., 1.);
}

template <typename EigenMatrix>
double cor_1cond(EigenMatrix& cov, int v1, int v2, int ev) {
    Eigen::Matrix3d m;
    m << cov(v1, v1), cov(v1, v2), cov(v1, ev), cov(v2, v1), cov(v2, v2), cov(v2, ev), cov(ev, v1), cov(ev, v2),
        cov(ev, ev);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(m);

    auto& d = eigen_solver.eigenvalues();
    auto& u = eigen_solver.eigenvectors();
    return cor_svd(d, u);
}

template <typename EigenMatrix>
double cor_general(EigenMatrix& cov) {
    Eigen::SelfAdjointEigenSolver<EigenMatrix> eigen_solver(cov);
    auto& d = eigen_solver.eigenvalues();
    auto& u = eigen_solver.eigenvectors();
    return cor_svd(d, u);
}

class LinearCorrelation : public IndependenceTest {
public:
    LinearCorrelation(const DataFrame& df) : m_df(df), m_cached_cov(false), m_indices(), m_cov() {
        auto continuous_indices = df.continuous_columns();

        if (continuous_indices.size() < 2) {
            throw std::invalid_argument("DataFrame does not contain enough continuous columns.");
        }

        if (m_df.null_count(continuous_indices) == 0) {
            m_cached_cov = true;
            for (int i = 0, size = continuous_indices.size(); i < size; ++i) {
                m_indices.insert(std::make_pair(m_df->column_name(continuous_indices[i]), i));
            }
            switch (m_df.same_type(continuous_indices)->id()) {
                case Type::DOUBLE:
                    m_cov = *(m_df.cov<arrow::DoubleType, false>(continuous_indices).release());
                    break;
                case Type::FLOAT:
                    m_cov = m_df.cov<arrow::FloatType, false>(continuous_indices)->template cast<double>();
                    break;
                default:
                    break;
            }
        }
    }

    double pvalue(const std::string& v1, const std::string& v2) const override {
        if (m_cached_cov)
            return pvalue_cached(v1, v2);
        else
            return pvalue_impl(v1, v2);
    }

    double pvalue(const std::string& v1, const std::string& v2, const std::string& ev) const override {
        if (m_cached_cov)
            return pvalue_cached(v1, v2, ev);
        else
            return pvalue_impl(v1, v2, ev);
    }

    double pvalue(const std::string& v1, const std::string& v2, const std::vector<std::string>& ev) const override {
        if (m_cached_cov)
            return pvalue_cached(v1, v2, ev);
        else
            return pvalue_impl(v1, v2, ev);
    }

    int num_variables() const override { return m_df->num_columns(); }

    std::vector<std::string> variable_names() const override { return m_df.column_names(); }

    const std::string& name(int i) const override { return m_df.name(i); }

    bool has_variables(const std::string& name) const override { return m_df.has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const override { return m_df.has_columns(cols); }

private:
    int cached_index(int v) const {
        auto it = m_indices.find(m_df->column_name(v));
        if (it == m_indices.end())
            throw std::invalid_argument("Continuous variable " + std::to_string(v) +
                                        " not present in LinearCorrelation.");
        return it->second;
    }

    int cached_index(const std::string& name) const {
        auto it = m_indices.find(name);
        if (it == m_indices.end())
            throw std::invalid_argument("Continuous variable " + name + " not present in LinearCorrelation.");
        return it->second;
    }

    double pvalue_cached(const std::string& v1, const std::string& v2) const;
    double pvalue_cached(const std::string& v1, const std::string& v2, const std::string& ev) const;
    double pvalue_cached(const std::string& v1, const std::string& v2, const std::vector<std::string>& ev) const;

    double pvalue_impl(const std::string& v1, const std::string& v2) const;
    double pvalue_impl(const std::string& v1, const std::string& v2, const std::string& ev) const;
    double pvalue_impl(const std::string& v1, const std::string& v2, const std::vector<std::string>& ev) const;

    const DataFrame m_df;
    bool m_cached_cov;
    std::unordered_map<std::string, int> m_indices;
    MatrixXd m_cov;
};

using DynamicLinearCorrelation = DynamicIndependenceTestAdaptator<LinearCorrelation>;

}  // namespace learning::independences::continuous

#endif  // PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_LINEARCORRELATION_HPP
