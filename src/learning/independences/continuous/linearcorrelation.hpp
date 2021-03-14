#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_LINEARCORRELATION_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_LINEARCORRELATION_HPP

#include <algorithm>
#include <dataset/dataset.hpp>
#include <learning/independences/independence.hpp>
#include <util/math_constants.hpp>

using dataset::DataFrame;
using learning::independences::IndependenceTest;
using Eigen::LLT, Eigen::Ref;

namespace learning::independences::continuous {

    double cor_pvalue(double cor, int df);

    template<typename EigenMat>
    double cor_0cond(const EigenMat& cov, int v1, int v2) {
        using CType = typename EigenMat::Scalar;
        if (cov(v1, v1) < util::machine_tol || cov(v2, v2) < util::machine_tol)
            return 0;

        auto cor = cov(v1, v2) / sqrt(cov(v1, v1) * cov(v2, v2));

        return std::clamp(cor, static_cast<CType>(-1.), static_cast<CType>(1.));
    }

    template<typename EigenValues, typename EigenVectors>
    double cor_svd(const EigenValues& d, const EigenVectors u) {
        double p11 = 0;
        double p12 = 0;
        double p22 = 0;
        double tol = d.rows() * d[d.rows()-1] * std::numeric_limits<double>::epsilon();
        for (int i = 0; i < d.rows(); ++i) {
            if (d[i] > tol) {
                double inv_d = 1./d[i];
                p11 += u(0, i) * u(0,i) * inv_d;
                p12 += u(0, i) * u(1,i) * inv_d;
                p22 += u(1, i) * u(1,i) * inv_d;
            }
        }

        if (p11 < util::machine_tol || p22 < util::machine_tol)
            return 0;

        return std::clamp(-p12 / (sqrt(p11 * p22)), -1., 1.);
    }

    template<typename EigenMatrix>
    double cor_1cond(EigenMatrix& cov, int v1, int v2, int ev) {
        Eigen::Matrix3d m;
        m << cov(v1, v1), cov(v1, v2), cov(v1, ev),
             cov(v2, v1), cov(v2, v2), cov(v2, ev),
             cov(ev, v1), cov(ev, v2), cov(ev, ev);

        auto eigen_solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(m);

        auto& d = eigen_solver.eigenvalues();
        auto& u = eigen_solver.eigenvectors();
        return cor_svd(d, u);
    }

    template<typename EigenMatrix>
    double cor_general(EigenMatrix& cov) {
        auto eigen_solver = Eigen::SelfAdjointEigenSolver<EigenMatrix>(cov);
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
                switch(m_df.same_type(continuous_indices)) {
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

        double pvalue(int v1, int v2) const override {
            if (m_cached_cov)
                return pvalue_cached(v1, v2);
            else
                return pvalue_impl(v1, v2);
        }

        double pvalue(const std::string& v1, const std::string& v2) const override {
            if (m_cached_cov)
                return pvalue_cached(v1, v2);
            else
                return pvalue_impl(v1, v2);
            
        }

        double pvalue(int v1, int v2, int ev) const override {
            if (m_cached_cov)
                return pvalue_cached(v1, v2, ev);
            else
                return pvalue_impl(v1, v2, ev);

        }

        double pvalue(const std::string& v1, const std::string& v2, const std::string& ev) const override {
            if (m_cached_cov)
                return pvalue_cached(m_indices.at(v1), m_indices.at(v2), m_indices.at(ev));
            else
                return pvalue_impl(v1, v2, ev);
        }

        double pvalue(int v1, int v2, const std::vector<int>& ev) const override {
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
       
        std::vector<std::string> variable_names() const override {
            return m_df.column_names();
        }

        const std::string& name(int i) const override {
            return m_df.name(i);
        }

        bool has_variables(const std::string& name) const override {
            return m_df.has_columns(name);
        }

        bool has_variables(const std::vector<std::string>& cols) const override {
            return m_df.has_columns(cols);
        }

    private:
        int cached_index(int v) const {
            auto it = m_indices.find(m_df->column_name(v));
            if (it == m_indices.end())
                throw std::invalid_argument("Continuous variable " + std::to_string(v) + " not present in LinearCorrelation.");
            return it->second;
        }

        int cached_index(const std::string& name) const {
            auto it = m_indices.find(name);
            if (it == m_indices.end())
                throw std::invalid_argument("Continuous variable " + name + " not present in LinearCorrelation.");
            return it->second;
        }

        template<typename VarType>
        double pvalue_cached(const VarType& v1, const VarType& v2) const;
        template<typename VarType>
        double pvalue_cached(const VarType& v1, const VarType& v2, const VarType& ev) const;
        template<typename VarType>
        double pvalue_cached(const VarType& v1, const VarType& v2, const std::vector<VarType>& ev) const;

        template<typename VarType>
        double pvalue_impl(const VarType& v1, const VarType& v2) const;
        template<typename VarType>
        double pvalue_impl(const VarType& v1, const VarType& v2, const VarType& ev) const;
        template<typename VarType>
        double pvalue_impl(const VarType& v1, const VarType& v2, const std::vector<VarType>& ev) const;

        const DataFrame m_df;
        bool m_cached_cov;
        std::unordered_map<std::string, int> m_indices;
        MatrixXd m_cov;
    };

    template<typename VarType>
    double LinearCorrelation::pvalue_cached(const VarType& v1, const VarType& v2) const {
        double cor = cor_0cond(m_cov, cached_index(v1), cached_index(v2));
        return cor_pvalue(cor, m_df->num_rows() - 2);
    }

    template<typename VarType>
    double LinearCorrelation::pvalue_impl(const VarType& v1, const VarType& v2) const {
        auto [cor, df] = [this, &v1, &v2]() {
            switch(m_df.col(v1)->type_id()) {
                case Type::DOUBLE: {
                    auto cov_ptr = m_df.cov<arrow::DoubleType>(v1, v2);
                    auto& cov = *cov_ptr;
                    double cor = cor_0cond(cov, 0, 1);
                    return std::make_pair(cor, m_df.valid_rows(v1, v2) - 2);
                }
                case Type::FLOAT: {
                    auto cov_ptr = m_df.cov<arrow::FloatType>(v1, v2);
                    auto& cov = *cov_ptr;
                    double cor = cor_0cond(cov, 0, 1);
                    return std::make_pair(cor, m_df.valid_rows(v1, v2) - 2);
                }
                default:
                    throw std::invalid_argument("Column " + m_df.name(v1) + " is not continuous");
            }
        }();

        return cor_pvalue(cor, df);
    }

    template<typename VarType>
    double LinearCorrelation::pvalue_cached(const VarType& v1, const VarType& v2, const VarType& ev) const {
        double cor = cor_1cond(m_cov, cached_index(v1), cached_index(v2), cached_index(ev));
        return cor_pvalue(cor, m_df->num_rows() - 3);
    }

    template<typename VarType>
    double LinearCorrelation::pvalue_impl(const VarType& v1, const VarType& v2, const VarType& ev) const {
        auto [cor, df] = [this, &v1, &v2, &ev]() {
            switch(m_df.col(v1)->type_id()) {
                case Type::DOUBLE: {
                    auto cov_ptr = m_df.cov<arrow::DoubleType>(v1, v2, ev);
                    auto& cov = *cov_ptr;
                    double cor = cor_general(cov);
                    return std::make_pair(cor, m_df.valid_rows(v1, v2, ev) - 3);

                }
                case Type::FLOAT: {
                    auto cov_ptr = m_df.cov<arrow::FloatType>(v1, v2, ev);
                    auto& cov = *cov_ptr;
                    double cor = cor_general(cov);
                    return std::make_pair(cor, m_df.valid_rows(v1, v2, ev) - 3);
                }
                default:
                    throw std::invalid_argument("Column " + m_df.name(v1) + " is not continuous");
            }
        }();

        return cor_pvalue(cor, df);
    }

    template<typename VarType>
    double LinearCorrelation::pvalue_cached(const VarType& v1, const VarType& v2, const std::vector<VarType>& ev) const {
        std::vector<int> cached_indices;
   
        cached_indices.push_back(cached_index(v1));
        cached_indices.push_back(cached_index(v2));

        for (auto it = ev.begin(), end = ev.end(); it != end; ++it) {
            cached_indices.push_back(cached_index(*it));
        }

        int k = cached_indices.size();
        MatrixXd cov(k, k);

        for (int i = 0; i < k; ++i) {
            cov(i,i) = m_cov(cached_indices[i], cached_indices[i]);
            for (int j = i+1; j < k; ++j) {
                cov(i, j) = cov(j, i) = m_cov(cached_indices[i], cached_indices[j]);
            }
        }

        double cor = cor_general(cov);
        return cor_pvalue(cor, m_df->num_rows() - 2 - k);
    }

    template<typename VarType>
    double LinearCorrelation::pvalue_impl(const VarType& v1, const VarType& v2, const std::vector<VarType>& ev) const {

        auto [cor, df] = [this, &v1, &v2, &ev]() {
            int k = ev.size();
            switch(m_df.col(v1)->type_id()) {
                case Type::DOUBLE: {
                    auto cov_ptr = m_df.cov<arrow::DoubleType>(v1, v2, ev) ;
                    auto& cov = *cov_ptr;
                    double cor = cor_general(cov);
                    return std::make_pair(cor, 
                                m_df.valid_rows(v1, v2, ev) - 2 - k
                            );
                }
                case Type::FLOAT: {
                    auto cov_ptr = m_df.cov<arrow::FloatType>(v1, v2, ev);
                    auto& cov = *cov_ptr;
                    double cor = cor_general(cov);
                    return std::make_pair(cor, 
                                m_df.valid_rows(v1, v2, ev) - 2 - k
                            );
                }
                default:
                    throw std::invalid_argument("Column " + m_df.name(v1) + " is not continuous");
            }
        }();

        return cor_pvalue(cor, df);
    }

    using DynamicLinearCorrelation = DynamicIndependenceTestAdaptator<LinearCorrelation>;
}

#endif //PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_LINEARCORRELATION_HPP
