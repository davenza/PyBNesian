#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_LINEARCORRELATION_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_LINEARCORRELATION_HPP

#include <dataset/dataset.hpp>
#include <learning/independences/independence.hpp>

using dataset::DataFrame;
using learning::independences::IndependenceTest;
using Eigen::LLT, Eigen::Ref;

namespace learning::independences::continuous {

    double cor_pvalue(double cor, int df);

    template<typename EigenMat>
    double cor_0cond(const EigenMat& cov, int v1, int v2) {
        return cov(v1, v2) / sqrt(cov(v1, v1) * cov(v2, v2));
    }

    template<typename EigenMat>
    double cor_1cond(const EigenMat& cov, int v1, int v2, int cond) {
        double a11 = cov(v2, v2)*cov(cond, cond) - cov(v2, cond)*cov(v2,cond);

        double det = cov(v1, v1)*a11
                     - cov(v1, v2)*(cov(v1, v2)*cov(cond, cond) - cov(v2, cond)*cov(v1, cond))
                     + cov(v1, cond)*(cov(v1, v2)*cov(v2, cond) - cov(v2, v2)* cov(v1, cond));
        
        double inv_det = 1. / det;
        double p12 = (cov(v1, cond) * cov(v2, cond) - cov(v1, v2)*cov(cond, cond)) * inv_det;
        double p11 = a11 * inv_det;
        double p22 = (cov(v1, v1) * cov(cond, cond) - cov(v1, cond) * cov(v1, cond)) * inv_det;

        return -p12 / (sqrt(p11 * p22));
    }

    template<typename EigenMatrix>
    double cor_general(EigenMatrix& cov) {
        using Scalar = typename EigenMatrix::Scalar;
        LLT<Ref<EigenMatrix>> llt(cov);

        int d = cov.rows();
        auto Lmatrix = llt.matrixL();
        EigenMatrix identity = EigenMatrix::Identity(d, d);

        // Solves and saves the result in identity
        Lmatrix.solveInPlace(identity);

        Scalar p11 = identity(d-1, d-1)*identity(d-1, d-1);
        Scalar p22 = identity(d-1, d-2)*identity(d-1, d-2) + identity(d-2, d-2)*identity(d-2, d-2);
        Scalar p12 = identity(d-1, d-1)*identity(d-1, d-2);
        
        return static_cast<double>(-p12 / sqrt(p11 * p22));
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

        double pvalue(int v1, int v2, int cond) const override {
            if (m_cached_cov)
                return pvalue_cached(v1, v2, cond);
            else
                return pvalue_impl(v1, v2, cond);

        }

        double pvalue(const std::string& v1, const std::string& v2, const std::string& cond) const override {
            if (m_cached_cov)
                return pvalue_cached(m_indices.at(v1), m_indices.at(v2), m_indices.at(cond));
            else
                return pvalue_impl(v1, v2, cond);
        }

        double pvalue(int v1, int v2, 
                        const typename std::vector<int>::const_iterator evidence_begin, 
                        const typename std::vector<int>::const_iterator evidence_end) const override {
            if (m_cached_cov)
                return pvalue_cached(v1, v2, evidence_begin, evidence_end);
            else
                return pvalue_impl(v1, v2, evidence_begin, evidence_end);
        }

         double pvalue(const std::string& v1, const std::string& v2, 
                            const typename std::vector<std::string>::const_iterator evidence_begin, 
                            const typename std::vector<std::string>::const_iterator evidence_end) const override {
            if (m_cached_cov)
                return pvalue_cached(v1, v2, evidence_begin, evidence_end);
            else
                return pvalue_impl(v1, v2, evidence_begin, evidence_end);
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
            return m_indices.at(m_df->column_name(v)); 
        }
        int cached_index(const std::string& name) const { return m_indices.at(name); }

        template<typename VarType>
        double pvalue_cached(const VarType& v1, const VarType& v2) const;
        template<typename VarType>
        double pvalue_cached(const VarType& v1, const VarType& v2, const VarType& cond) const;
        template<typename VarType, typename Iter>
        double pvalue_cached(const VarType& v1, const VarType& v2, Iter evidence_begin, Iter evidence_end) const;

        template<typename VarType>
        double pvalue_impl(const VarType& v1, const VarType& v2) const;
        template<typename VarType>
        double pvalue_impl(const VarType& v1, const VarType& v2, const VarType& cond) const;
        template<typename VarType, typename Iter>
        double pvalue_impl(const VarType& v1, const VarType& v2, Iter evidence_begin, Iter evidence_end) const;

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
    double LinearCorrelation::pvalue_cached(const VarType& v1, const VarType& v2, const VarType& cond) const {
        double cor = cor_1cond(m_cov, cached_index(v1), cached_index(v2), cached_index(cond));
        return cor_pvalue(cor, m_df->num_rows() - 3);
    }

    template<typename VarType>
    double LinearCorrelation::pvalue_impl(const VarType& v1, const VarType& v2, const VarType& cond) const {
        auto [cor, df] = [this, &v1, &v2, &cond]() {
            switch(m_df.col(v1)->type_id()) {
                case Type::DOUBLE: {
                    auto cov_ptr = m_df.cov<arrow::DoubleType>(v1, v2, cond);
                    auto& cov = *cov_ptr;
                    double cor = cor_1cond(cov, 0, 1, 2);
                    return std::make_pair(cor, m_df.valid_rows(v1, v2, cond) - 3);

                }
                case Type::FLOAT: {
                    auto cov_ptr = m_df.cov<arrow::FloatType>(v1, v2, cond);
                    auto& cov = *cov_ptr;
                    double cor = cor_1cond(cov, 0, 1, 2);
                    return std::make_pair(cor, m_df.valid_rows(v1, v2, cond) - 3);
                }
                default:
                    throw std::invalid_argument("Column " + m_df.name(v1) + " is not continuous");
            }
        }();

        return cor_pvalue(cor, df);
    }

    template<typename VarType, typename Iter>
    double LinearCorrelation::pvalue_cached(const VarType& v1, const VarType& v2, Iter evidence_begin, Iter evidence_end) const {
        std::vector<int> cached_indices;
   
        for (auto it = evidence_begin; it != evidence_end; ++it) {
            cached_indices.push_back(cached_index(*it));
        }

        cached_indices.push_back(cached_index(v1));
        cached_indices.push_back(cached_index(v2));

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

    template<typename VarType, typename Iter>
    double LinearCorrelation::pvalue_impl(const VarType& v1, const VarType& v2, Iter evidence_begin, Iter evidence_end) const {

        auto [cor, df] = [this, &v1, &v2, evidence_begin, evidence_end]() {
            int k = std::distance(evidence_begin, evidence_end);
            switch(m_df.col(v1)->type_id()) {
                case Type::DOUBLE: {
                    auto cov_ptr = m_df.cov<arrow::DoubleType>(std::make_pair(evidence_begin, evidence_end), v1, v2) ;
                    auto& cov = *cov_ptr;
                    double cor = cor_general(cov);
                    return std::make_pair(cor, 
                                m_df.valid_rows(v1, v2, std::make_pair(evidence_begin, evidence_end)) - 2 - k
                            );
                }
                case Type::FLOAT: {
                    auto cov_ptr = m_df.cov<arrow::FloatType>(std::make_pair(evidence_begin, evidence_end), v1, v2);
                    auto& cov = *cov_ptr;
                    double cor = cor_general(cov);
                    return std::make_pair(cor, 
                                m_df.valid_rows(v1, v2, std::make_pair(evidence_begin, evidence_end)) - 2 - k
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
