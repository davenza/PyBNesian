#include <learning/independences/continuous/linearcorrelation.hpp>
#include <boost/math/distributions/students_t.hpp>

using boost::math::cdf, boost::math::complement;
using boost::math::students_t_distribution;

namespace learning::independences::continuous {

double cor_pvalue(double cor, int df) {
    double statistic = cor * sqrt(df) / sqrt(1 - cor * cor);
    students_t_distribution tdist(static_cast<double>(df));
    return 2 * cdf(complement(tdist, fabs(statistic)));
}

double LinearCorrelation::pvalue_cached(const std::string& v1, const std::string& v2) const {
    double cor = cor_0cond(m_cov, cached_index(v1), cached_index(v2));
    return cor_pvalue(cor, m_df->num_rows() - 2);
}

double LinearCorrelation::pvalue_impl(const std::string& v1, const std::string& v2) const {
    auto [cor, df] = [this, &v1, &v2]() {
        switch (m_df.col(v1)->type_id()) {
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

double LinearCorrelation::pvalue_cached(const std::string& v1, const std::string& v2, const std::string& ev) const {
    double cor = cor_1cond(m_cov, cached_index(v1), cached_index(v2), cached_index(ev));
    return cor_pvalue(cor, m_df->num_rows() - 3);
}

double LinearCorrelation::pvalue_impl(const std::string& v1, const std::string& v2, const std::string& ev) const {
    auto [cor, df] = [this, &v1, &v2, &ev]() {
        switch (m_df.col(v1)->type_id()) {
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

double LinearCorrelation::pvalue_cached(const std::string& v1,
                                        const std::string& v2,
                                        const std::vector<std::string>& ev) const {
    std::vector<int> cached_indices;

    cached_indices.push_back(cached_index(v1));
    cached_indices.push_back(cached_index(v2));

    for (auto it = ev.begin(), end = ev.end(); it != end; ++it) {
        cached_indices.push_back(cached_index(*it));
    }

    int k = cached_indices.size();
    MatrixXd cov(k, k);

    for (int i = 0; i < k; ++i) {
        cov(i, i) = m_cov(cached_indices[i], cached_indices[i]);
        for (int j = i + 1; j < k; ++j) {
            cov(i, j) = cov(j, i) = m_cov(cached_indices[i], cached_indices[j]);
        }
    }

    double cor = cor_general(cov);
    return cor_pvalue(cor, m_df->num_rows() - 2 - k);
}

double LinearCorrelation::pvalue_impl(const std::string& v1,
                                      const std::string& v2,
                                      const std::vector<std::string>& ev) const {
    auto [cor, df] = [this, &v1, &v2, &ev]() {
        int k = ev.size();
        switch (m_df.col(v1)->type_id()) {
            case Type::DOUBLE: {
                auto cov_ptr = m_df.cov<arrow::DoubleType>(v1, v2, ev);
                auto& cov = *cov_ptr;
                double cor = cor_general(cov);
                return std::make_pair(cor, m_df.valid_rows(v1, v2, ev) - 2 - k);
            }
            case Type::FLOAT: {
                auto cov_ptr = m_df.cov<arrow::FloatType>(v1, v2, ev);
                auto& cov = *cov_ptr;
                double cor = cor_general(cov);
                return std::make_pair(cor, m_df.valid_rows(v1, v2, ev) - 2 - k);
            }
            default:
                throw std::invalid_argument("Column " + m_df.name(v1) + " is not continuous");
        }
    }();

    return cor_pvalue(cor, df);
}

}  // namespace learning::independences::continuous