#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MUTUAL_INFORMATION_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MUTUAL_INFORMATION_HPP

#include <dataset/dataset.hpp>
#include <learning/independences/independence.hpp>

using dataset::DataFrame;
using learning::independences::IndependenceTest;

namespace learning::independences::hybrid {

class MutualInformation : public IndependenceTest {
public:
    MutualInformation(const DataFrame& df, bool asymptotic_df = true) : m_df(df), m_asymptotic_df(asymptotic_df) {
        for (int i = 0; i < m_df->num_columns(); ++i) {
            if (!m_df.is_discrete(i) && !m_df.is_continuous(i))
                throw std::invalid_argument("Wrong data type (" + m_df.col(i)->type()->ToString() + ") for column " +
                                            m_df.name(i) + ".");
        }
    }

    double pvalue(const std::string& x, const std::string& y) const override;
    double pvalue(const std::string& x, const std::string& y, const std::string& z) const override;
    double pvalue(const std::string& x, const std::string& y, const std::vector<std::string>& z) const override;

    double mi(const std::string& x, const std::string& y) const;
    double mi(const std::string& x, const std::string& y, const std::string& z) const;
    double mi(const std::string& x, const std::string& y, const std::vector<std::string>& z) const;

    int num_variables() const override { return m_df->num_columns(); }
    std::vector<std::string> variable_names() const override { return m_df.column_names(); }
    const std::string& name(int i) const override { return m_df.name(i); }
    bool has_variables(const std::string& name) const override { return m_df.has_columns(name); }
    bool has_variables(const std::vector<std::string>& cols) const override { return m_df.has_columns(cols); }

private:
    double mi_discrete(const std::string& x, const std::string& y) const;
    template <bool contains_null, typename IndicesArrowType, typename ContinuousArrowType>
    double mi_mixed_impl(const std::string& discrete, const std::string& continuous) const;
    double mi_mixed(const std::string& x, const std::string& y) const;
    template <typename ArrowType>
    double mi_continuous_impl(const std::string& x, const std::string& y) const;
    double mi_continuous(const std::string& x, const std::string& y) const;

    double cmi_discrete_discrete(const std::string& x,
                                 const std::string& y,
                                 const std::vector<std::string>& discrete_z) const;
    template <bool contains_null, typename IndicesArrowType, typename ContinuousArrowType>
    double cmi_discrete_continuous_impl(const std::string& x, const std::string& y, const std::string& z) const;
    double cmi_discrete_continuous(const std::string& x, const std::string& y, const std::string& z) const;

    double cmi_general_both_discrete(const std::string& x,
                                     const std::string& y,
                                     const std::vector<std::string>& discrete_z,
                                     const std::vector<std::string>& continuous_z) const;

    double cmi_general_mixed(const std::string& x_discrete,
                             const std::string& y_continuous,
                             const std::vector<std::string>& discrete_z,
                             const std::vector<std::string>& continuous_z) const;

    double cmi_general_both_continuous(const std::string& x,
                                       const std::string& y,
                                       const std::vector<std::string>& discrete_z,
                                       const std::vector<std::string>& continuous_z) const;

    double cmi_general(const std::string& x,
                       const std::string& y,
                       const std::vector<std::string>& discrete_z,
                       const std::vector<std::string>& continuous_z) const;

    double discrete_df(const std::string& x, const std::string& y) const;
    double discrete_df(const std::string& x, const std::string& y, const std::string& z) const;
    double discrete_df(const std::string& x,
                       const std::string& y,
                       const std::vector<std::string>& discrete_z,
                       const std::vector<std::string>& continuous_z) const;
    double mixed_df(const std::string& x, const std::string& y) const;
    double mixed_df(const std::string& discrete, const std::string& continuous, const std::string& z) const;
    double mixed_df(const std::string& discrete,
                    const std::string& continuous,
                    const std::vector<std::string>& discrete_z,
                    const std::vector<std::string>& continuous_z) const;
    double continuous_df(const std::string&, const std::string&, const std::string& z) const;
    double continuous_df(const std::string&,
                         const std::string&,
                         const std::vector<std::string>& discrete_z,
                         const std::vector<std::string>& continuous_z) const;
    double calculate_df(const std::string& x, const std::string& y) const;
    double calculate_df(const std::string& x, const std::string& y, const std::string& z) const;
    double calculate_df(const std::string& x,
                        const std::string& y,
                        const std::vector<std::string>& discrete_z,
                        const std::vector<std::string>& continuous_z) const;

    DataFrame m_df;
    bool m_asymptotic_df;
};

using DynamicMutualInformation = DynamicIndependenceTestAdaptator<MutualInformation>;

}  // namespace learning::independences::hybrid

#endif  // PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MUTUAL_INFORMATION_HPP
