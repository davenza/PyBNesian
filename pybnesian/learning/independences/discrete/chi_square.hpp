#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_DISCRETE_CHI_SQUARE_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_DISCRETE_CHI_SQUARE_HPP

#include <learning/independences/independence.hpp>

namespace learning::independences::discrete {

class ChiSquare : public IndependenceTest {
public:
    ChiSquare(const DataFrame& df) : m_df(df) {
        auto discrete_indices = df.discrete_columns();

        if (discrete_indices.size() < 2) {
            throw std::invalid_argument("DataFrame does not contain enough categorical columns.");
        }
    }

    double pvalue(const std::string& v1, const std::string& v2) const override;
    double pvalue(const std::string& v1, const std::string& v2, const std::string& ev) const override;
    double pvalue(const std::string& v1, const std::string& v2, const std::vector<std::string>& ev) const override;

    int num_variables() const override { return m_df->num_columns(); }
    std::vector<std::string> variable_names() const override { return m_df.column_names(); }
    const std::string& name(int i) const override { return m_df.name(i); }
    bool has_variables(const std::string& name) const override { return m_df.has_columns(name); }
    bool has_variables(const std::vector<std::string>& cols) const override { return m_df.has_columns(cols); }

private:
    const DataFrame m_df;
};

using DynamicChiSquare = DynamicIndependenceTestAdaptator<ChiSquare>;

}  // namespace learning::independences::discrete

#endif  // PYBNESIAN_LEARNING_INDEPENDENCES_DISCRETE_CHI_SQUARE_HPP