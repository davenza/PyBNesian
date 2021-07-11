#ifndef PYBNESIAN_LEARNING_SCORES_BIC_HPP
#define PYBNESIAN_LEARNING_SCORES_BIC_HPP

#include <learning/scores/scores.hpp>
#include <learning/parameters/mle_LinearGaussianCPD.hpp>

using learning::scores::Score;
using namespace dataset;
using learning::parameters::MLE;
using models::BayesianNetworkBase, models::BayesianNetworkType, models::GaussianNetworkType, models::GaussianNetwork;

namespace learning::scores {

class BIC : public Score {
public:
    BIC(const DataFrame& df) : m_df(df) {}

    double local_score(const BayesianNetworkBase& model,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override;

    double local_score(const BayesianNetworkBase& model,
                       const std::shared_ptr<FactorType>& node_type,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override;

    std::string ToString() const override { return "BIC"; }

    bool has_variables(const std::string& name) const override { return m_df.has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const override { return m_df.has_columns(cols); }

    bool compatible_bn(const BayesianNetworkBase& model) const override { return m_df.has_columns(model.nodes()); }

    bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
        return m_df.has_columns(model.joint_nodes());
    }

    DataFrame data() const override { return m_df; }

private:
    double bic_lineargaussian(const std::string& variable, const std::vector<std::string>& parents) const;
    double bic_discrete(const std::string& variable, const std::vector<std::string>& parents) const;
    double bic_clg(const std::string& variable,
                   const std::vector<std::string>& discrete_parents,
                   const std::vector<std::string>& continuous_parents) const;

    bool are_all_discrete(const BayesianNetworkBase& model, const std::vector<std::string>& vars) const;

    const DataFrame m_df;
};

using DynamicBIC = DynamicScoreAdaptator<BIC>;

}  // namespace learning::scores

#endif  // PYBNESIAN_LEARNING_SCORES_BIC_HPP
