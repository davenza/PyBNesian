#ifndef PYBNESIAN_LEARNING_SCORES_BDE_HPP
#define PYBNESIAN_LEARNING_SCORES_BDE_HPP

#include <factors/discrete/DiscreteFactor.hpp>
#include <learning/scores/scores.hpp>

using factors::discrete::DiscreteFactorType;

namespace learning::scores {

class BDe : public Score {
public:
    BDe(const DataFrame& df, double iss = 1) : m_df(df), m_iss(iss) {}

    double local_score(const BayesianNetworkBase& model,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override;

    double local_score(const BayesianNetworkBase& model,
                       const std::shared_ptr<FactorType>& node_type,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override;

    std::string ToString() const override { return "BDe"; }

    bool has_variables(const std::string& name) const override { return m_df.has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const override { return m_df.has_columns(cols); }

    bool compatible_bn(const BayesianNetworkBase& model) const override {
        const auto& model_type = model.type_ref();
        return model_type.is_homogeneous() && *model_type.default_node_type() == DiscreteFactorType::get_ref() &&
               m_df.has_columns(model.nodes());
    }

    bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
        const auto& model_type = model.type_ref();
        return model_type.is_homogeneous() && *model_type.default_node_type() == DiscreteFactorType::get_ref() &&
               m_df.has_columns(model.joint_nodes());
    }

    DataFrame data() const override { return m_df; }

private:
    double bde_impl_noparents(const std::string& variable) const;
    double bde_impl_parents(const std::string& variable, const std::vector<std::string>& parents) const;

    const DataFrame m_df;
    double m_iss;
};

using DynamicBDe = DynamicScoreAdaptator<BDe>;

}  // namespace learning::scores

#endif  // PYBNESIAN_LEARNING_SCORES_BDE_HPP