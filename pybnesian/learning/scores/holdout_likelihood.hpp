#ifndef PYBNESIAN_LEARNING_SCORES_HOLDOUT_LIKELIHOOD_HPP
#define PYBNESIAN_LEARNING_SCORES_HOLDOUT_LIKELIHOOD_HPP

#include <dataset/holdout_adaptator.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <learning/scores/scores.hpp>

using dataset::HoldOut;
using learning::scores::Score;
using models::GaussianNetwork, models::SemiparametricBN;

namespace learning::scores {

class HoldoutLikelihood : public Score {
public:
    HoldoutLikelihood(const DataFrame& df,
                      double test_ratio = 0.2,
                      unsigned int seed = std::random_device{}(),
                      Arguments construction_args = Arguments())
        : m_holdout(df, test_ratio, seed), m_arguments(construction_args) {}

    double local_score(const BayesianNetworkBase& model,
                       const std::string& variable,
                       const std::vector<std::string>& evidence) const override;

    double local_score(const BayesianNetworkBase& model,
                       const std::shared_ptr<FactorType>& variable_type,
                       const std::string& variable,
                       const std::vector<std::string>& evidence) const override;

    const DataFrame& training_data() const { return m_holdout.training_data(); }
    const DataFrame& test_data() const { return m_holdout.test_data(); }

    const HoldOut& holdout() { return m_holdout; }

    std::string ToString() const override { return "HoldoutLikelihood"; }

    bool has_variables(const std::string& name) const override { return m_holdout.training_data().has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const override {
        return m_holdout.training_data().has_columns(cols);
    }

    bool compatible_bn(const BayesianNetworkBase& model) const override {
        return m_holdout.training_data().has_columns(model.nodes());
    }

    bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
        return m_holdout.training_data().has_columns(model.joint_nodes());
    }

    DataFrame data() const override { return training_data(); }

private:
    template <typename FactorType>
    double factor_score(const std::string& variable, const std::vector<std::string>& evidence) const;

    HoldOut m_holdout;
    Arguments m_arguments;
};

template <typename FactorType>
double HoldoutLikelihood::factor_score(const std::string& variable, const std::vector<std::string>& evidence) const {
    FactorType cpd(variable, evidence);
    cpd.fit(training_data());
    return cpd.slogl(test_data());
}

using DynamicHoldoutLikelihood = DynamicScoreAdaptator<HoldoutLikelihood>;

}  // namespace learning::scores

#endif  // PYBNESIAN_LEARNING_SCORES_HOLDOUT_LIKELIHOOD_HPP
