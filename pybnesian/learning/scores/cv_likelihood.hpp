#ifndef PYBNESIAN_LEARNING_SCORES_CV_LIKELIHOOD_HPP
#define PYBNESIAN_LEARNING_SCORES_CV_LIKELIHOOD_HPP

#include <dataset/crossvalidation_adaptator.hpp>
#include <learning/scores/scores.hpp>

using dataset::CrossValidation;
using factors::FactorType;
using learning::scores::Score;
using models::BayesianNetworkBase, models::BayesianNetworkType;

namespace learning::scores {

class CVLikelihood : public Score {
public:
    CVLikelihood(const DataFrame& df,
                 int k = 10,
                 unsigned int seed = std::random_device{}(),
                 Arguments construction_args = Arguments())
        : m_cv(df, k, seed), m_arguments(construction_args) {}

    double local_score(const BayesianNetworkBase& model,
                       const std::string& variable,
                       const std::vector<std::string>& evidence) const override;

    double local_score(const BayesianNetworkBase& model,
                       const std::shared_ptr<FactorType>& variable_type,
                       const std::string& variable,
                       const std::vector<std::string>& evidence) const override;

    const CrossValidation& cv() { return m_cv; }

    std::string ToString() const override { return "CVLikelihood"; }

    bool has_variables(const std::string& name) const override { return m_cv.data().has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const override { return m_cv.data().has_columns(cols); }

    bool compatible_bn(const BayesianNetworkBase& model) const override {
        return m_cv.data().has_columns(model.nodes());
    }

    bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
        return m_cv.data().has_columns(model.joint_nodes());
    }

    DataFrame data() const override { return m_cv.data(); }

private:
    template <typename FactorType>
    double factor_score(const std::string& variable, const std::vector<std::string>& evidence) const;

    CrossValidation m_cv;
    Arguments m_arguments;
};

using DynamicCVLikelihood = DynamicScoreAdaptator<CVLikelihood>;

}  // namespace learning::scores

#endif  // PYBNESIAN_CV_LIKELIHOOD_HPP
