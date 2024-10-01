#ifndef PYBNESIAN_LEARNING_SPBN_SCORE_HPP
#define PYBNESIAN_LEARNING_SPBN_SCORE_HPP

#include <learning/scores/scores.hpp>
#include <learning/scores/holdout_likelihood.hpp>
#include <learning/scores/cv_likelihood.hpp>

using learning::scores::ValidatedScore, learning::scores::HoldoutLikelihood, learning::scores::CVLikelihood;

namespace learning::scores {
/**
 * @brief This class mixes the functionality of CVLikelihood and HoldoutLikelihood. First, it applies a HoldOut split
 over the data. Then:
    - It estimates the training score using a CVLikelihood over the training data.
    - It estimates the validation score using the training data to estimate the parameters and calculating the
 log-likelihood on the holdout data.

 *
 */
class ValidatedLikelihood : public ValidatedScore {
public:
    ValidatedLikelihood(const DataFrame& df,
                        double test_ratio = 0.2,
                        int k = 10,
                        unsigned int seed = std::random_device{}(),
                        Arguments construction_args = Arguments())
        : m_holdout(df, test_ratio, seed, construction_args),
          m_cv(m_holdout.training_data(), k, seed, construction_args) {}

    using ValidatedScore::local_score;

    double local_score(const BayesianNetworkBase& model,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override {
        return m_cv.local_score(model, variable, parents);
    }
    /**
     * @brief Calculates the cross-validated log-likelihood of a variable given its parents.
     *
     * @param model
     * @param variable_type
     * @param variable
     * @param parents
     * @return double
     */
    double local_score(const BayesianNetworkBase& model,
                       const std::shared_ptr<FactorType>& variable_type,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override {
        return m_cv.local_score(model, variable_type, variable, parents);
    }

    std::string ToString() const override { return "ValidatedLikelihood"; }

    bool has_variables(const std::string& name) const override { return m_cv.has_variables(name); }

    bool has_variables(const std::vector<std::string>& cols) const override { return m_cv.has_variables(cols); }

    bool compatible_bn(const BayesianNetworkBase& model) const override { return m_cv.has_variables(model.nodes()); }

    bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
        return m_cv.has_variables(model.joint_nodes());
    }

    double vlocal_score(const BayesianNetworkBase& model,
                        const std::string& variable,
                        const std::vector<std::string>& evidence) const override {
        return m_holdout.local_score(model, variable, evidence);
    }
    /**
     * @brief Calculates the validated local score of a variable given the evidence.
     *
     * @param model BayesianNetworkBase
     * @param variable_type FactorType
     * @param variable the variable name
     * @param evidence the evidence vector
     * @return double the validated local score
     */
    double vlocal_score(const BayesianNetworkBase& model,
                        const std::shared_ptr<FactorType>& variable_type,
                        const std::string& variable,
                        const std::vector<std::string>& evidence) const override {
        return m_holdout.local_score(model, variable_type, variable, evidence);
    }

    const DataFrame& training_data() { return m_holdout.training_data(); }

    const DataFrame& validation_data() { return m_holdout.test_data(); }

    const HoldoutLikelihood& holdout() const { return m_holdout; }

    const CVLikelihood& cv() const { return m_cv; }

    DataFrame data() const override { return m_cv.data(); }

private:
    HoldoutLikelihood m_holdout;
    CVLikelihood m_cv;
};

using DynamicValidatedLikelihood = DynamicScoreAdaptator<ValidatedLikelihood>;

}  // namespace learning::scores

#endif  // PYBNESIAN_LEARNING_SPBN_SCORE_HPP
