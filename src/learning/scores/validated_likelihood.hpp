#ifndef PYBNESIAN_LEARNING_SPBN_SCORE_HPP
#define PYBNESIAN_LEARNING_SPBN_SCORE_HPP

#include <learning/scores/scores.hpp>

using learning::scores::ValidatedScoreSPBN;

namespace learning::scores {

    class ValidatedLikelihood : public ValidatedScoreSPBN {
    public:
        ValidatedLikelihood(const DataFrame& df, double test_ratio, int k) 
            : ValidatedLikelihood(df, test_ratio, k, std::random_device{}()) {}
        ValidatedLikelihood(const DataFrame& df, double test_ratio, int k, unsigned int seed) 
            : m_holdout(df, test_ratio, seed),
              m_cv(m_holdout.training_data(), k, seed) {}

        
        double local_score(const BayesianNetworkBase& model, int variable) const override {
            return m_cv.local_score(model, variable);
        }

        double local_score(const BayesianNetworkBase& model, const std::string& variable) const override {
            return m_cv.local_score(model, variable);
        }

        double local_score(const BayesianNetworkBase& model,
                           int variable,
                           const std::vector<int>& evidence) const override {
            return m_cv.local_score(model, variable, evidence);
        }

        double local_score(const BayesianNetworkBase& model,
                           const std::string& variable,
                           const std::vector<std::string>& evidence) const override {
            return m_cv.local_score(model, variable, evidence);
        }

        double local_score(NodeType variable_type,
                           const std::string& variable,
                           const std::vector<std::string>& evidence) const override {
            return m_cv.local_score(variable_type, variable, evidence);
        }

        std::string ToString() const override {
            return "ValidatedLikelihood";
        }
        
        bool is_decomposable() const override {
            return true;
        }

        ScoreType type() const override {
            return ScoreType::ValidatedLikelihood;
        }

        bool has_variables(const std::string& name) const override {
            return m_cv.has_variables(name);
        }

        bool has_variables(const std::vector<std::string>& cols) const override {
            return m_cv.has_variables(cols);
        }

        bool compatible_bn(const BayesianNetworkBase& model) const override {
            return m_cv.has_variables(model.nodes());
        }

        bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
            return m_cv.has_variables(model.all_nodes());
        }

        double vlocal_score(const BayesianNetworkBase& model, int variable) const override {
            return m_holdout.local_score(model, variable);
        }

        double vlocal_score(const BayesianNetworkBase& model, const std::string& variable) const override {
            return m_holdout.local_score(model, variable);
        }

        double vlocal_score(const BayesianNetworkBase& model,
                            int variable,
                            const std::vector<int>& evidence) const override {
            return m_holdout.local_score(model, variable, evidence);
        }

        double vlocal_score(const BayesianNetworkBase& model,
                            const std::string& variable,
                            const std::vector<std::string>& evidence) const override {
            return m_holdout.local_score(model, variable, evidence);
        }

        double vlocal_score(NodeType variable_type,
                            const std::string& variable,
                            const std::vector<std::string>& evidence) const override {
            return m_holdout.local_score(variable_type, variable, evidence);
        }

        const DataFrame& training_data() {
            return m_holdout.training_data();
        }

        const DataFrame& validation_data() {
            return m_holdout.test_data();
        }

        const HoldoutLikelihood& holdout() const {
            return m_holdout;
        }

        const CVLikelihood& cv() const {
            return m_cv;
        }
    private:
        HoldoutLikelihood m_holdout;
        CVLikelihood m_cv;
    };

    using DynamicValidatedLikelihood = DynamicScoreAdaptator<ValidatedLikelihood>;
}

#endif //PYBNESIAN_LEARNING_SPBN_SCORE_HPP
