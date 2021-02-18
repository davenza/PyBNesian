#ifndef PYBNESIAN_LEARNING_SCORES_CV_LIKELIHOOD_HPP
#define PYBNESIAN_LEARNING_SCORES_CV_LIKELIHOOD_HPP

#include <dataset/crossvalidation_adaptator.hpp>
#include <learning/scores/scores.hpp>

using dataset::CrossValidation;
using learning::scores::Score, learning::scores::ScoreSPBN;
using factors::NodeType;
using models::BayesianNetworkBase, models::BayesianNetworkType, 
      models::SemiparametricBNBase;

namespace learning::scores {

    class CVLikelihood : public ScoreSPBN {
    public:
        CVLikelihood(const DataFrame& df, int k) : m_cv(df, k) {}
        CVLikelihood(const DataFrame& df, int k, unsigned int seed) : m_cv(df, k, seed) {}

        double local_score(const BayesianNetworkBase& model, int variable) const override {
            return local_score(model, model.name(variable));
        }

        double local_score(const BayesianNetworkBase& model, const std::string& variable) const override {
            auto parents = model.parents(variable);
            return local_score(model, variable, parents);
        }

        double local_score(const BayesianNetworkBase& model,
                           int variable,
                           const std::vector<int>& evidence) const override {
            std::vector<std::string> evidence_str;
            for (auto ev : evidence) {
                evidence_str.push_back(model.name(ev));
            }

            return local_score(model, model.name(variable), evidence_str);
        }

        double local_score(const BayesianNetworkBase& model,
                           const std::string& variable,
                           const std::vector<std::string>& evidence) const override;

        double local_score(NodeType variable_type,
                           const std::string& variable,
                           const std::vector<std::string>& evidence) const override;

        const CrossValidation& cv() { return m_cv; }

        std::string ToString() const override {
            return "CVLikelihood";
        }

        bool is_decomposable() const override {
            return true;
        }

        ScoreType type() const override {
            return ScoreType::CVLikelihood;
        }

        bool has_variables(const std::string& name) const override {
            return m_cv.data().has_columns(name);
        }

        bool has_variables(const std::vector<std::string>& cols) const override {
            return m_cv.data().has_columns(cols);
        }

        bool compatible_bn(const BayesianNetworkBase& model) const override{
            return m_cv.data().has_columns(model.nodes());
        }

        bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
            return m_cv.data().has_columns(model.all_nodes());
        }
    private:
        template<typename FactorType>
        double factor_score(const std::string& variable, const std::vector<std::string>& evidence) const;

        CrossValidation m_cv;
    };

    template<typename FactorType>
    double CVLikelihood::factor_score(const std::string& variable, const std::vector<std::string>& evidence) const {
        FactorType cpd(variable, evidence);
        double loglik = 0;
        for (auto [train_df, test_df] : m_cv.loc(variable, evidence)) {
            cpd.fit(train_df);
            loglik += cpd.slogl(test_df);
        }

        return loglik;
    }

    using DynamicCVLikelihood = DynamicScoreAdaptator<CVLikelihood>;
}

#endif //PYBNESIAN_CV_LIKELIHOOD_HPP
