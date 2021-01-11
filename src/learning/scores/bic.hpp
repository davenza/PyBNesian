#ifndef PYBNESIAN_LEARNING_SCORES_BIC_HPP
#define PYBNESIAN_LEARNING_SCORES_BIC_HPP

#include <learning/scores/scores.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <learning/parameters/mle_LinearGaussianCPD.hpp>

using learning::scores::ScoreType, learning::scores::Score;
using factors::continuous::LinearGaussianCPD;
using namespace dataset;
using models::BayesianNetworkBase, models::BayesianNetworkType, models::GaussianNetwork;
using learning::parameters::MLE;

namespace learning::scores {

    class BIC : public Score {
    public:
        BIC(const DataFrame& df) : m_df(df) {}

        double local_score(const BayesianNetworkBase& model, int variable) const override {
            return local_score(model, model.name(variable));
        }

        double local_score(const BayesianNetworkBase& model, const std::string& variable) const override {
            auto parents = model.parents(variable);
            return local_score(model, variable, parents);
        }

        double local_score(const BayesianNetworkBase& model, int variable, const std::vector<int>& evidence) const override {
            std::vector<std::string> evidence_str;
            for (auto ev : evidence) {
                evidence_str.push_back(model.name(ev));
            }

            return local_score(model, model.name(variable), evidence_str);
        }

        double local_score(const BayesianNetworkBase& model,
                           const std::string& variable,
                           const std::vector<std::string>& evidence) const override;

        std::string ToString() const override {
            return "BIC";
        }

        bool is_decomposable() const override {
            return true;
        }

        ScoreType type() const override {
            return ScoreType::BIC;
        }

        bool compatible_bn(const BayesianNetworkBase& model) const override{
            return m_df.has_columns(model.nodes());
        }

        bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
            return m_df.has_columns(model.all_nodes());
        }
    private:
        const DataFrame m_df;        
    };

    using DynamicBIC = DynamicScoreAdaptator<BIC>;
}

#endif //PYBNESIAN_LEARNING_SCORES_BIC_HPP