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
            return local_score<>(model, variable);
        }

        double local_score(const BayesianNetworkBase& model, const std::string& variable) const override {
            return local_score<>(model, variable);
        }

        double local_score(const BayesianNetworkBase& model, int variable,
                            const typename std::vector<int>::const_iterator evidence_begin, 
                            const typename std::vector<int>::const_iterator evidence_end) const override {
            return local_score<>(model, variable, evidence_begin, evidence_end);
        }

        double local_score(const BayesianNetworkBase& model, const std::string& variable,
                            const typename std::vector<std::string>::const_iterator evidence_begin, 
                            const typename std::vector<std::string>::const_iterator evidence_end) const override {
            return local_score<>(model, variable, evidence_begin, evidence_end);
        }


        template<typename VarType>
        double local_score(const BayesianNetworkBase& model, const VarType& variable) const {
            auto parents = model.parent_indices(variable);
            return local_score(model, variable, parents.begin(), parents.end());
        }
    
        template<typename VarType, typename EvidenceIter>
        double local_score(const BayesianNetworkBase& model, 
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

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
            return m_df->num_columns() == model.num_nodes() && m_df.has_columns(model.nodes());
        }

        bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
            return m_df->num_columns() == model.num_total_nodes() && m_df.has_columns(model.all_nodes());
        }
    private:
        const DataFrame m_df;        
    };

    template<typename VarType, typename EvidenceIter>
    double BIC::local_score(const BayesianNetworkBase& model,
                            const VarType& variable, 
                            const EvidenceIter evidence_begin,
                            const EvidenceIter evidence_end) const {
        switch (model.type()) {
            case BayesianNetworkType::GBN: {
                MLE<LinearGaussianCPD> mle;

                auto mle_params = mle.estimate(m_df, variable, evidence_begin, evidence_end);

                auto rows = m_df.valid_rows(variable, std::make_pair(evidence_begin, evidence_end));
                auto num_evidence = std::distance(evidence_begin, evidence_end);
                auto loglik = 0.5 * (1 + num_evidence - rows) 
                                - 0.5 * rows*std::log(2*util::pi<double>) 
                                - rows * std::log(std::sqrt(mle_params.variance));

                return loglik - std::log(rows) * 0.5 * (num_evidence + 2);
            }
            default:
               throw std::invalid_argument("Bayesian network type " + model.type().ToString() 
                                            + " not valid for score BIC");
        }
    }

    using DynamicBIC = DynamicScoreAdaptator<BIC>;
}

#endif //PYBNESIAN_LEARNING_SCORES_BIC_HPP