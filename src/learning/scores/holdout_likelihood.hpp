#ifndef PYBNESIAN_LEARNING_SCORES_HOLDOUT_LIKELIHOOD_HPP
#define PYBNESIAN_LEARNING_SCORES_HOLDOUT_LIKELIHOOD_HPP

#include <dataset/holdout_adaptator.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <learning/scores/scores.hpp>

using dataset::HoldOut;
using models::GaussianNetwork, models::SemiparametricBN;
using learning::scores::ScoreSPBN;

namespace learning::scores {

    class HoldoutLikelihood : public Score, public ScoreSPBN {
    public:
        HoldoutLikelihood(const DataFrame& df, double test_ratio) : m_holdout(df, test_ratio) { }
        HoldoutLikelihood(const DataFrame& df, double test_ratio, long unsigned int seed) : m_holdout(df, test_ratio, seed) { }

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

        double local_score(FactorType variable_type, int variable, 
                            const typename std::vector<int>::const_iterator evidence_begin, 
                            const typename std::vector<int>::const_iterator evidence_end) const override {
            return local_score<>(variable_type, variable, evidence_begin, evidence_end);
        }

        double local_score(FactorType variable_type, const std::string& variable, 
                            const typename std::vector<std::string>::const_iterator evidence_begin, 
                            const typename std::vector<std::string>::const_iterator evidence_end) const override {
            return local_score<>(variable_type, variable, evidence_begin, evidence_end);
        }

        template<typename VarType, typename EvidenceIter>
        double local_score(FactorType variable_type, 
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

        const DataFrame& training_data() const { return m_holdout.training_data(); }
        const DataFrame& test_data() const { return m_holdout.test_data(); }

        const HoldOut& holdout() { return m_holdout; }

        std::string ToString() const override {
            return "HoldoutLikelihood";
        }

        bool is_decomposable() const override {
            return true;
        }

        ScoreType type() const override {
            return ScoreType::HOLDOUT_LIKELIHOOD;
        }
    private:
        HoldOut m_holdout;
    };

    template<typename VarType, typename EvidenceIter>
    double HoldoutLikelihood::local_score(const BayesianNetworkBase& model,
                                          const VarType& variable, 
                                          const EvidenceIter evidence_begin, 
                                          const EvidenceIter evidence_end) const {
        switch (model.type()) {
            case BayesianNetworkType::GBN: {
                LinearGaussianCPD cpd(training_data().name(variable), training_data().names(evidence_begin, evidence_end));
                cpd.fit(training_data());
                return cpd.slogl(test_data());
            }
            case BayesianNetworkType::SPBN: {
                const auto& spbn = dynamic_cast<const SemiparametricBNBase&>(model);
                FactorType variable_type = spbn.node_type(variable);
                return local_score(variable_type, variable, evidence_begin, evidence_end);   
            }
            default:
                throw std::invalid_argument("Bayesian network type " + model.type().ToString() 
                                                + " not valid for score HoldoutLikelihood");
        }
        

    }

    template<typename VarType, typename EvidenceIter>
    double HoldoutLikelihood::local_score(FactorType variable_type,
                                          const VarType& variable,
                                          const EvidenceIter evidence_begin, 
                                          const EvidenceIter evidence_end) const {
        if (variable_type == FactorType::LinearGaussianCPD) {
            LinearGaussianCPD cpd(training_data().name(variable), training_data().names(evidence_begin, evidence_end));
            cpd.fit(training_data());
            return cpd.slogl(test_data());
        } else {
            CKDE cpd(training_data().name(variable), training_data().names(evidence_begin, evidence_end));
            cpd.fit(training_data());
            return cpd.slogl(test_data());
        }
    }
}

#endif //PYBNESIAN_LEARNING_SCORES_HOLDOUT_LIKELIHOOD_HPP