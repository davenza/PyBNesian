#ifndef PYBNESIAN_LEARNING_SCORES_CV_LIKELIHOOD_HPP
#define PYBNESIAN_LEARNING_SCORES_CV_LIKELIHOOD_HPP

#include <dataset/crossvalidation_adaptator.hpp>
#include <learning/scores/scores.hpp>

using dataset::CrossValidation;
using learning::scores::Score, learning::scores::ScoreSPBN;
using factors::FactorType;
using models::BayesianNetworkBase, models::BayesianNetworkType, 
      models::SemiparametricBNBase;

namespace learning::scores {

    class CVLikelihood : public Score, public ScoreSPBN {
    public:
        CVLikelihood(const DataFrame& df, int k) : m_cv(df, k) {}
        CVLikelihood(const DataFrame& df, int k, unsigned int seed) : m_cv(df, k, seed) {}

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

        const CrossValidation& cv() { return m_cv; }

        std::string ToString() const override {
            return "CVLikelihood";
        }

        bool is_decomposable() const override {
            return true;
        }

        ScoreType type() const override {
            return ScoreType::PREDICTIVE_LIKELIHOOD;
        }

    private:
        CrossValidation m_cv;
    };

    template<typename VarType, typename EvidenceIter>
    double CVLikelihood::local_score(const BayesianNetworkBase& model,
                                     const VarType& variable, 
                                     const EvidenceIter evidence_begin,
                                     const EvidenceIter evidence_end) const {

        switch (model.type()) {
            case BayesianNetworkType::GBN: {
                LinearGaussianCPD cpd(m_cv.data().name(variable), m_cv.data().names(evidence_begin, evidence_end));
                double loglik = 0;
                for (auto [train_df, test_df] : m_cv.loc(variable, std::make_pair(evidence_begin, evidence_end))) {
                    cpd.fit(train_df);
                    loglik += cpd.slogl(test_df);
                }

                return loglik;
            }
            case BayesianNetworkType::SPBN: {
                const auto& spbn = dynamic_cast<const SemiparametricBNBase&>(model);
                FactorType variable_type = spbn.node_type(variable);
                return local_score(variable_type, variable, evidence_begin, evidence_end);   
            }
            default:
                throw std::invalid_argument("Bayesian network type " + model.type().ToString() 
                                                + " not valid for score CVLikelihood");
        }
    }

    template<typename VarType, typename EvidenceIter>
    double CVLikelihood::local_score(FactorType variable_type,
                                     const VarType& variable, 
                                     const EvidenceIter evidence_begin, 
                                     const EvidenceIter evidence_end) const {
        if (variable_type == FactorType::LinearGaussianCPD) {
            LinearGaussianCPD cpd(m_cv.data().name(variable), m_cv.data().names(evidence_begin, evidence_end));

            double loglik = 0;
            for (auto [train_df, test_df] : m_cv.loc(variable, std::make_pair(evidence_begin, evidence_end))) {
                cpd.fit(train_df);
                loglik += cpd.slogl(test_df);
            }

            return loglik;
        } else {
            CKDE cpd(m_cv.data().name(variable), m_cv.data().names(evidence_begin, evidence_end));

            double loglik = 0;
            for (auto [train_df, test_df] : m_cv.loc(variable, std::make_pair(evidence_begin, evidence_end))) {
                cpd.fit(train_df);
                loglik += cpd.slogl(test_df);
            }

            return loglik;
        }
    }
}

#endif //PYBNESIAN_CV_LIKELIHOOD_HPP