#ifndef PGM_DATASET_CV_LIKELIHOOD_HPP
#define PGM_DATASET_CV_LIKELIHOOD_HPP

#include <dataset/dataset.hpp>
#include <dataset/crossvalidation_adaptator.hpp>
#include <factors/factors.hpp>

using factors::FactorType;

namespace learning::scores {

    class CVLikelihood {
    public:

        CVLikelihood(const DataFrame& df, int k) : m_cv(df, k) {}
        CVLikelihood(const DataFrame& df, int k, int seed) : m_cv(df, k, seed) {}

        template<typename Model>
        double score(const Model& model) {
            double s = 0;
            for (auto node = 0; node < model.num_nodes(); ++node) {
                s += local_score(model, node);
            }
            
            return s;
        }

        template<typename Model, typename VarType, util::enable_if_gaussian_network_t<Model, int> = 0>
        double local_score(const Model& model, const VarType& variable) const {
            auto parents = model.parent_indices(variable);
            return local_score(model, variable, parents.begin(), parents.end());
        }
        
        template<typename Model, typename VarType, typename EvidenceIter, util::enable_if_gaussian_network_t<Model, int> = 0>
        double local_score(const Model& model, 
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

        template<typename Model, typename VarType, util::enable_if_semiparametricbn_t<Model, int> = 0>
        double local_score(const Model& model, const VarType& variable) const {
            auto parents = model.parent_indices(variable);
            FactorType variable_type = model.node_type(variable);
            
            return local_score(variable_type, variable, parents.begin(), parents.end());
        }

        template<typename Model, typename VarType, typename EvidenceIter, util::enable_if_semiparametricbn_t<Model, int> = 0>
        double local_score(const Model& model, 
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const {
            FactorType variable_type = model.node_type(variable);
            return local_score(variable_type, variable, evidence_begin, evidence_end);
        }

        template<typename VarType, typename EvidenceIter>
        double local_score(FactorType variable_type,
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

        const CrossValidation& cv() { return m_cv; }

    private:
        CrossValidation m_cv;
    };

    template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<util::is_gaussian_network_v<Model>, int> = 0>
    double CVLikelihood::local_score(const Model&,
                                     const VarType& variable, 
                                     const EvidenceIter evidence_begin,
                                     const EvidenceIter evidence_end) const {
        
        LinearGaussianCPD cpd(m_cv.data().name(variable), m_cv.data().names(evidence_begin, evidence_end));
        double loglik = 0;
        for (auto [train_df, test_df] : m_cv.loc(variable, std::make_pair(evidence_begin, evidence_end))) {
            cpd.fit(train_df);
            loglik += cpd.slogpdf(test_df);
        }

        return loglik;
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
                loglik += cpd.slogpdf(test_df);
            }

            return loglik;
        } else {
            CKDE cpd(m_cv.data().name(variable), m_cv.data().names(evidence_begin, evidence_end));

            double loglik = 0;
            for (auto [train_df, test_df] : m_cv.loc(variable, std::make_pair(evidence_begin, evidence_end))) {
                cpd.fit(train_df);
                loglik += cpd.slogpdf(test_df);
            }

            return loglik;
        }
    }
}

#endif //PGM_DATASET_CV_LIKELIHOOD_HPP