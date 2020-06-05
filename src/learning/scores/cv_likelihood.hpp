#ifndef PGM_DATASET_CV_LIKELIHOOD_HPP
#define PGM_DATASET_CV_LIKELIHOOD_HPP

#include <dataset/dataset.hpp>
#include <dataset/crossvalidation_adaptator.hpp>

namespace learning::scores {

    class CVLikelihood {
    public:
        inline static constexpr bool is_decomposable = true;

        CVLikelihood(const DataFrame& df, int k) : m_cv(df, k) {}
        CVLikelihood(const DataFrame& df, int k, int seed) : m_cv(df, k, seed) {}

        template<typename Model>
        double score(const Model& model);

        template<typename Model, typename VarType, typename EvidenceType, std::enable_if_t<is_decomposable, int> = 0>
        double local_score(const Model& model,
                            const VarType& variable, 
                            const EvidenceType& evidence) const
        {
            return local_score(model, variable, evidence.begin(), evidence.end());
        }

        template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<util::is_gaussian_network_v<Model>, int> = 0>
        double local_score(const Model& model, const VarType& variable, const EvidenceIter evidence_begin, const EvidenceIter evidence_end) const;

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
}

#endif //PGM_DATASET_CV_LIKELIHOOD_HPP