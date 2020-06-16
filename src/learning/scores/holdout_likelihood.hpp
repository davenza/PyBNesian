#ifndef PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP
#define PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP

#include <dataset/dataset.hpp>

namespace learning::scores {

    class HoldoutLikelihood {
    public:
        inline static constexpr bool is_decomposable = true;

        HoldoutLikelihood(const DataFrame& df, double test_ratio, int seed = std::random_device{}()) : m_holdout(df, test_ratio, seed) { }


        template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<util::is_gaussian_network_v<Model>, int> = 0>
        double local_score(const Model& model, const VarType& variable, const EvidenceIter evidence_begin, const EvidenceIter evidence_end) const;


    private:
        HoldOut m_holdout;
    }

    template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<util::is_gaussian_network_v<Model>, int> = 0>
    double HoldoutLikelihood::local_score(const Model&,
                                        const VarType& variable, 
                                        const EvidenceIter evidence_begin,
                                        const EvidenceIter evidence_end) const {
        LinearGaussianCPD cpd(m_cv.data().name(variable), m_cv.data().names(evidence_begin, evidence_end));
        cpd.fit(m_hold_out.training_data());
        return cpd.slogpdf(m_hold_out.test_data());
    }
}




#endif //PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP