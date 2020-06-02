#include <dataset/dataset.hpp>
#include <dataset/crossvalidation_adaptator.hpp>

namespace learning::scores {

    class CVLikelihood {
    public:
        inline static constexpr bool is_decomposable = true;

        CVLikelihood(DataFrame& df, int k) : m_df(df), m_cv(k) {}
        CVLikelihood(DataFrame& df, int k, int seed) : m_df(df), m_cv(k, seed) {}

        template<typename Model>
        double score(const Model& model);

        template<typename Model, typename VarType, typename EvidenceType, std::enable_if_t<is_decomposable, int> = 0>
        double local_score(const Model& model,
                            const VarType& variable, 
                            const EvidenceType& evidence) const
        {
            return local_score(model, variable, evidence.begin(), evidence.end());
        }


        template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<is_decomposable, int> = 0>
        double local_score(const Model& model,
                            const VarType& variable, 
                            EvidenceIter evidence_begin,
                                    EvidenceIter evidence_end) const;

    private:
        DataFrame& m_df;
        CrossValidation m_cv;
    };


    template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<BIC::is_decomposable, int> = 0>
    double CVLikelihood::local_score(const Model&,
                                        const VarType& variable, 
                                        EvidenceIter evidence_begin,
                                        EvidenceIter evidence_end) const {

        MLE<LinearGaussianCPD> mle;


    }
}
