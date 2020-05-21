#ifndef PGM_DATASET_SCORES_HPP
#define PGM_DATASET_SCORES_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/parameter/mle.hpp>
#include <util/math_constants.hpp>

using factors::continuous::LinearGaussianCPD;
using namespace dataset;
using models::GaussianNetwork;
using learning::parameter::MLE;

namespace learning::scores {

    // template<typename Score>
    // struct score_traits;


    template<typename Model>
    class BIC {
    public:
        inline static constexpr bool is_decomposable = true;

        static double score(const DataFrame& df, Model& model);

        template<typename VarType, typename EvidenceType, std::enable_if_t<is_decomposable, int> = 0>
        static double local_score(const DataFrame& df, 
                                  const VarType& variable, 
                                  const EvidenceType& evidence) 
        {
            return local_score(df, variable, evidence.begin(), evidence.end());
        }


        template<typename VarType, typename EvidenceIter, std::enable_if_t<is_decomposable, int> = 0>
        static double local_score(const DataFrame& df, 
                                  const VarType& variable, 
                                  EvidenceIter evidence_begin,
                                  EvidenceIter evidence_end);
      
        
    };

    template<>
    template<typename VarType, typename EvidenceIter, std::enable_if_t<BIC<GaussianNetwork>::is_decomposable, int> = 0>
    double BIC<GaussianNetwork>::local_score(const DataFrame& df, 
                                             const VarType& variable, 
                                             EvidenceIter evidence_begin,
                                             EvidenceIter evidence_end) {

        MLE<LinearGaussianCPD> mle;

        auto mle_params = mle.estimate(df, variable, evidence_begin, evidence_end);

        auto rows = df->num_rows();
        auto loglik = (1-rows) / 2 - (rows / 2)*std::log(2*util::pi<double>) - rows * std::log(std::sqrt(mle_params.variance));

        return loglik - std::log(rows) * 0.5 * (std::distance(evidence_begin, evidence_end) + 2);
    }

}

#endif //PGM_DATASET_SCORES_HPP