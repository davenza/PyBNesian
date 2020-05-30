#ifndef PGM_DATASET_MLE_BASE_HPP
#define PGM_DATASET_MLE_BASE_HPP

#include <dataset/dataset.hpp>


using namespace dataset;

namespace learning::parameter {

    template<typename CPD>
    class MLE {
    public:
        template<typename VarType, typename EvidenceType>
        typename CPD::ParamsClass estimate(const DataFrame& df, const VarType& variable,  const EvidenceType& evidence) {
            return estimate(df, variable, evidence.begin(), evidence.end());
        }

        template<typename VarType, typename EvidenceIter>
        typename CPD::ParamsClass estimate(const DataFrame& df, const VarType& variable,  EvidenceIter evidence_begin, EvidenceIter evidence_end);
    };
}

#endif //PGM_DATASET_MLE_BASE_HPP
