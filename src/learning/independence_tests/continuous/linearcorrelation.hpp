#ifndef PGM_DATASET_LINEARCORRELATION_HPP
#define PGM_DATASET_LINEARCORRELATION_HPP

#include <dataset/dataset.hpp>

using dataset::DataFrame;

class LinearCorrelation {
public:
    LinearCorrelation(const DataFrame& df) : m_df(df) {}


    template<typename VarType, typename EvidenceIter>
    double pvalue(const VarType& v1, const VarType& v2, EvidenceIter evidence_begin, EvidenceIter evidence_end) const {
        
    }

private:
    const DataFrame m_df;
};

#endif //PGM_DATASET_LINEARCORRELATION_HPP