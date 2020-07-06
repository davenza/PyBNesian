#ifndef PGM_DATASET_MLE_BASE_HPP
#define PGM_DATASET_MLE_BASE_HPP

#include <dataset/dataset.hpp>

using namespace dataset;

#include <models/SemiparametricBN_NodeType.hpp>

using models::NodeType;

#include <factors/continuous/LinearGaussianCPD.hpp>

using factors::continuous::LinearGaussianCPD;

namespace learning::parameters {

    template<typename CPD>
    class MLE {
    public:
        // template<typename VarType, typename EvidenceType>
        // typename CPD::ParamsClass estimate(const DataFrame& df, const VarType& variable,  const EvidenceType& evidence) {
        //     return estimate(df, variable, evidence.begin(), evidence.end());
        // }

        template<typename VarType, typename EvidenceIter>
        typename CPD::ParamsClass estimate(const DataFrame& df, const VarType& variable,  EvidenceIter evidence_begin, EvidenceIter evidence_end);
    };

    py::object mle_python_wrapper(NodeType n);
}

#endif //PGM_DATASET_MLE_BASE_HPP
