#ifndef PGM_DATASET_MLE_DISCRETEFACTOR_HPP
#define PGM_DATASET_MLE_DISCRETEFACTOR_HPP

#include <factors/discrete/DiscreteFactor.hpp>

using factors::discrete::DiscreteFactor;

namespace learning::parameters {

    template<bool contains_null, typename VarType, typename EvidenceIter>
    VectorXd _generate_values(const DataFrame& df, 
                                                            const VarType& variable, 
                                                            EvidenceIter evidence_begin, 
                                                            EvidenceIter evidence_end,
                                                            VectorXi& cardinality) {
    
    }
    

    template<bool contains_null, typename VarType, typename EvidenceIter>
    typename DiscreteFactor::ParamsClass _fit(const DataFrame& df, 
                                                const VarType& variable, 
                                                EvidenceIter evidence_begin, 
                                                EvidenceIter evidence_end) {
        
        auto num_variables = std::distance(evidence_begin, evidence_end) + 1;

        VectorXi cardinality(num_variables);
        VectorXi strides(num_variables);

        auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));

        cardinality(0) = dict_variable->dictionary()->length();
        strides(0) = 1;

        int i = 1;
        for(auto it = evidence_end; it != evidence_end; ++it, ++i) {
            auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
            cardinality(i) = dict_evidence->dictionary()->length();
            strides(i) = cardinality(i) * strides(i-1);
        }

        auto values = _generate_values<contains_null>(df, variable, evidence_begin, evidence_end, cardinality);

        return typename DiscreteFactor::ParamsClass {
            .values = values,
            .cardinality = cardinality,
            .strides = strides
        };
    }


    template<>
    template<typename VarType, typename EvidenceIter>
    typename DiscreteFactor::ParamsClass MLE<DiscreteFactor>::estimate(const DataFrame& df, 
                                                                        const VarType& variable, 
                                                                        EvidenceIter evidence_begin, 
                                                                        EvidenceIter evidence_end) {
        
        auto evidence_pair = std::make_pair(evidence_begin, evidence_end);
        auto type_id = df.same_type(variable, evidence_pair);

        bool contains_null = df.null_count(variable, evidence_pair) > 0;

        if (type_id != Type::DICTIONARY) {
            throw py::value_error("Wrong data type to fit DiscreteFactor. Ccategorical data is expected.");
        }

        if (contains_null)
            return _fit<true>(df, variable, evidence_begin, evidence_end);
        else
            return _fit<false>(df, variable, evidence_begin, evidence_end);

    }
}

#endif //PGM_DATASET_MLE_DISCRETEFACTOR_HPP
