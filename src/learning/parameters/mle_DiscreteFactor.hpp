#ifndef PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETEFACTOR_HPP
#define PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETEFACTOR_HPP

#include <learning/parameters/mle_base.hpp>
#include <factors/discrete/DiscreteFactor.hpp>

using factors::discrete::DiscreteFactor;
using factors::discrete::discrete_indices;

namespace learning::parameters {


    template<typename VarType, typename EvidenceIter>
    VectorXd _joint_counts(const DataFrame& df, 
                                const VarType& variable, 
                                EvidenceIter evidence_begin, 
                                EvidenceIter evidence_end,
                                VectorXi& cardinality,
                                VectorXi& strides) {
        auto joint_values = cardinality.prod();

        VectorXd counts = VectorXd::Zero(joint_values);
        VectorXi indices = discrete_indices(df, variable, evidence_begin, evidence_end, strides);

        // Compute counts
        for (auto i = 0; i < indices.rows(); ++i) {
            ++counts(indices(i));
        }

        return counts;
    }
    

    template<typename VarType, typename EvidenceIter>
    typename DiscreteFactor::ParamsClass _fit(const DataFrame& df, 
                                              const VarType& variable, 
                                              const EvidenceIter& evidence_begin, 
                                              const EvidenceIter& evidence_end) {
        
        auto num_variables = std::distance(evidence_begin, evidence_end) + 1;

        VectorXi cardinality(num_variables);
        VectorXi strides(num_variables);

        auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));
        
        cardinality(0) = dict_variable->dictionary()->length();
        strides(0) = 1;

        int i = 1;
        for(auto it = evidence_begin; it != evidence_end; ++it, ++i) {
            auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
            cardinality(i) = dict_evidence->dictionary()->length();
            strides(i) = strides(i-1)*cardinality(i-1);            
        }

        auto logprob = _joint_counts(df, variable, evidence_begin, evidence_end, cardinality, strides);

        // Normalize the CPD.
        auto parent_configurations = cardinality.bottomRows(num_variables-1).prod();

        for (auto k = 0; k < parent_configurations; ++k) {
            auto offset = k*cardinality(0);

            int sum_configuration = 0;
            for(auto i = 0; i < cardinality(0); ++i) {
                sum_configuration += logprob(offset + i);
            }

            if (sum_configuration == 0) {
                auto loguniform = std::log(1. / cardinality(0));
                for(auto i = 0; i < cardinality(0); ++i) {
                    logprob(offset + i) = loguniform;
                }       
            } else {
                double logsum_configuration = std::log(sum_configuration);
                for(auto i = 0; i < cardinality(0); ++i) {
                    logprob(offset + i) = std::log(logprob(offset + i)) - logsum_configuration;
                }
            }
        }

        return typename DiscreteFactor::ParamsClass {
            .logprob = logprob,
            .cardinality = cardinality,
            .strides = strides
        };
    }


    template<>
    template<typename VarType, typename EvidenceIter>
    typename DiscreteFactor::ParamsClass MLE<DiscreteFactor>::estimate(const DataFrame& df, 
                                                                       const VarType& variable, 
                                                                       const EvidenceIter& evidence_begin, 
                                                                       const EvidenceIter& evidence_end) {
        
        auto evidence_pair = std::make_pair(evidence_begin, evidence_end);
        auto type_id = df.same_type(variable, evidence_pair);

        if (type_id != Type::DICTIONARY) {
            throw py::value_error("Wrong data type to fit DiscreteFactor. Categorical data is expected.");
        }

        return _fit(df, variable, evidence_begin, evidence_end);

    }
}

#endif //PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETEFACTOR_HPP
