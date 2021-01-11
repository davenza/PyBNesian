#ifndef PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETEFACTOR_HPP
#define PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETEFACTOR_HPP

#include <learning/parameters/mle_base.hpp>
#include <factors/discrete/DiscreteFactor.hpp>

using factors::discrete::DiscreteFactor;
using factors::discrete::discrete_indices;

namespace learning::parameters {

    VectorXd _joint_counts(const DataFrame& df, 
                           const std::string& variable, 
                           const std::vector<std::string>& evidence, 
                           VectorXi& cardinality,
                           VectorXi& strides);
    //     auto joint_values = cardinality.prod();

    //     VectorXd counts = VectorXd::Zero(joint_values);
    //     VectorXi indices = discrete_indices(df, variable, evidence, strides);

    //     // Compute counts
    //     for (auto i = 0; i < indices.rows(); ++i) {
    //         ++counts(indices(i));
    //     }

    //     return counts;
    // }
    
    typename DiscreteFactor::ParamsClass _fit(const DataFrame& df,
                                              const std::string& variable,
                                              const std::vector<std::string>& evidence);
        
    //     auto num_variables = evidence.size() + 1;

    //     VectorXi cardinality(num_variables);
    //     VectorXi strides(num_variables);

    //     auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));
        
    //     cardinality(0) = dict_variable->dictionary()->length();
    //     strides(0) = 1;

    //     int i = 1;
    //     for(const auto& ev : evidence) {
    //         auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(ev));
    //         cardinality(i) = dict_evidence->dictionary()->length();
    //         strides(i) = strides(i-1)*cardinality(i-1);            
    //     }

    //     auto logprob = _joint_counts(df, variable, evidence, cardinality, strides);

    //     // Normalize the CPD.
    //     auto parent_configurations = cardinality.bottomRows(num_variables-1).prod();

    //     for (auto k = 0; k < parent_configurations; ++k) {
    //         auto offset = k*cardinality(0);

    //         int sum_configuration = 0;
    //         for(auto i = 0; i < cardinality(0); ++i) {
    //             sum_configuration += logprob(offset + i);
    //         }

    //         if (sum_configuration == 0) {
    //             auto loguniform = std::log(1. / cardinality(0));
    //             for(auto i = 0; i < cardinality(0); ++i) {
    //                 logprob(offset + i) = loguniform;
    //             }       
    //         } else {
    //             double logsum_configuration = std::log(sum_configuration);
    //             for(auto i = 0; i < cardinality(0); ++i) {
    //                 logprob(offset + i) = std::log(logprob(offset + i)) - logsum_configuration;
    //             }
    //         }
    //     }

    //     return typename DiscreteFactor::ParamsClass {
    //         .logprob = logprob,
    //         .cardinality = cardinality,
    //         .strides = strides
    //     };
    // }
}

#endif //PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETEFACTOR_HPP
