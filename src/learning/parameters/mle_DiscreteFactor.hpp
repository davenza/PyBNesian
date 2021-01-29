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
    typename DiscreteFactor::ParamsClass _fit(const DataFrame& df,
                                              const std::string& variable,
                                              const std::vector<std::string>& evidence);
}

#endif //PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETEFACTOR_HPP
