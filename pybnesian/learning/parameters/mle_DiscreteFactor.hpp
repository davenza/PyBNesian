#ifndef PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETEFACTOR_HPP
#define PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETEFACTOR_HPP

#include <learning/parameters/mle_base.hpp>
#include <factors/discrete/DiscreteFactor.hpp>

using factors::discrete::discrete_indices;
using factors::discrete::DiscreteFactor;

namespace learning::parameters {

typename DiscreteFactor::ParamsClass _fit(const DataFrame& df,
                                          const std::string& variable,
                                          const std::vector<std::string>& evidence);

}  // namespace learning::parameters

#endif  // PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETEFACTOR_HPP
