#ifndef PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETECPD_HPP
#define PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETECPD_HPP

#include <learning/parameters/mle_base.hpp>
#include <factors/discrete/DiscreteCPD.hpp>

using factors::discrete::discrete_indices;
using factors::discrete::DiscreteCPD;

namespace learning::parameters {

VectorXd _joint_counts(const DataFrame& df,
                       const std::string& variable,
                       const std::vector<std::string>& evidence,
                       VectorXi& cardinality,
                       VectorXi& strides);
typename DiscreteCPD::ParamsClass _fit(const DataFrame& df,
                                          const std::string& variable,
                                          const std::vector<std::string>& evidence);

}  // namespace learning::parameters

#endif  // PYBNESIAN_LEARNING_PARAMETERS_MLE_DISCRETECPD_HPP
