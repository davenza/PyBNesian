#ifndef PYBNESIAN_LEARNING_PARAMETERS_MLE_BASE_HPP
#define PYBNESIAN_LEARNING_PARAMETERS_MLE_BASE_HPP

#include <dataset/dataset.hpp>

using namespace dataset;

namespace learning::parameters {

template <typename CPD>
class MLE {
public:
    typename CPD::ParamsClass estimate(const DataFrame& df,
                                       const std::string& variable,
                                       const std::vector<std::string>& evidence);
};

}  // namespace learning::parameters

#endif  // PYBNESIAN_LEARNING_PARAMETERS_MLE_BASE_HPP
