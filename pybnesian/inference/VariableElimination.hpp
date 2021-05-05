#ifndef PYBNESIAN_INFERENCE_VARIABLEELIMINATION_HPP
#define PYBNESIAN_INFERENCE_VARIABLEELIMINATION_HPP

#include <factors/factors.hpp>
#include <models/BayesianNetwork.hpp>

using factors::Factor;

namespace inference {

class VariableElimination {
public:
    VariableElimination(std::shared_ptr<BayesianNetworkBase> model) : m_model(model) {}

    std::shared_ptr<Factor> query(const std::vector<std::string>& q,
                                  std::vector<std::pair<std::string, Expression>> evidence) const;

private:
    std::shared_ptr<BayesianNetworkBase> m_model;
};

}  // namespace inference

#endif  // PYBNESIAN_INFERENCE_VARIABLEELIMINATION_HPP