#include <learning/scores/holdout_likelihood.hpp>
#include <models/BayesianNetwork.hpp>

using models::BayesianNetworkType;

namespace learning::scores {

double HoldoutLikelihood::local_score(const BayesianNetworkBase& model,
                                      const std::string& variable,
                                      const std::vector<std::string>& evidence) const {
    return local_score(model, *model.node_type(variable), variable, evidence);
}

double HoldoutLikelihood::local_score(const BayesianNetworkBase& model,
                                      const FactorType& variable_type,
                                      const std::string& variable,
                                      const std::vector<std::string>& evidence) const {
    auto cpd = variable_type.new_cfactor(model, variable, evidence);
    cpd->fit(training_data());
    return cpd->slogl(test_data());
}

}  // namespace learning::scores