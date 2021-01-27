#include <learning/scores/holdout_likelihood.hpp>
#include <models/BayesianNetwork.hpp>

using models::BayesianNetworkType, models::SemiparametricBNBase;

namespace learning::scores {


    double HoldoutLikelihood::local_score(const BayesianNetworkBase& model,
                    const std::string& variable,
                    const std::vector<std::string>& evidence) const {
        switch (model.type()) {
            case BayesianNetworkType::Gaussian: {
                LinearGaussianCPD cpd(variable, evidence);
                cpd.fit(training_data());
                return cpd.slogl(test_data());
            }
            case BayesianNetworkType::Semiparametric: {
                const auto& spbn = dynamic_cast<const SemiparametricBNBase&>(model);
                FactorType variable_type = spbn.node_type(variable);
                return local_score(variable_type, variable, evidence);   
            }
            default:
                throw std::invalid_argument("Bayesian network type " + 
                                            models::BayesianNetworkType_ToString(model.type()) 
                                                + " not valid for score HoldoutLikelihood");
        }
    }
    
    double HoldoutLikelihood::local_score(FactorType variable_type,
                                          const std::string& variable,
                                          const std::vector<std::string>& evidence) const {
        if (variable_type == FactorType::LinearGaussianCPD) {
            LinearGaussianCPD cpd(variable, evidence);
            cpd.fit(training_data());
            return cpd.slogl(test_data());
        } else {
            CKDE cpd(variable, evidence);
            cpd.fit(training_data());
            return cpd.slogl(test_data());
        }
    }
}