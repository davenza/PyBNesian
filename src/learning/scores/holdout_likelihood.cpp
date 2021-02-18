#include <learning/scores/holdout_likelihood.hpp>
#include <models/BayesianNetwork.hpp>

using models::BayesianNetworkType, models::SemiparametricBNBase;

namespace learning::scores {


    double HoldoutLikelihood::local_score(const BayesianNetworkBase& model,
                                          const std::string& variable,
                                          const std::vector<std::string>& evidence) const {
        switch (model.type()) {
            case BayesianNetworkType::Gaussian: {
                return factor_score<LinearGaussianCPD>(variable, evidence);
            }
            case BayesianNetworkType::Semiparametric: {
                const auto& spbn = dynamic_cast<const SemiparametricBNBase&>(model);
                NodeType variable_type = spbn.node_type(variable);
                return local_score(variable_type, variable, evidence);   
            }
            case BayesianNetworkType::KDENetwork: {
                return factor_score<CKDE>(variable, evidence);
            }
            default:
                throw std::invalid_argument("Bayesian network type \"" + model.type().ToString()
                                            + "\" not valid for score HoldoutLikelihood");
        }
    }
    
    double HoldoutLikelihood::local_score(NodeType variable_type,
                                          const std::string& variable,
                                          const std::vector<std::string>& evidence) const {
        switch (variable_type) {
            case NodeType::LinearGaussianCPD: {
                return factor_score<LinearGaussianCPD>(variable, evidence);
            }
            case NodeType::CKDE: {
                return factor_score<CKDE>(variable, evidence);
            }
            default:
                throw std::invalid_argument("HoldoutLikelihood only implemented for LinearGaussianCPD and CKDE.");
        }
    }
}