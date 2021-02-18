#include <learning/scores/cv_likelihood.hpp>

namespace learning::scores {

    double CVLikelihood::local_score(const BayesianNetworkBase& model,
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
                                            + "\" not valid for score CVLikelihood");
        }                  
    }

    double CVLikelihood::local_score(NodeType variable_type,
                                     const std::string& variable,
                                     const std::vector<std::string>& evidence) const {
        if (variable_type == NodeType::LinearGaussianCPD) {
            return factor_score<LinearGaussianCPD>(variable, evidence);
        } else {
            return factor_score<CKDE>(variable, evidence);
        }
    }
}
