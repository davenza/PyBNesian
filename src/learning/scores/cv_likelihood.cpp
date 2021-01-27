#include <learning/scores/cv_likelihood.hpp>

namespace learning::scores {


    double CVLikelihood::local_score(const BayesianNetworkBase& model,
                                     const std::string& variable,
                                     const std::vector<std::string>& evidence) const {
        switch (model.type()) {
            case BayesianNetworkType::Gaussian: {
                LinearGaussianCPD cpd(variable, evidence);
                double loglik = 0;
                for (auto [train_df, test_df] : m_cv.loc(variable, evidence)) {
                    cpd.fit(train_df);
                    loglik += cpd.slogl(test_df);
                }

                return loglik;
            }
            case BayesianNetworkType::Semiparametric: {
                const auto& spbn = dynamic_cast<const SemiparametricBNBase&>(model);
                FactorType variable_type = spbn.node_type(variable);
                return local_score(variable_type, variable, evidence);   
            }
            default:
                throw std::invalid_argument("Bayesian network type " + 
                                            models::BayesianNetworkType_ToString(model.type()) 
                                                + " not valid for score CVLikelihood");
        }                  
    }

    double CVLikelihood::local_score(FactorType variable_type,
                                     const std::string& variable,
                                     const std::vector<std::string>& evidence) const {
        if (variable_type == FactorType::LinearGaussianCPD) {
            LinearGaussianCPD cpd(variable, evidence);

            double loglik = 0;
            for (auto [train_df, test_df] : m_cv.loc(variable, evidence)) {
                cpd.fit(train_df);
                loglik += cpd.slogl(test_df);
            }

            return loglik;
        } else {
            CKDE cpd(variable, evidence);

            double loglik = 0;
            for (auto [train_df, test_df] : m_cv.loc(variable, evidence)) {
                cpd.fit(train_df);
                loglik += cpd.slogl(test_df);
            }

            return loglik;
        }
    }
}