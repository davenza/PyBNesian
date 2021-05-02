#include <learning/scores/cv_likelihood.hpp>

namespace learning::scores {

double CVLikelihood::local_score(const BayesianNetworkBase& model,
                                 const std::string& variable,
                                 const std::vector<std::string>& evidence) const {
    return local_score(model, *model.node_type(variable), variable, evidence);
}

double CVLikelihood::local_score(const BayesianNetworkBase& model,
                                 const FactorType& variable_type,
                                 const std::string& variable,
                                 const std::vector<std::string>& evidence) const {
    auto cpd = variable_type.new_cfactor(model, variable, evidence);

    double loglik = 0;
    for (auto [train_df, test_df] : m_cv.loc(variable, evidence)) {
        cpd->fit(train_df);
        loglik += cpd->slogl(test_df);
    }

    return loglik;
}

}  // namespace learning::scores
