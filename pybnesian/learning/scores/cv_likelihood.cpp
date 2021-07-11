#include <learning/scores/cv_likelihood.hpp>

namespace learning::scores {

double CVLikelihood::local_score(const BayesianNetworkBase& model,
                                 const std::string& variable,
                                 const std::vector<std::string>& evidence) const {
    return local_score(model, model.underlying_node_type(m_cv.data(), variable), variable, evidence);
}

double CVLikelihood::local_score(const BayesianNetworkBase& model,
                                 const std::shared_ptr<FactorType>& variable_type,
                                 const std::string& variable,
                                 const std::vector<std::string>& evidence) const {
    auto [args, kwargs] = m_arguments.args(variable, variable_type);
    auto cpd = variable_type->new_factor(model, variable, evidence, args, kwargs);

    double loglik = 0;
    for (auto [train_df, test_df] : m_cv.loc(variable, evidence)) {
        cpd->fit(train_df);
        loglik += cpd->slogl(test_df);
    }

    return loglik;
}

}  // namespace learning::scores
