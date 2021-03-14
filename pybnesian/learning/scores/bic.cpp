#include <learning/scores/bic.hpp>

using models::GaussianNetworkType;

namespace learning::scores {

double BIC::bic_lineargaussian(const std::string& variable, const std::vector<std::string>& parents) const {
    MLE<LinearGaussianCPD> mle;

    auto mle_params = mle.estimate(m_df, variable, parents);

    if (mle_params.variance < util::machine_tol) {
        return -std::numeric_limits<double>::infinity();
    }

    auto rows = m_df.valid_rows(variable, parents);
    auto num_parents = parents.size();
    auto loglik = 0.5 * (1 + static_cast<double>(num_parents) - static_cast<double>(rows)) -
                  0.5 * rows * std::log(2 * util::pi<double>) - rows * 0.5 * std::log(mle_params.variance);

    return loglik - std::log(rows) * 0.5 * (num_parents + 2);
}

double BIC::local_score(const BayesianNetworkBase& model,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const {
    if (*model.node_type(variable) == LinearGaussianCPDType::get_ref()) {
        return bic_lineargaussian(variable, parents);
    }

    throw std::invalid_argument("Bayesian network type \"" + model.type_ref().ToString() +
                                "\" not valid for score BIC");
}

double BIC::local_score(const BayesianNetworkBase&,
                        const FactorType& node_type,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const {
    if (node_type == LinearGaussianCPDType::get_ref()) {
        return bic_lineargaussian(variable, parents);
    }

    throw std::invalid_argument("Node type \"" + node_type.ToString() + "\" not valid for score BIC");
}

}  // namespace learning::scores