#include <learning/scores/bde.hpp>

namespace learning::scores {

double BDe::bde_impl_noparents(const std::string& variable) const {
    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, variable, {});
    auto joint_counts = factors::discrete::joint_counts(m_df, variable, {}, cardinality, strides);

    double alpha = m_iss / cardinality(0);

    auto num_rows = 0;
    auto res = -cardinality(0) * std::lgamma(alpha);
    for (auto i = 0; i < joint_counts.rows(); ++i) {
        num_rows += joint_counts(i);
        res += std::lgamma(joint_counts(i) + alpha);
    }

    res += std::lgamma(m_iss) - std::lgamma(m_iss + num_rows);

    return res;
}

double BDe::bde_impl_parents(const std::string& variable, const std::vector<std::string>& parents) const {
    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, variable, parents);
    auto joint_counts = factors::discrete::joint_counts(m_df, variable, parents, cardinality, strides);

    auto cardinality_prod = cardinality.prod();
    double alpha = m_iss / cardinality_prod;
    auto parent_configurations = cardinality_prod / cardinality(0);

    auto res = -cardinality_prod * std::lgamma(alpha);
    for (auto k = 0; k < parent_configurations; ++k) {
        auto offset = k * cardinality(0);
        auto sum = 0;

        for (auto i = 0; i < cardinality(0); ++i) {
            auto m = joint_counts(offset + i);
            res += std::lgamma(m + alpha);
            sum += m;
        }

        auto sum_alpha = alpha * cardinality(0);
        res += std::lgamma(sum_alpha) - std::lgamma(sum_alpha + sum);
    }

    return res;
}

double BDe::local_score(const BayesianNetworkBase& model,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const {
    if (*model.node_type(variable) == DiscreteFactorType::get_ref()) {
        if (parents.empty())
            return bde_impl_noparents(variable);
        else
            return bde_impl_parents(variable, parents);
    }

    throw std::invalid_argument("Bayesian network type \"" + model.type_ref().ToString() +
                                "\" not valid for score BGe");
}

double BDe::local_score(const BayesianNetworkBase&,
                        const std::shared_ptr<FactorType>& node_type,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const {
    if (*node_type != DiscreteFactorType::get_ref()) {
        if (parents.empty())
            return bde_impl_noparents(variable);
        else
            return bde_impl_parents(variable, parents);
    }

    throw std::invalid_argument("Node type \"" + node_type->ToString() + "\" not valid for score BGe");
}

}  // namespace learning::scores