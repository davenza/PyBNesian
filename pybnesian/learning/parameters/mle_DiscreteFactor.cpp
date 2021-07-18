#include <learning/parameters/mle_DiscreteFactor.hpp>

namespace learning::parameters {

typename DiscreteFactor::ParamsClass _fit(const DataFrame& df,
                                          const std::string& variable,
                                          const std::vector<std::string>& evidence) {
    auto num_variables = evidence.size() + 1;

    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(df, variable, evidence);

    auto joint_counts = factors::discrete::joint_counts(df, variable, evidence, cardinality, strides);

    // Normalize the CPD.
    auto parent_configurations = cardinality.bottomRows(num_variables - 1).prod();

    VectorXd logprob(joint_counts.rows());

    for (auto k = 0; k < parent_configurations; ++k) {
        auto offset = k * cardinality(0);

        int sum_configuration = 0;
        for (auto i = 0; i < cardinality(0); ++i) {
            sum_configuration += joint_counts(offset + i);
        }

        if (sum_configuration == 0) {
            auto loguniform = std::log(1. / cardinality(0));
            for (auto i = 0; i < cardinality(0); ++i) {
                logprob(offset + i) = loguniform;
            }
        } else {
            double logsum_configuration = std::log(static_cast<double>(sum_configuration));
            for (auto i = 0; i < cardinality(0); ++i) {
                logprob(offset + i) = std::log(static_cast<double>(joint_counts(offset + i))) - logsum_configuration;
            }
        }
    }

    return typename DiscreteFactor::ParamsClass{/*.logprob = */ logprob,
                                                /*.cardinality = */ cardinality};
}

template <>
typename DiscreteFactor::ParamsClass MLE<DiscreteFactor>::estimate(const DataFrame& df,
                                                                   const std::string& variable,
                                                                   const std::vector<std::string>& evidence) {
    auto type = df.same_type(variable, evidence);

    if (type->id() != Type::DICTIONARY) {
        throw std::invalid_argument("Wrong data type to fit DiscreteFactor. Categorical data is expected.");
    }

    return _fit(df, variable, evidence);
}

}  // namespace learning::parameters