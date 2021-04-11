#include <learning/parameters/mle_DiscreteFactor.hpp>

namespace learning::parameters {

VectorXd _joint_counts(const DataFrame& df,
                       const std::string& variable,
                       const std::vector<std::string>& evidence,
                       VectorXi& cardinality,
                       VectorXi& strides) {
    auto joint_values = cardinality.prod();

    VectorXd counts = VectorXd::Zero(joint_values);
    VectorXi indices = discrete_indices(df, variable, evidence, strides);

    // Compute counts
    for (auto i = 0; i < indices.rows(); ++i) {
        ++counts(indices(i));
    }

    return counts;
}

typename DiscreteFactor::ParamsClass _fit(const DataFrame& df,
                                          const std::string& variable,
                                          const std::vector<std::string>& evidence) {
    auto num_variables = evidence.size() + 1;

    VectorXi cardinality(num_variables);
    VectorXi strides(num_variables);

    auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));

    cardinality(0) = dict_variable->dictionary()->length();
    strides(0) = 1;

    for (size_t i = 1; i < num_variables; ++i) {
        auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(evidence[i - 1]));
        cardinality(i) = dict_evidence->dictionary()->length();
        strides(i) = strides(i - 1) * cardinality(i - 1);
    }

    auto logprob = _joint_counts(df, variable, evidence, cardinality, strides);

    // Normalize the CPD.
    auto parent_configurations = cardinality.bottomRows(num_variables - 1).prod();

    for (auto k = 0; k < parent_configurations; ++k) {
        auto offset = k * cardinality(0);

        int sum_configuration = 0;
        for (auto i = 0; i < cardinality(0); ++i) {
            sum_configuration += logprob(offset + i);
        }

        if (sum_configuration == 0) {
            auto loguniform = std::log(1. / cardinality(0));
            for (auto i = 0; i < cardinality(0); ++i) {
                logprob(offset + i) = loguniform;
            }
        } else {
            double logsum_configuration = std::log(sum_configuration);
            for (auto i = 0; i < cardinality(0); ++i) {
                logprob(offset + i) = std::log(logprob(offset + i)) - logsum_configuration;
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
        throw py::value_error("Wrong data type to fit DiscreteFactor. Categorical data is expected.");
    }

    return _fit(df, variable, evidence);
}

}  // namespace learning::parameters