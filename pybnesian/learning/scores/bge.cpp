#include <learning/scores/bge.hpp>

using models::GaussianNetworkType;

namespace learning::scores {

double BGe::bge_no_parents(const std::string& variable, int total_nodes, double nu) const {
    auto type = m_df.same_type(variable);
    switch (type->id()) {
        case Type::DOUBLE:
            return bge_no_parents<arrow::DoubleType>(variable, total_nodes, nu);
        case Type::FLOAT:
            return bge_no_parents<arrow::FloatType>(variable, total_nodes, nu);
        default:
            throw std::invalid_argument("Variable " + variable +
                                        " has data type "
                                        "\"" +
                                        type->ToString() +
                                        "\" but BGe"
                                        " requires \"double\" or \"float\" data type.");
    }
}

double BGe::bge_parents(const std::string& variable,
                        const std::vector<std::string>& parents,
                        int total_nodes,
                        VectorXd& nu) const {
    auto type = m_df.same_type(variable);
    switch (type->id()) {
        case Type::DOUBLE:
            return bge_parents<arrow::DoubleType>(variable, parents, total_nodes, nu);
        case Type::FLOAT:
            return bge_parents<arrow::FloatType>(variable, parents, total_nodes, nu);
        default:
            throw std::invalid_argument("Variables has data type \"" + type->ToString() +
                                        "\" but BGe"
                                        " requires \"double\" or \"float\" data type.");
    }
}

void BGe::generate_cached_r(MatrixXd& r, const std::string& variable, const std::vector<std::string>& parents) const {
    int var_index = cached_index(variable);
    r(0, 0) = m_cached_sse(var_index, var_index);

    for (size_t i = 0, end = parents.size(); i < end; ++i) {
        int ei_index = cached_index(parents[i]);
        r(i + 1, i + 1) = m_cached_sse(ei_index, ei_index);

        r(0, i + 1) = r(i + 1, 0) = m_cached_sse(var_index, ei_index);

        for (size_t j = i + 1; j < end; ++j) {
            int ej_index = cached_index(parents[j]);
            r(i + 1, j + 1) = r(j + 1, i + 1) = m_cached_sse(ei_index, ej_index);
        }
    }
}

void BGe::generate_r(MatrixXd& r, const std::string& variable, const std::vector<std::string>& parents) const {
    auto type = m_df.same_type(variable, parents);
    switch (type->id()) {
        case Type::DOUBLE: {
            r = *m_df.sse<arrow::DoubleType>(variable, parents);
            break;
        }
        case Type::FLOAT: {
            r = m_df.sse<arrow::FloatType>(variable, parents)->template cast<double>();
            break;
        }
        default:
            throw std::invalid_argument("Variables has data type \"" + type->ToString() +
                                        "\" but BGe"
                                        " requires \"double\" or \"float\" data type.");
    }
}

void BGe::generate_cached_means(VectorXd& means,
                                const std::string& variable,
                                const std::vector<std::string>& parents) const {
    int var_index = cached_index(variable);
    means(0) = m_cached_means(var_index);

    for (size_t i = 0, end = parents.size(); i < end; ++i) {
        int ei_index = cached_index(parents[i]);
        means(i + 1) = m_cached_means(ei_index);
    }
}

void BGe::generate_means(VectorXd& means, const std::string& variable, const std::vector<std::string>& parents) const {
    auto type = m_df.same_type(variable, parents);
    switch (type->id()) {
        case Type::DOUBLE: {
            means = m_df.means<arrow::DoubleType>(variable, parents);
            break;
        }
        case Type::FLOAT: {
            means = m_df.means<arrow::FloatType>(variable, parents).template cast<double>();
            break;
        }
        default:
            throw std::invalid_argument("Variables has data type \"" + type->ToString() +
                                        "\" but BGe"
                                        " requires \"double\" or \"float\" data type.");
    }
}

double BGe::bge_impl(const BayesianNetworkBase& model,
                     const std::string& variable,
                     const std::vector<std::string>& parents) const {
    if (parents.empty()) {
        double nu = [this, &variable]() {
            if (m_nu) {
                return (*m_nu)(m_df.index(variable));
            } else {
                return m_df.mean(variable);
            }
        }();

        auto col = m_df.col(variable);

        return bge_no_parents(variable, model.num_nodes(), nu);
    } else {
        VectorXd nu = [this, &variable, &parents]() {
            if (m_nu) {
                VectorXd res(parents.size() + 1);
                res(0) = (*m_nu)(m_df.index(variable));
                int i = 0;
                for (const auto& e : parents) {
                    res(++i) = (*m_nu)(m_df.index(e));
                }

                return res;
            } else {
                auto combined_bitmap = m_df.combined_bitmap(variable, parents);
                if (combined_bitmap) {
                    return m_df.means(combined_bitmap, variable, parents);
                } else {
                    return m_df.means(variable, parents);
                }
            }
        }();

        return bge_parents(variable, parents, model.num_nodes(), nu);
    }
}

double BGe::local_score(const BayesianNetworkBase& model,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const {
    if (*model.node_type(variable) == LinearGaussianCPDType::get_ref()) {
        return bge_impl(model, variable, parents);
    }

    throw std::invalid_argument("Bayesian network type \"" + model.type_ref().ToString() +
                                "\" not valid for score BGe");
}

double BGe::local_score(const BayesianNetworkBase& model,
                        const std::shared_ptr<FactorType>& node_type,
                        const std::string& variable,
                        const std::vector<std::string>& parents) const {
    if (*node_type != LinearGaussianCPDType::get_ref()) {
        return bge_impl(model, variable, parents);
    }

    throw std::invalid_argument("Node type \"" + node_type->ToString() + "\" not valid for score BGe");
}

}  // namespace learning::scores