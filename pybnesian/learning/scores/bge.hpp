#ifndef PYBNESIAN_LEARNING_SCORES_BGE_HPP
#define PYBNESIAN_LEARNING_SCORES_BGE_HPP

#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>

using dataset::DataFrame;
using learning::scores::Score;
using models::BayesianNetworkBase, models::BayesianNetworkType, models::GaussianNetworkType;

namespace learning::scores {

class BGe : public Score {
public:
    BGe(const DataFrame& df,
        double iss_mu = 1,
        std::optional<double> iss_w = std::nullopt,
        std::optional<VectorXd> nu = std::nullopt)
        : m_df(df),
          m_iss_mu(iss_mu),
          m_iss_w(),
          m_nu(),
          m_cached_sse(),
          m_cached_means(),
          m_is_cached(false),
          m_cached_indices() {
        if (iss_w) {
            if (*iss_w <= df->num_columns() - 1) {
                throw std::invalid_argument(
                    "Imaginary sample size for Wishart prior must be greater than "
                    " num_columns - 1 (" +
                    std::to_string(df->num_columns() - 1) + ").");
            }

            m_iss_w = *iss_w;
        } else {
            m_iss_w = df->num_columns() + 2;
        }

        if (nu) {
            if (nu->rows() != df->num_columns()) {
                throw std::invalid_argument("\"nu\" argument contains " + std::to_string(nu->rows()) +
                                            " elements, "
                                            "but DataFrame \"df\" contains " +
                                            std::to_string(df->num_columns()) + " columns.");
            }
        }

        m_nu = nu;

        auto continuous_indices = df.continuous_columns();

        if (m_df.null_count(continuous_indices) == 0) {
            m_is_cached = true;
            for (int i = 0, size = continuous_indices.size(); i < size; ++i) {
                m_cached_indices.insert(std::make_pair(m_df->column_name(continuous_indices[i]), i));
            }

            switch (m_df.same_type(continuous_indices)->id()) {
                case Type::DOUBLE:
                    m_cached_means = m_df.means<arrow::DoubleType>(continuous_indices);
                    m_cached_sse = std::move(*m_df.sse<arrow::DoubleType, false>(continuous_indices));
                    break;
                case Type::FLOAT:
                    m_cached_means = m_df.means<arrow::FloatType>(continuous_indices).template cast<double>();
                    m_cached_sse = m_df.sse<arrow::FloatType, false>(continuous_indices)->template cast<double>();
                    break;
                default:
                    break;
            }
        }
    }

    double local_score(const BayesianNetworkBase& model,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override;

    double local_score(const BayesianNetworkBase& model,
                       const std::shared_ptr<FactorType>& node_type,
                       const std::string& variable,
                       const std::vector<std::string>& parents) const override;

    std::string ToString() const override { return "BGe"; }

    bool has_variables(const std::string& name) const override { return m_df.has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const override { return m_df.has_columns(cols); }

    bool compatible_bn(const BayesianNetworkBase& model) const override {
        const auto& model_type = model.type_ref();
        return model_type.is_homogeneous() && *model_type.default_node_type() == LinearGaussianCPDType::get_ref() &&
               m_df.has_columns(model.nodes());
    }

    bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
        const auto& model_type = model.type_ref();
        return model_type.is_homogeneous() && *model_type.default_node_type() == LinearGaussianCPDType::get_ref() &&
               m_df.has_columns(model.joint_nodes());
    }

    DataFrame data() const override { return m_df; }

private:
    int cached_index(int v) const {
        auto it = m_cached_indices.find(m_df->column_name(v));
        if (it == m_cached_indices.end())
            throw std::invalid_argument("Continuous variable " + std::to_string(v) + " not present in BGe.");
        return it->second;
    }
    int cached_index(const std::string& name) const {
        auto it = m_cached_indices.find(name);
        if (it == m_cached_indices.end())
            throw std::invalid_argument("Continuous variable " + name + " not present in BGe.");
        return it->second;
    }

    double bge_impl(const BayesianNetworkBase& model,
                    const std::string& variable,
                    const std::vector<std::string>& parents) const;

    template <typename ArrowType>
    double bge_no_parents(const std::string& variable, int total_nodes, double nu) const;
    double bge_no_parents(const std::string& variable, int total_nodes, double nu) const;

    template <typename ArrowType>
    double bge_parents(const std::string& variable,
                       const std::vector<std::string>& parents,
                       int total_nodes,
                       VectorXd& nu) const;
    double bge_parents(const std::string& variable,
                       const std::vector<std::string>& parents,
                       int total_nodes,
                       VectorXd& nu) const;

    void generate_cached_r(MatrixXd& r, const std::string& variable, const std::vector<std::string>& parents) const;
    void generate_r(MatrixXd& r, const std::string& variable, const std::vector<std::string>& parents) const;

    void generate_cached_means(VectorXd& means,
                               const std::string& variable,
                               const std::vector<std::string>& parents) const;
    void generate_means(VectorXd& means, const std::string& variable, const std::vector<std::string>& parents) const;

    const DataFrame m_df;
    double m_iss_mu;
    double m_iss_w;
    std::optional<VectorXd> m_nu;
    MatrixXd m_cached_sse;
    VectorXd m_cached_means;
    bool m_is_cached;
    std::unordered_map<std::string, int> m_cached_indices;
};

template <typename ArrowType>
double BGe::bge_no_parents(const std::string& variable, int total_nodes, double nu) const {
    double N = m_df.valid_rows(variable);

    double logprob = 0.5 * (log(m_iss_mu) - log(N + m_iss_mu));
    logprob += lgamma(0.5 * (N + m_iss_w - total_nodes + 1)) - lgamma(0.5 * (m_iss_w - total_nodes + 1));
    logprob -= 0.5 * N * log(util::pi<double>);

    double t = m_iss_mu * (m_iss_w - total_nodes - 1) / (m_iss_mu + 1);
    logprob += 0.5 * (m_iss_w - total_nodes + 1) * log(t);

    double mean = m_df.mean(variable);
    double nu_diff = mean - nu;

    double sse = [this, &variable, mean]() {
        if (m_df.null_count(variable) == 0) {
            auto column = m_df.to_eigen<false, ArrowType, false>(variable);
            return (column->array() - mean).matrix().squaredNorm();
        } else {
            auto column = m_df.to_eigen<false, ArrowType, true>(variable);
            return (column->array() - mean).matrix().squaredNorm();
        }
    }();

    double r = t + sse + ((N * m_iss_mu) / (N + m_iss_mu) * nu_diff * nu_diff);

    logprob -= 0.5 * (N + m_iss_w - total_nodes + 1) * log(r);
    return logprob;
}

template <typename ArrowType>
double BGe::bge_parents(const std::string& variable,
                        const std::vector<std::string>& evidence,
                        int total_nodes,
                        VectorXd& nu) const {
    double N = m_df.valid_rows(variable, evidence);
    double p = evidence.size();

    double logprob = 0.5 * (log(m_iss_mu) - log(N + m_iss_mu));
    logprob += lgamma(0.5 * (N + m_iss_w - total_nodes + p + 1)) - lgamma(0.5 * (m_iss_w - total_nodes + p + 1));
    logprob -= 0.5 * N * log(util::pi<double>);

    double t = m_iss_mu * (m_iss_w - total_nodes - 1) / (m_iss_mu + 1);
    // This is easier than bnlearn
    logprob += 0.5 * (m_iss_w - total_nodes + 2 * p + 1) * log(t);

    VectorXd means_full(evidence.size() + 1);
    MatrixXd r_full(evidence.size() + 1, evidence.size() + 1);

    if (m_is_cached) {
        generate_cached_means(means_full, variable, evidence);
        generate_cached_r(r_full, variable, evidence);
    } else {
        generate_means(means_full, variable, evidence);
        generate_r(r_full, variable, evidence);
    }

    for (size_t i = 0, end = evidence.size() + 1; i < end; ++i) {
        r_full(i, i) += t;
    }

    double cte_r = (N * m_iss_mu) / (N + m_iss_mu);

    for (size_t i = 1, end = evidence.size() + 1; i < end; ++i) {
        r_full(0, i) += cte_r * (means_full(0) - nu(0)) * (means_full(i) - nu(i));
        r_full(i, 0) = r_full(0, i);

        for (size_t j = i; j < end; ++j) {
            r_full(i, j) += cte_r * (means_full(i) - nu(i)) * (means_full(j) - nu(j));
            r_full(j, i) = r_full(i, j);
        }
    }

    r_full(0, 0) += cte_r * (means_full(0) - nu(0)) * (means_full(0) - nu(0));

    // Can be implemented with Cholesky.
    logprob -= 0.5 * (N + m_iss_w - total_nodes + p + 1) * log(r_full.determinant());
    auto r_parents = r_full.bottomRightCorner(evidence.size(), evidence.size());
    logprob += 0.5 * (N + m_iss_w - total_nodes + p) * log(r_parents.determinant());
    return logprob;
}

using DynamicBGe = DynamicScoreAdaptator<BGe>;

}  // namespace learning::scores

#endif  // PYBNESIAN_LEARNING_SCORES_BGE_HPP
