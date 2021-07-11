#ifndef PYBNESIAN_LEARNING_SCORES_SCORES_HPP
#define PYBNESIAN_LEARNING_SCORES_SCORES_HPP

#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <dataset/dynamic_dataset.hpp>

using dataset::DynamicDataFrame, dataset::DynamicAdaptator;
using models::BayesianNetworkBase, models::GaussianNetwork, models::SemiparametricBN;
using models::ConditionalBayesianNetworkBase;

namespace learning::scores {

class Score {
public:
    virtual ~Score() {}
    virtual double score(const BayesianNetworkBase& model) const {
        double s = 0;
        for (const auto& node : model.nodes()) {
            s += local_score(model, node);
        }

        return s;
    }

    virtual double local_score(const BayesianNetworkBase& model, const std::string& variable) const {
        auto parents = model.parents(variable);
        return local_score(model, variable, parents);
    }

    virtual double local_score(const BayesianNetworkBase& model,
                               const std::string& variable,
                               const std::vector<std::string>& parents) const = 0;
    virtual double local_score(const BayesianNetworkBase& model,
                               const std::shared_ptr<FactorType>& node_type,
                               const std::string& variable,
                               const std::vector<std::string>& parents) const = 0;

    virtual std::string ToString() const = 0;
    virtual bool has_variables(const std::string& name) const = 0;
    virtual bool has_variables(const std::vector<std::string>& cols) const = 0;
    virtual bool compatible_bn(const BayesianNetworkBase& model) const = 0;
    virtual bool compatible_bn(const ConditionalBayesianNetworkBase& model) const = 0;
    virtual DataFrame data() const = 0;
};

class ValidatedScore : public Score {
public:
    virtual ~ValidatedScore() {}

    virtual double vscore(const BayesianNetworkBase& model) const {
        double s = 0;
        for (const auto& node : model.nodes()) {
            s += vlocal_score(model, node);
        }

        return s;
    }

    virtual double vlocal_score(const BayesianNetworkBase& model, const std::string& variable) const {
        auto parents = model.parents(variable);
        return vlocal_score(model, variable, parents);
    }

    virtual double vlocal_score(const BayesianNetworkBase& model,
                                const std::string& variable,
                                const std::vector<std::string>& parents) const = 0;
    virtual double vlocal_score(const BayesianNetworkBase&,
                                const std::shared_ptr<FactorType>& variable_type,
                                const std::string& variable,
                                const std::vector<std::string>& parents) const = 0;
};

class DynamicScore {
public:
    virtual ~DynamicScore() {}
    virtual Score& static_score() = 0;
    virtual Score& transition_score() = 0;

    virtual bool has_variables(const std::string& name) const = 0;
    virtual bool has_variables(const std::vector<std::string>& cols) const = 0;
};

template <typename BaseScore>
class DynamicScoreAdaptator : public DynamicScore, public DynamicAdaptator<BaseScore> {
public:
    template <typename... Args>
    DynamicScoreAdaptator(const DynamicDataFrame& df, const Args&... args) : DynamicAdaptator<BaseScore>(df, args...) {}

    Score& static_score() override { return this->static_element(); }

    Score& transition_score() override { return this->transition_element(); }

    bool has_variables(const std::string& name) const override {
        return DynamicAdaptator<BaseScore>::has_variables(name);
    }

    bool has_variables(const std::vector<std::string>& cols) const override {
        return DynamicAdaptator<BaseScore>::has_variables(cols);
    }
};

}  // namespace learning::scores

#endif  // PYBNESIAN_LEARNING_SCORES_SCORES_HPP
