#ifndef PYBNESIAN_LEARNING_SCORES_SCORES_HPP
#define PYBNESIAN_LEARNING_SCORES_SCORES_HPP

#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <dataset/dynamic_dataset.hpp>

using models::BayesianNetworkBase, models::GaussianNetwork, models::SemiparametricBN;
using models::ConditionalBayesianNetworkBase;
using dataset::DynamicDataFrame, dataset::DynamicAdaptator;

namespace learning::scores {

    class ScoreType
    {
    public:
        enum Value : uint8_t
        {
            BIC,
            BGe,
            CVLikelihood,
            HoldoutLikelihood,
            ValidatedLikelihood
        };

        struct Hash
        {
            inline std::size_t operator ()(ScoreType const score_type) const
            {
                return static_cast<std::size_t>(score_type.value);
            }
        };

        using HashType = Hash;

        ScoreType() = default;
        constexpr ScoreType(Value opset_type) : value(opset_type) { }

        operator Value() const { return value; }  
        explicit operator bool() = delete;

        constexpr bool operator==(ScoreType a) const { return value == a.value; }
        constexpr bool operator==(Value v) const { return value == v; }
        constexpr bool operator!=(ScoreType a) const { return value != a.value; }
        constexpr bool operator!=(Value v) const { return value != v; }

        std::string ToString() const { 
            switch(value) {
                case Value::BIC:
                    return "BIC";
                case Value::BGe:
                    return "BGe";
                case Value::CVLikelihood:
                    return "CVLikelihood";
                case Value::HoldoutLikelihood:
                    return "HoldoutLikelihood";
                case Value::ValidatedLikelihood:
                    return "ValidatedLikelihood";
                default:
                    throw std::invalid_argument("Unreachable code in ScoreType.");
            }
        }

    private:
        Value value;
    };

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

        virtual double local_score(const BayesianNetworkBase&, int) const = 0;
        virtual double local_score(const BayesianNetworkBase&, const std::string&) const = 0;
        virtual double local_score(const BayesianNetworkBase&, int, const std::vector<int>&) const = 0;
        virtual double local_score(const BayesianNetworkBase&, const std::string&, const std::vector<std::string>&) const = 0;

        virtual std::string ToString() const = 0;
        virtual bool is_decomposable() const = 0;
        virtual ScoreType type() const = 0;
        virtual bool has_variables(const std::string& name) const = 0;
        virtual bool has_variables(const std::vector<std::string>& cols) const = 0;
        virtual bool compatible_bn(const BayesianNetworkBase& model) const = 0;
        virtual bool compatible_bn(const ConditionalBayesianNetworkBase& model) const = 0;
    };

    class ScoreSPBN : public virtual Score {
    public:
        using Score::local_score;
        virtual ~ScoreSPBN() {}
        // virtual double local_score(FactorType variable_type, int variable, const std::vector<int>& evidence) const = 0;
        virtual double local_score(FactorType variable_type,
                                   const std::string& variable,
                                   const std::vector<std::string>& evidence) const = 0;
    };

    class ValidatedScore : public virtual Score {
    public:
        // using Score::local_score;
        virtual ~ValidatedScore() {}

        virtual double vscore(const BayesianNetworkBase& model) const {
            double s = 0;
            for (const auto& node : model.nodes()) {
                s += vlocal_score(model, node);
            }

            return s;
        }

        virtual double vlocal_score(const BayesianNetworkBase&, int) const = 0;
        virtual double vlocal_score(const BayesianNetworkBase&, const std::string&) const = 0;
        virtual double vlocal_score(const BayesianNetworkBase&, int, const std::vector<int>&) const = 0;
        virtual double vlocal_score(const BayesianNetworkBase&, const std::string&, const std::vector<std::string>&) const = 0;
    };

    class ValidatedScoreSPBN : public ValidatedScore, public ScoreSPBN {
    public:
        using ValidatedScore::vlocal_score;

        virtual ~ValidatedScoreSPBN() {}
        virtual double vlocal_score(FactorType variable_type,
                                   const std::string& variable,
                                   const std::vector<std::string>& evidence) const = 0;
    };

    class DynamicScore {
    public:
        virtual ~DynamicScore() {}
        virtual Score& static_score() = 0;
        virtual Score& transition_score() = 0;

        virtual bool has_variables(const std::string& name) const = 0;
        virtual bool has_variables(const std::vector<std::string>& cols) const = 0;
        virtual bool compatible_bn(const BayesianNetworkBase& model) const = 0;
        virtual bool compatible_bn(const ConditionalBayesianNetworkBase& model) const = 0;
    };

    template<typename BaseScore>
    class DynamicScoreAdaptator : public DynamicScore, public DynamicAdaptator<BaseScore> {
    public:
        template<typename... Args>
        DynamicScoreAdaptator(const DynamicDataFrame& df,
                              const Args&... args) : DynamicAdaptator<BaseScore>(df, args...) {}

        Score& static_score() override {
            return this->static_element();
        }

        Score& transition_score() override {
            return this->transition_element();
        }

        bool has_variables(const std::string& name) const override {
            return DynamicAdaptator<BaseScore>::has_variables(name);
        }

        bool has_variables(const std::vector<std::string>& cols) const override {
            return DynamicAdaptator<BaseScore>::has_variables(cols);
        }

        bool compatible_bn(const BayesianNetworkBase& model) const override {
            return DynamicAdaptator<BaseScore>::has_variables(model.nodes());
        }

        bool compatible_bn(const ConditionalBayesianNetworkBase& model) const override {
            return DynamicAdaptator<BaseScore>::has_variables(model.all_nodes());
        }
    };
}

#endif //PYBNESIAN_LEARNING_SCORES_SCORES_HPP
