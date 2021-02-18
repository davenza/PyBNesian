#include <util/validate_options.hpp>
#include <learning/scores/bge.hpp>
#include <learning/scores/validated_likelihood.hpp>

using learning::scores::BGe, learning::scores::ValidatedLikelihood;
using learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet, learning::operators::OperatorPool;

namespace util {

    BayesianNetworkType check_valid_bn_string(const std::string& bn_type) {
        if (bn_type == "gbn") return BayesianNetworkType::Gaussian;
        if (bn_type == "spbn") return BayesianNetworkType::Semiparametric;
        if (bn_type == "kdebn") return BayesianNetworkType::KDENetwork;
        else
            throw std::invalid_argument("Wrong Bayesian Network type \"" + bn_type + "\" specified. The possible alternatives are " 
                                        "\"gbn\" (Gaussian Bayesian networks), \"spbn\" (Semiparametric Bayesian networks)"
                                        " or \"kdebn\" (KDE Bayesian network).");
    }

    ScoreType check_valid_score_string(const std::optional<std::string>& score, BayesianNetworkType bn_type) {
        if (score) {
            if (*score == "bic") return ScoreType::BIC;
            if (*score == "bge") return ScoreType::BGe;
            if (*score == "cv-lik") return ScoreType::CVLikelihood;
            if (*score == "holdout-lik") return ScoreType::HoldoutLikelihood;
            if (*score == "validated-lik") return ScoreType::ValidatedLikelihood;
            else
                throw std::invalid_argument("Wrong Bayesian Network score \"" + *score + "\" specified. The possible alternatives are " 
                                        "\"bic\" (Bayesian Information Criterion), \"bge\" (Bayesian Gaussian equivalent), "
                                        "\"cv-lik\" (Cross-Validated likelihood), \"holdout-l\" (Hold-out likelihood) "
                                        " or \"validated-lik\" (Validated likelihood with cross-validation).");
        } else {
            switch(bn_type) {
                case BayesianNetworkType::Gaussian:
                    return ScoreType::BIC;
                case BayesianNetworkType::Semiparametric:
                case BayesianNetworkType::KDENetwork:
                    return ScoreType::ValidatedLikelihood;
                default:
                    throw std::invalid_argument("Wrong BayesianNetworkType. Unreachable code!");
            }
        }
    }

    OperatorSetTypeS check_valid_operators_string(const std::optional<std::vector<std::string>>& operators, 
                                                  BayesianNetworkType bn_type) {

        if (operators && !operators->empty()) {
            OperatorSetTypeS ops;
            for (auto& op : *operators) {
                if (op == "arcs") ops.insert(OperatorSetType::ARCS);
                else if (op == "node_type") ops.insert(OperatorSetType::NODE_TYPE);
                else
                    throw std::invalid_argument("Wrong operator set \"" + op + "\". Valid choices are:"
                                                "\"arcs\" (Changes in arcs; addition, removal and flip) or "
                                                "\"node_type\" (Change of node type)");
            }
            
            return ops;
        } else {
            switch(bn_type) {
                case BayesianNetworkType::Gaussian:
                    return {OperatorSetType::ARCS};
                case BayesianNetworkType::Semiparametric:
                    return {OperatorSetType::ARCS, OperatorSetType::NODE_TYPE};
                case BayesianNetworkType::KDENetwork:
                    return {OperatorSetType::ARCS};
                default:
                    throw std::invalid_argument("Wrong BayesianNetworkType. Unreachable code!");
            }
        }

    }

    std::unique_ptr<Score> check_valid_score(const DataFrame& df, 
                                             BayesianNetworkType bn_type,
                                             ScoreType score,
                                             int seed,
                                             int num_folds,
                                             double test_holdout_ratio) {
        static std::unordered_map<BayesianNetworkType, 
                                  std::unordered_set<ScoreType, typename ScoreType::HashType>,
                                  typename BayesianNetworkType::HashType>
        map_bn_score {
            { BayesianNetworkType::Gaussian, { 
                                ScoreType::BIC, 
                                ScoreType::BGe,
                                ScoreType::CVLikelihood,
                                ScoreType::HoldoutLikelihood,
                                ScoreType::ValidatedLikelihood } 
            },
            { BayesianNetworkType::Semiparametric, { ScoreType::ValidatedLikelihood } },
            { BayesianNetworkType::KDENetwork, { ScoreType::ValidatedLikelihood } }
        };
        
        if (map_bn_score[bn_type].count(score) == 0) {
            throw std::invalid_argument("Score \"" + score.ToString() + "\" is not compabible with "
                                        "Bayesian network type \"" + bn_type.ToString() + "\"");
        }

        switch (score) {
            case ScoreType::BIC:
                return std::make_unique<BIC>(df);
            case ScoreType::BGe:
                return std::make_unique<BGe>(df);
            case ScoreType::CVLikelihood:
                return std::make_unique<CVLikelihood>(df, num_folds, seed);
            case ScoreType::HoldoutLikelihood:
                return std::make_unique<HoldoutLikelihood>(df, test_holdout_ratio, seed);
            case ScoreType::ValidatedLikelihood:
                return std::make_unique<ValidatedLikelihood>(df, test_holdout_ratio, num_folds, seed);
            default:
                throw std::invalid_argument("Wrong ScoreType. Unreachable code!");
        }
    }

    std::shared_ptr<OperatorSet> check_valid_operators(BayesianNetworkType bn_type, 
                                                       const OperatorSetTypeS& operators,
                                                       const ArcStringVector& arc_blacklist,
                                                       const ArcStringVector& arc_whitelist,
                                                       int max_indegree,
                                                       const FactorStringTypeVector& type_whitelist) {
        static std::unordered_map<BayesianNetworkType,
                                  std::unordered_set<OperatorSetType, typename OperatorSetType::HashType>,
                                  typename BayesianNetworkType::HashType>
        map_bn_operators {
            { BayesianNetworkType::Gaussian, { OperatorSetType::ARCS }},
            { BayesianNetworkType::Semiparametric, { OperatorSetType::ARCS, OperatorSetType::NODE_TYPE }},
            { BayesianNetworkType::KDENetwork, { OperatorSetType::ARCS }}
        };

        auto bn_set = map_bn_operators[bn_type];
        for (auto op : operators) {
            if (bn_set.count(op) == 0) {
                throw std::invalid_argument("Operator \"" + op.ToString() + "\" is not compabible with " 
                                            "Bayesian network type \"" + bn_type.ToString() + "\"");
            }
        }

        std::vector<std::shared_ptr<OperatorSet>> res;

        for (auto op : operators) {
            switch (op) {
                case OperatorSetType::ARCS:
                    res.push_back(std::make_shared<ArcOperatorSet>(arc_blacklist, arc_whitelist, max_indegree));
                    break;
                case OperatorSetType::NODE_TYPE:
                    res.push_back(std::make_shared<ChangeNodeTypeSet>(type_whitelist));
                    break;
                default:
                    throw std::invalid_argument("Wrong OperatorSetType. Unreachable code!");
            }
        }

        if (res.size() == 1) {
            return std::move(res[0]);
        } else {
            return std::make_unique<OperatorPool>(std::move(res));
        }
    }
}