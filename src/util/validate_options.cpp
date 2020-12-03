#include <util/validate_options.hpp>

using learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet;

namespace util {

    BayesianNetworkType check_valid_bn_string(const std::string& bn_type) {
        if (bn_type == "gbn") return BayesianNetworkType::GBN;
        if (bn_type == "spbn") return BayesianNetworkType::SPBN;
        else
            throw std::invalid_argument("Wrong Bayesian Network type \"" + bn_type + "\" specified. The possible alternatives are " 
                                        "\"gbn\" (Gaussian Bayesian networks) or \"spbn\" (Semiparametric Bayesian networks).");
    }

    ScoreType check_valid_score_string(const std::optional<std::string>& score, BayesianNetworkType bn_type) {
        if (score) {
            if (*score == "bic") return ScoreType::BIC;
            if (*score == "predic-l") return ScoreType::PREDICTIVE_LIKELIHOOD;
            if (*score == "holdout-l") return ScoreType::HOLDOUT_LIKELIHOOD;
            else
                throw std::invalid_argument("Wrong Bayesian Network score \"" + *score + "\" specified. The possible alternatives are " 
                                        "\"bic\" (Bayesian Information Criterion), \"predic-l\" (Predictive Log-likelihood) or"
                                        "\"holdout-l\" (Holdout likelihood).");
        } else {
            switch(bn_type) {
                case BayesianNetworkType::GBN:
                    return ScoreType::BIC;
                case BayesianNetworkType::SPBN:
                    return ScoreType::PREDICTIVE_LIKELIHOOD;
                default:
                    throw std::invalid_argument("Wrong BayesianNetworkType. Unreachable code!");
            }
        }
    }

    OperatorSetTypeS check_valid_operators_string(const std::optional<std::vector<std::string>>& operators, 
                                                  BayesianNetworkType bn_type) {

        if (operators) {
            OperatorSetTypeS ops(operators->size());
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
                case BayesianNetworkType::GBN:
                    return {OperatorSetType::ARCS};
                case BayesianNetworkType::SPBN:
                    return {OperatorSetType::ARCS, OperatorSetType::NODE_TYPE};
                default:
                    throw std::invalid_argument("Wrong BayesianNetworkType. Unreachable code!");
            }
        }

    }

    std::shared_ptr<Score> check_valid_score(const DataFrame& df, 
                                             BayesianNetworkType bn_type,
                                             ScoreType score,
                                             int seed,
                                             int num_folds,
                                             double test_holdout_ratio) {
        static std::unordered_map<BayesianNetworkType, 
                                  std::unordered_set<ScoreType, typename ScoreType::HashType>, 
                                  typename BayesianNetworkType::HashType>
        map_bn_score {
            { BayesianNetworkType::GBN, { 
                                ScoreType::BIC, 
                                ScoreType::PREDICTIVE_LIKELIHOOD,
                                ScoreType::HOLDOUT_LIKELIHOOD
                                 } 
            },
            { BayesianNetworkType::SPBN, { ScoreType::PREDICTIVE_LIKELIHOOD } }
        };
        
        if (map_bn_score[bn_type].count(score) == 0) {
            throw std::invalid_argument("Score \"" + score.ToString() + "\" is not compabible with "
                                        "Bayesian network type \"" + bn_type.ToString() + "\"");
        }

        switch (score) {
            case ScoreType::BIC:
                return std::make_unique<BIC>(df);
            case ScoreType::PREDICTIVE_LIKELIHOOD:
                return std::make_unique<CVLikelihood>(df, num_folds, seed);
            case ScoreType::HOLDOUT_LIKELIHOOD:
                return std::make_unique<HoldoutLikelihood>(df, test_holdout_ratio, seed);
            default:
                throw std::invalid_argument("Wrong ScoreType. Unreachable code!");
        }
    }

    std::vector<std::shared_ptr<OperatorSet>> check_valid_operators(BayesianNetworkType bn_type, 
                                                                    const OperatorSetTypeS& operators,
                                                                    const ArcStringVector& arc_blacklist,
                                                                    const ArcStringVector& arc_whitelist,
                                                                    int max_indegree,
                                                                    const FactorStringTypeVector& type_whitelist) {
        static std::unordered_map<BayesianNetworkType,
                                  std::unordered_set<OperatorSetType, typename OperatorSetType::HashType>,
                                  typename BayesianNetworkType::HashType>
        map_bn_operators {
            { BayesianNetworkType::GBN, { OperatorSetType::ARCS }},
            { BayesianNetworkType::SPBN, { OperatorSetType::ARCS, OperatorSetType::NODE_TYPE }}
        };

        auto bn_set = map_bn_operators[bn_type];
        for (auto& op : operators) {
            if (bn_set.count(op) == 0) {
                throw std::invalid_argument("Operator \"" + op.ToString() + "\" is not compabible with " 
                                            "Bayesian network type \"" + bn_type.ToString() + "\"");
            }
        }

        std::vector<std::shared_ptr<OperatorSet>> res;

        for (auto& op : operators) {
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

        return res;
    }
}