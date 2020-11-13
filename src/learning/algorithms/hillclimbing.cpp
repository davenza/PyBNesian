#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_dtype.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>
#include <learning/operators/operators.hpp>

using namespace dataset;

using Eigen::VectorXd, Eigen::MatrixXd;;
using models::GaussianNetwork, models::BayesianNetworkType;
using learning::scores::ScoreType, learning::scores::BIC, learning::scores::CVLikelihood, learning::scores::HoldoutLikelihood;
using learning::operators::OperatorSet, learning::operators::OperatorSetType, 
      learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet;

using util::ArcVector;


namespace learning::algorithms {

    using OperatorSetTypeS = std::unordered_set<OperatorSetType, typename OperatorSetType::HashType>;

    BayesianNetworkType check_valid_bn_string(std::string& bn_type) {
        if (bn_type == "gbn") return BayesianNetworkType::GBN;
        if (bn_type == "spbn") return BayesianNetworkType::SPBN;
        else
            throw std::invalid_argument("Wrong Bayesian Network type \"" + bn_type + "\" specified. The possible alternatives are " 
                                        "\"gbn\" (Gaussian Bayesian networks) or \"spbn\" (Semiparametric Bayesian networks).");
    }

    ScoreType check_valid_score_string(std::string& score) {
        if (score == "bic") return ScoreType::BIC;
        if (score == "predic-l") return ScoreType::PREDICTIVE_LIKELIHOOD;
        else
            throw std::invalid_argument("Wrong Bayesian Network score \"" + score + "\" specified. The possible alternatives are " 
                                    "\"bic\" (Bayesian Information Criterion) or \"predic-l\" (Predicitive Log-likelihood).");
        
    }

    OperatorSetTypeS check_valid_operators_string(std::vector<std::string>& operators) {
        std::unordered_set<OperatorSetType, typename OperatorSetType::HashType> ops(operators.size());
        for (auto& op : operators) {
            if (op == "arcs") ops.insert(OperatorSetType::ARCS);
            else if (op == "node_type") ops.insert(OperatorSetType::NODE_TYPE);
            else
                throw std::invalid_argument("Wrong operator set \"" + op + "\". Valid choices are:"
                                            "\"arcs\" (Changes in arcs; addition, removal and flip) or "
                                            "\"node_type\" (Change of node type)");
        }
        return ops;
    }

    void check_valid_score(BayesianNetworkType bn_type, ScoreType score) {
        static std::unordered_map<BayesianNetworkType, 
                                  std::unordered_set<ScoreType, typename ScoreType::HashType>, 
                                  typename BayesianNetworkType::HashType>
        map_bn_score {
            { BayesianNetworkType::GBN, { ScoreType::BIC, ScoreType::PREDICTIVE_LIKELIHOOD } },
            { BayesianNetworkType::SPBN, { ScoreType::PREDICTIVE_LIKELIHOOD } }
        };
        
        if (map_bn_score[bn_type].count(score) == 0) {
            throw std::invalid_argument("Score \"" + score.ToString() + "\" is not compabible with "
                                        "Bayesian network type \"" + bn_type.ToString() + "\"");
        }
    }

    void check_valid_operators(BayesianNetworkType bn_type, OperatorSetTypeS& operators) 
    {
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
    }

    py::object call_hc_validated_spbn(const DataFrame& df, 
                               ArcVector arc_blacklist, 
                               ArcVector arc_whitelist, 
                               FactorTypeVector type_whitelist, 
                               int max_indegree,
                               int max_iters,
                               double epsilon,
                               int patience,
                               int verbose) 
    {
        using Model = SemiparametricBN;
        auto nodes = df.column_names();

        Model m(nodes, arc_whitelist, type_whitelist);

        HoldoutLikelihood validation_score(df, 0.2, 0);
        std::shared_ptr<Score> training_score = std::make_shared<CVLikelihood>(validation_score.training_data(), 10, 0);
        
        auto arc_set = std::make_shared<ArcOperatorSet>(training_score);
        auto nodetype_set = std::make_shared<ChangeNodeTypeSet>(training_score);
        std::vector<std::shared_ptr<OperatorSet>> v{std::move(arc_set), std::move(nodetype_set)};

        OperatorPool pool(m, training_score, std::move(v));        
        GreedyHillClimbing hc;
        return py::cast(hc.estimate_validation(df, pool, validation_score, m, arc_blacklist, arc_whitelist,
                                                type_whitelist, max_indegree, max_iters, epsilon, patience, verbose));
    }

    py::object call_hc_validated_gbn(const DataFrame& df, ArcVector arc_blacklist, ArcVector arc_whitelist, int max_indegree, int max_iters,
                               double epsilon, int patience, int verbose) 
    {
        using Model = GaussianNetwork;
        auto nodes = df.column_names();

        Model m(nodes, arc_whitelist);

        HoldoutLikelihood validation_score(df, 0.2);
        std::shared_ptr<Score> training_score = std::make_shared<CVLikelihood>(validation_score.training_data(), 10);

        auto arc_set = std::make_shared<ArcOperatorSet>(training_score);

        std::vector<std::shared_ptr<OperatorSet>> v {std::move(arc_set)};
        OperatorPool pool(m, training_score, std::move(v));
        
        GreedyHillClimbing hc;
        FactorTypeVector type_whitelist;
        return py::cast(hc.estimate_validation(df, pool, validation_score, m, arc_blacklist, arc_whitelist, 
                                                type_whitelist, max_indegree, max_iters, epsilon, patience, verbose));
    }

    py::object call_hc(const DataFrame& df, ArcVector arc_blacklist, ArcVector arc_whitelist, int max_indegree, int max_iters,
                 double epsilon, ScoreType score_type, int verbose) 
    {
        using Model = GaussianNetwork;
        auto nodes = df.column_names();

        Model m(nodes, arc_whitelist);

        GreedyHillClimbing hc;

        switch(score_type) {
            case ScoreType::BIC: {
                std::shared_ptr<Score> score = std::make_shared<BIC>(df);
                auto arc_set = std::make_shared<ArcOperatorSet>(score);
                std::vector<std::shared_ptr<OperatorSet>> v {std::move(arc_set)};
                OperatorPool pool(m, score, std::move(v));
                return py::cast(hc.estimate(df, pool, m, arc_blacklist, arc_whitelist, max_indegree, max_iters, epsilon, verbose));
            }
                break;
            case ScoreType::PREDICTIVE_LIKELIHOOD:
                throw std::invalid_argument("call_hc() cannot be called with a cross-validated score.");
            default:
                throw std::invalid_argument("Wrong Score type!");
        }   
    }
    
    // TODO: Include start model.
    // TODO: Include test ratio of holdout / number k-folds.
    py::object hc(const DataFrame& df, std::string bn_str, std::string score_str, std::vector<std::string> operators_str,
            ArcVector& arc_blacklist, ArcVector& arc_whitelist, FactorTypeVector& type_whitelist,
                  int max_indegree, int max_iters, double epsilon, int patience, int verbose) {
        
        auto bn_type = learning::algorithms::check_valid_bn_string(bn_str);
        auto score_type = learning::algorithms::check_valid_score_string(score_str);
        auto operators_type = learning::algorithms::check_valid_operators_string(operators_str);

        learning::algorithms::check_valid_score(bn_type, score_type);
        learning::algorithms::check_valid_operators(bn_type, operators_type);
        
        util::check_edge_list(df, arc_blacklist);
        util::check_edge_list(df, arc_whitelist);

        util::check_node_type_list(df, type_whitelist);

        if (max_iters == 0) max_iters = std::numeric_limits<int>::max();

        if (score_type == ScoreType::PREDICTIVE_LIKELIHOOD) {
            switch(bn_type) {
                case BayesianNetworkType::GBN: {
                    return call_hc_validated_gbn(df, arc_blacklist, arc_whitelist, max_indegree, max_iters,
                                            epsilon, patience, verbose);
                }
                case BayesianNetworkType::SPBN: {
                    return call_hc_validated_spbn(df, arc_blacklist, arc_whitelist, type_whitelist, 
                                            max_indegree, max_iters, epsilon, patience, verbose);
                }
                default:
                    throw std::invalid_argument("Wrong BayesianNetwork type!");
            }
        } else {
            return call_hc(df, arc_blacklist, arc_whitelist, max_indegree, max_iters,
                        epsilon, score_type, verbose);
        }
    }
}