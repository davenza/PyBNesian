#include <iostream>
#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_dtype.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>

using namespace dataset;

using Eigen::VectorXd, Eigen::MatrixXd;;
using models::GaussianNetwork, models::BayesianNetworkType;
using learning::scores::BIC, learning::scores::ScoreType;
// learning::scores::CVLikelihood;
using graph::arc_vector, graph::DagType;
using learning::operators::ArcOperatorSet, learning::operators::OperatorPool, learning::operators::OperatorSetType;

// #include <util/benchmark_basic.hpp>

namespace learning::algorithms {

    DagType check_valid_dag_string(std::string& dag_type) {
        if (dag_type == "matrix") return DagType::MATRIX;
        if (dag_type == "list") return DagType::LIST;
        else
            throw std::invalid_argument("Wrong DAG type \"" + dag_type + "\"specified. The possible alternatives are " 
                                "\"matrix\" (Adjacency matrix) or \"list\" (Adjacency list).");
    }

    BayesianNetworkType check_valid_bn_string(std::string& bn_type) {
        if (bn_type == "gbn") return BayesianNetworkType::GBN;
        if (bn_type == "spbn") return BayesianNetworkType::SPBN;
        else
            throw std::invalid_argument("Wrong Bayesian Network type \"" + bn_type + "\" specified. The possible alternatives are " 
                                        "\"gbn\" (Gaussian Bayesian networks) or \"spbn\" (Semiparametric Bayesian networks).");
        }
    }

    ScoreType check_valid_score_string(std::string& score) {
        if (score == "bic") return ScoreType::BIC;
        if (score == "predic-l") return ScoreType::PREDICTIVE_LIKELIHOOD;
        else
            throw std::invalid_argument("Wrong Bayesian Network score \"" + score + "\"specified. The possible alternatives are " 
                                    "\"bic\" (Bayesian Information Criterion) or \"predic-l\" (Predicitive Log-likelihood).");
        
    }

    std::vector<OperatorSetType> check_valid_operators_string(std::vector<std::string>& operators) {
        std::vector<OperatorSetType> ops(operators.size());
        for (auto& op : operators) {
            if (op == "arcs") ops.push_back(OperatorSetType::ARCS);
            else if (op == "node_type") ops.push_back(OperatorSetType::NODE_TYPE);
            else
                throw std::invalid_argument("Wrong operator set \"" + op + "\". Valid choices are:"
                                            "\"arcs\" (Changes in arcs; addition, removal and flip) or "
                                            "\"node_type\" (Change of node type)");
        }
        return ops;
    }

    void check_valid_score(BayesianNetworkType bn_type, ScoreType score) {
        static std::unordered_map<BayesianNetworkType, std::unordered_set<ScoreType>> map_bn_score {
            { BayesianNetworkType::GBN, { ScoreType::BIC, ScoreType::PREDICTIVE_LIKELIHOOD } },
            { BayesianNetworkType::SPBN, { ScoreType::PREDICTIVE_LIKELIHOOD } }
        };
        if (map_bn_score[bn_type].count(score) == 0) {
            throw std::invalid_argument("Score \"" + score + "\" is not compabible with Bayesian network type \"" + bn_type + "\"");
        }
    }

    void check_valid_operators(BayesianNetworkType bn_type, std::vector<OperatorSetType>& operators) {
        static std::unordered_map<BayesianNetworkType, std::unordered_set<OperatorSetType>> map_bn_operators {
            { BayesianNetworkType::GBN, { OperatorSetType::ARCS }},
            { BayesianNetworkType::SPBN, { OperatorSetType::ARCS, OperatorSetType::NODE_TYPE }}
        };

        auto bn_set = map_bn_operators[bn_type];
        for (auto& op : operators) {
            if (bn_set.count(op) == 0) {
                // throw std::invalid_argument("Operator \"" + op + "\" is not compabible with Bayesian network type \"" + bn_type + "\"");
            }
        }
    }
    
    // TODO: Include start model.
    void hc(py::handle data, std::string bn_str, std::string score_str, std::vector<std::string> operators_str,
                  std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, 
                  int max_indegree, double epsilon, std::string dag_type_str) {
        
        auto dag_type = learning::algorithms::check_valid_dag_string(dag_type_str);
        auto bn_type = learning::algorithms::check_valid_bn_string(bn_str);
        auto score_type = learning::algorithms::check_valid_score_string(score_str);
        auto operators_type = learning::algorithms::check_valid_operators_string(operators_str);

        learning::algorithms::check_valid_score(bn_type, score_type);
        learning::algorithms::check_valid_operators(bn_type, operators_type);
        
        
        auto rb = dataset::to_record_batch(data);
        auto df = DataFrame(rb);

        auto blacklist_cpp = util::check_edge_list(df, blacklist);
        auto whitelist_cpp = util::check_edge_list(df, whitelist);

        auto nodes = df.column_names();



        // GreedyHillClimbing<GaussianNetwork<>> hc;

        // GaussianNetwork<> gbn = (whitelist_cpp.size() > 0) ? GaussianNetwork<>(nodes, whitelist_cpp) :
        //                                                      GaussianNetwork<>(nodes);

        // // // GaussianNetwork_L gbn = (whitelist_cpp.size() > 0) ? GaussianNetwork_L(nodes, whitelist_cpp) :
        // // //                                                    GaussianNetwork_L(nodes);

        // if (str_score == "bic") {
        //     BIC bic(df);
        // //     // ArcOperatorsType<GaussianNetwork, BIC<GaussianNetwork>> arc_op(df, gbn, whitelist_cpp, blacklist_cpp, max_indegree);
        //     // ArcOperatorSet arc_op(bic, gbn, whitelist_cpp, blacklist_cpp, max_indegree);

        //     // auto unique_arc_op = std::make_unique(arc_op);
        //     // auto arc_op = std::make_unique<ArcOperatorSet<auto, auto>>(bic, gbn, whitelist_cpp, blacklist_cpp, max_indegree);
        //     // auto arc_op = std::make_unique<ArcOperatorSet>(bic, gbn, whitelist_cpp, blacklist_cpp, max_indegree);

        // //     OperatorPool<GaussianNetwork<>> op_pool({arc_op});

        // // //     // BENCHMARK_PRE_SCOPE(10)
        // //     hc.estimate(op_pool, epsilon, gbn);
        // //     // BENCHMARK_POST_SCOPE(10)
        // }
        // // else if (str_score == "predictive-l") {
        // //     CVLikelihood cv(df, 10);

        // //     ArcOperatorsType arc_op(cv, gbn, whitelist_cpp, blacklist_cpp, max_indegree);
        // //     hc.estimate(arc_op, epsilon, gbn);
        // // }
        //  else {
        //     throw std::invalid_argument("Wrong score \"" + str_score + "\". Currently supported scores: \"bic\".");
        // }
    }

    
    template<typename Model>
    template<typename Operators>
    void GreedyHillClimbing<Model>::estimate(Operators& op,
                                             double epsilon,
                                             const Model& start) {


        // Model::requires(df);

        // auto current_model = start;
        // op.cache_scores(current_model);
        
        // while(true) {

        //     auto best_op = op.find_max(current_model);
        //     if (!best_op || best_op->delta() <= epsilon) {
        //         break;
        //     }

        //     best_op->apply(current_model);

        //     op.update_scores(best_op);
        //     // std::cout << "New op" << std::endl;
        // }

        // // std::cout << "Final score: " << op.score() << std::endl;
    }


}