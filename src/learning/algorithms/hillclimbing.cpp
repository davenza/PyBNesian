#include <iostream>
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
using graph::DagType;
using learning::operators::ArcOperatorSet, learning::operators::OperatorPool, learning::operators::OperatorSetType;
using util::ArcVector;

// #include <util/benchmark_basic.hpp>

namespace learning::algorithms {

    using OperatorSetTypeS = std::unordered_set<OperatorSetType, typename OperatorSetType::HashType>;

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

    ScoreType check_valid_score_string(std::string& score) {
        if (score == "bic") return ScoreType::BIC;
        if (score == "predic-l") return ScoreType::PREDICTIVE_LIKELIHOOD;
        else
            throw std::invalid_argument("Wrong Bayesian Network score \"" + score + "\"specified. The possible alternatives are " 
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

    // template<typename Model, typename Score>
    // void call_hc(Model& model, Score& score, const arc_vector& blacklist, const arc_vector& whitelist, 
    //              int max_indegree, double epsilon, std::vector<OperatorSetType>& opset_type) {

    //     GreedyHillClimbing hc;
    //     if (opset_type.size() == 1) {
    //         switch(opset_type[0]) {
    //             case OperatorSetType::ARCS:
    //                 ArcOperatorSet op(score, model, whitelist, blacklist, max_indegree);
    //                 hc.estimate(op, epsilon, model);
    //                 break;
    //             case OperatorSetType::NODE_TYPE:
    //                 break;
    //                 // ArcOperatorSet op(score, model, whitelist, blacklist, max_indegree);
    //         }
    //     }
    // }


    // template<typename Model, typename ...Args>
    // void call_hc(const DataFrame& df, Model& model, const arc_vector& blacklist, const arc_vector& whitelist,
    //              int max_indegree, double epsilon, ScoreType& score_type, Args... args);

    // template<typename D, typename ...Args>
    // void call_hc(const DataFrame& df, GaussianNetwork<D>& model, const arc_vector& blacklist, const arc_vector& whitelist,
    //              int max_indegree, double epsilon, ScoreType& score_type, Args... args) {

    //              }

    // template<typename D, typename ...Args>
    // void call_hc(const DataFrame& df, SemiparametricBN<D>& model, const arc_vector& blacklist, const arc_vector& whitelist,
    //              int max_indegree, double epsilon, ScoreType& score_type, Args... args) {
                     
    //              }


    // template<typename DagType, typename ...Args>
    // void call_hc(const DataFrame& df, std::vector<std::string> nodes, const arc_vector& blacklist, const arc_vector& whitelist,
    //              int max_indegree, double epsilon, BayesianNetworkType& bn_type, Args... args) {
    //     switch(bn_type) {
    //         case BayesianNetworkType::GBN: {
    //             using ModelType = GaussianNetwork<DagType>;
    //             ModelType m = (whitelist.size() > 0) ? ModelType(nodes, whitelist) : ModelType(nodes);
    //             call_hc(df, m, blacklist, whitelist, max_indegree, epsilon, args...);
    //         }
    //             break;
    //         case BayesianNetworkType::SPBN: {
    //             using ModelType = SemiparametricBN<DagType>;
    //             ModelType m = (whitelist.size() > 0) ? ModelType(nodes, whitelist) : ModelType(nodes);
    //             call_hc(df, m, blacklist, whitelist, max_indegree, epsilon, args...);
    //         }
    //             break;
    //     }
    // }

    template<typename DagType>
    void call_hc_validated_spbn(const DataFrame& df, 
                               ArcVector arc_blacklist, 
                               ArcVector arc_whitelist, 
                               NodeTypeVector type_whitelist, 
                               int max_indegree, 
                               double epsilon,
                               ScoreType score_type, 
                               OperatorSetTypeS operators) 
    {
        using Model = SemiparametricBN<DagType>;
        auto nodes = df.column_names();

        Model m = Model(nodes, arc_whitelist, type_whitelist);

        HoldoutLikelihood validation_score(df, 0.2);
        CVLikelihood training_score(validation_score.training_data(), 10);

        OperatorPool pool(m, training_score, operators, arc_blacklist, arc_whitelist, type_whitelist, max_indegree);
        
        GreedyHillClimbing hc;
        hc.estimate_validation(df, pool, validation_score, epsilon, m);
    }

    void call_hc_validated_spbn(const DataFrame& df, 
                                ArcVector arc_blacklist, 
                                ArcVector arc_whitelist,
                                NodeTypeVector type_whitelist,
                                int max_indegree, 
                                double epsilon, 
                                DagType dag_type, 
                                ScoreType score_type, 
                                OperatorSetTypeS operators) 
    {
        switch(dag_type) {
            case DagType::MATRIX:
                call_hc_validated_spbn<AdjMatrixDag>(df, arc_blacklist, arc_whitelist, type_whitelist,
                                                     max_indegree, epsilon, score_type, operators);
                break;
            case DagType::LIST:
                call_hc_validated_spbn<AdjListDag>(df, arc_blacklist, arc_whitelist, type_whitelist, 
                                                   max_indegree, epsilon, score_type, operators);
                break;
        }
    }

    template<typename DagType>
    void call_hc_validated_gbn(const DataFrame& df, ArcVector arc_blacklist, ArcVector arc_whitelist, int max_indegree, double epsilon,
                               ScoreType score_type, OperatorSetTypeS operators) 
    {
        using Model = GaussianNetwork<DagType>;
        auto nodes = df.column_names();

        Model m(nodes, arc_whitelist);

        HoldoutLikelihood validation_score(df, 0.2);
        CVLikelihood training_score(validation_score.training_data(), 10);

        OperatorPool pool(m, training_score, operators, arc_blacklist, arc_whitelist, {}, max_indegree);
        
        GreedyHillClimbing hc;
        hc.estimate_validation(df, pool, validation_score, epsilon, m);
    }

    void call_hc_validated_gbn(const DataFrame& df, ArcVector arc_blacklist, ArcVector arc_whitelist, int max_indegree, double epsilon,
                               DagType dag_type, ScoreType score_type, OperatorSetTypeS operators) 
    {
        switch(dag_type) {
            case DagType::MATRIX:
                call_hc_validated_gbn<AdjMatrixDag>(df, arc_blacklist, arc_whitelist, max_indegree, epsilon, score_type, operators);
                break;
            case DagType::LIST:
                call_hc_validated_gbn<AdjListDag>(df, arc_blacklist, arc_whitelist, max_indegree, epsilon, score_type, operators);
                break;
        }
    }
    
    // TODO: Include start model.
    // TODO: Include test ratio of holdout / number k-folds.
    void hc(py::handle data, std::string bn_str, std::string score_str, std::vector<std::string> operators_str,
            std::vector<py::tuple> arc_blacklist, std::vector<py::tuple> arc_whitelist, std::vector<py::tuple> type_whitelist,
                  int max_indegree, double epsilon, std::string dag_type_str) {
        
        auto dag_type = learning::algorithms::check_valid_dag_string(dag_type_str);
        auto bn_type = learning::algorithms::check_valid_bn_string(bn_str);
        auto score_type = learning::algorithms::check_valid_score_string(score_str);
        auto operators_type = learning::algorithms::check_valid_operators_string(operators_str);

        learning::algorithms::check_valid_score(bn_type, score_type);
        learning::algorithms::check_valid_operators(bn_type, operators_type);
        
        auto rb = dataset::to_record_batch(data);
        auto df = DataFrame(rb);

        auto arc_blacklist_cpp = util::check_edge_list(df, arc_blacklist);
        auto arc_whitelist_cpp = util::check_edge_list(df, arc_whitelist);

        auto node_type_whitelist = util::check_node_type_list(df, type_whitelist);

        if (score_type == ScoreType::PREDICTIVE_LIKELIHOOD) {
            switch(bn_type) {
                case BayesianNetworkType::GBN:
                    call_hc_validated_gbn(df, arc_blacklist_cpp, arc_whitelist_cpp, max_indegree, epsilon, dag_type, score_type, operators_type);
                    break;
                // case BayesianNetworkType::SPBN:
            }
        } else {

        }

        // switch(op) {
        //     case BayesianNetworkType::GBN:
        //         // call_hc_gbn();
        //     case BayesianNetworkType::SPBN:
        // };

    }
    

    
    template<typename Operators, typename Model>
    void GreedyHillClimbing::estimate(const DataFrame& df,
                                      Operators& op,
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

    template<typename OperatorPool, typename ValidationScore, typename Model>
    void estimate_validation(const DataFrame& df,
                             OperatorPool& op_pool, 
                             ValidationScore& validation_score, 
                             double epsilon, 
                             const Model& start) {
        Model::requires(df);

        auto current_model = start;
        auto best_model = start;

        VectorXd local_validation(current_model.num_nodes());
        for (auto n = 0; n < current_model.num_nodes(); ++n) {
            local_validation(n) = validation_score.local_score(current_model, n);
        }
        auto best_validation_score = local_validation.sum();

        op_pool.cache_scores(current_model);
        
        while(true) {
            auto best_op = op_pool.find_max(current_model);
            if (!best_op || best_op->delta() <= epsilon) {
                break;
            }

            // if (validation)

            // if(validation_score.local_score())
            best_op->apply(current_model);
            op_pool.update_scores(current_model, best_op);
            // std::cout << "New op" << std::endl;
        }

        // std::cout << "Final score: " << op.score() << std::endl;
    }


}