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
using learning::operators::ArcOperatorSet, learning::operators::OperatorPool, learning::operators::OperatorSetType,
      learning::operators::OperatorTabuSet;
using util::ArcVector;

// #include <util/benchmark_basic.hpp>

namespace learning::algorithms {

    using OperatorSetTypeS = std::unordered_set<OperatorSetType, typename OperatorSetType::HashType>;

    DagType check_valid_dag_string(std::string& dag_type) {
        if (dag_type == "matrix") return DagType::MATRIX;
        if (dag_type == "list") return DagType::LIST;
        else
            throw std::invalid_argument("Wrong DAG type \"" + dag_type + "\" specified. The possible alternatives are " 
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

    template<typename DagType>
    void call_hc_validated_spbn(const DataFrame& df, 
                               ArcVector arc_blacklist, 
                               ArcVector arc_whitelist, 
                               FactorTypeVector type_whitelist, 
                               int max_indegree,
                               int max_iters,
                               double epsilon,
                               int patience,
                               OperatorSetTypeS operators) 
    {
        using Model = SemiparametricBN<DagType>;
        auto nodes = df.column_names();

        Model m = Model(nodes, arc_whitelist, type_whitelist);

        HoldoutLikelihood validation_score(df, 0.2, 0);
        CVLikelihood training_score(validation_score.training_data(), 10, 0);

        OperatorPool pool(m, training_score, operators, arc_blacklist, arc_whitelist, type_whitelist, max_indegree);
        
        GreedyHillClimbing hc;
        hc.estimate_validation(df, pool, validation_score, max_iters, epsilon, patience, m);
    }

    void call_hc_validated_spbn(const DataFrame& df, 
                                ArcVector arc_blacklist, 
                                ArcVector arc_whitelist,
                                FactorTypeVector type_whitelist,
                                int max_indegree, 
                                int max_iters,
                                double epsilon,
                                int patience,
                                DagType dag_type,
                                OperatorSetTypeS operators) 
    {
        switch(dag_type) {
            case DagType::MATRIX:
                call_hc_validated_spbn<AdjMatrixDag>(df, arc_blacklist, arc_whitelist, type_whitelist,
                                                     max_indegree, max_iters, epsilon, patience, operators);
                break;
            case DagType::LIST:
                call_hc_validated_spbn<AdjListDag>(df, arc_blacklist, arc_whitelist, type_whitelist, 
                                                   max_indegree, max_iters, epsilon, patience, operators);
                break;
        }
    }

    template<typename DagType>
    void call_hc_validated_gbn(const DataFrame& df, ArcVector arc_blacklist, ArcVector arc_whitelist, int max_indegree, int max_iters,
                               double epsilon, int patience, OperatorSetTypeS operators) 
    {
        using Model = GaussianNetwork<DagType>;
        auto nodes = df.column_names();

        Model m(nodes, arc_whitelist);

        HoldoutLikelihood validation_score(df, 0.2);
        CVLikelihood training_score(validation_score.training_data(), 10);

        OperatorPool pool(m, training_score, operators, arc_blacklist, arc_whitelist, max_indegree);
        
        GreedyHillClimbing hc;
        hc.estimate_validation(df, pool, validation_score, max_iters, epsilon, patience, m);
    }

    void call_hc_validated_gbn(const DataFrame& df, ArcVector arc_blacklist, ArcVector arc_whitelist, int max_indegree, int max_iters,
                               double epsilon, int patience, DagType dag_type, OperatorSetTypeS& operators) 
    {
        switch(dag_type) {
            case DagType::MATRIX:
                call_hc_validated_gbn<AdjMatrixDag>(df, arc_blacklist, arc_whitelist, max_indegree, max_iters, epsilon, patience, operators);
                break;
            case DagType::LIST:
                call_hc_validated_gbn<AdjListDag>(df, arc_blacklist, arc_whitelist, max_indegree, max_iters, epsilon, patience, operators);
                break;
        }
    }

    template<typename DagType>
    void call_hc(const DataFrame& df, ArcVector arc_blacklist, ArcVector arc_whitelist, int max_indegree, int max_iters,
                 double epsilon, ScoreType score_type, OperatorSetTypeS& operators) 
    {
        using Model = GaussianNetwork<DagType>;
        auto nodes = df.column_names();

        Model m(nodes, arc_whitelist);

        GreedyHillClimbing hc;

        switch(score_type) {
            case ScoreType::BIC: {
                BIC score(df);
                OperatorPool pool(m, score, operators, arc_blacklist, arc_whitelist, max_indegree);
                hc.estimate(df, pool, max_iters, epsilon, m);
            }
                break;
            case ScoreType::PREDICTIVE_LIKELIHOOD:
                std::invalid_argument("call_hc() cannot be called with a cross-validated score.");

        }   
    }

    void call_hc(const DataFrame& df, ArcVector arc_blacklist, ArcVector arc_whitelist, int max_indegree,
                 int max_iters, double epsilon, DagType dag_type, ScoreType score_type, OperatorSetTypeS operators) 
    {
        switch(dag_type) {
            case DagType::MATRIX:
                call_hc<AdjMatrixDag>(df, arc_blacklist, arc_whitelist, max_indegree, max_iters, epsilon, score_type, operators);
                break;
            case DagType::LIST:
                call_hc<AdjListDag>(df, arc_blacklist, arc_whitelist, max_indegree, max_iters, epsilon, score_type, operators);
                break;
        }
    }
    
    // TODO: Include start model.
    // TODO: Include test ratio of holdout / number k-folds.
    void hc(py::handle data, std::string bn_str, std::string score_str, std::vector<std::string> operators_str,
            std::vector<py::tuple> arc_blacklist, std::vector<py::tuple> arc_whitelist, std::vector<py::tuple> type_whitelist,
                  int max_indegree, int max_iters, double epsilon, int patience, std::string dag_type_str) {
        
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

        auto type_whitelist_cpp = util::check_node_type_list(df, type_whitelist);

        if (max_iters == 0) max_iters = std::numeric_limits<int>::max();

        if (score_type == ScoreType::PREDICTIVE_LIKELIHOOD) {
            switch(bn_type) {
                case BayesianNetworkType::GBN:
                    call_hc_validated_gbn(df, arc_blacklist_cpp, arc_whitelist_cpp, max_indegree, max_iters,
                                            epsilon, patience, dag_type, operators_type);
                    break;
                case BayesianNetworkType::SPBN:
                    call_hc_validated_spbn(df, arc_blacklist_cpp, arc_whitelist_cpp, type_whitelist_cpp, 
                                            max_indegree, max_iters, epsilon, patience, dag_type, operators_type);
            }
        } else {
            call_hc(df, arc_blacklist_cpp, arc_whitelist_cpp, max_indegree, max_iters,
                        epsilon, dag_type, score_type, operators_type);
        }
    }
    

    
    template<typename OperatorPool, typename Model>
    void GreedyHillClimbing::estimate(const DataFrame& df,
                                      OperatorPool& op,
                                      int max_iters,
                                      double epsilon,
                                      const Model& start) {


        Model::requires(df);

        auto current_model = start;
        op.cache_scores(current_model);
        
        auto iter = 0;
        while(iter < max_iters) {

            auto best_op = op.find_max(current_model);
            if (!best_op || best_op->delta() <= epsilon) {
                break;
            }

            best_op->apply(current_model);
            op.update_scores(current_model, best_op.get());
        //     // std::cout << "New op" << std::endl;
            ++iter;
        }

        std::cout << "Final score: " << op.score() << std::endl;
        std::cout << "Final model: " << current_model << std::endl;
    }

    template<typename OperatorPool, typename ValidationScore, typename Model>
    void GreedyHillClimbing::estimate_validation(const DataFrame& df,
                             OperatorPool& op_pool, 
                             ValidationScore& validation_score, 
                             int max_iters,
                             double epsilon, 
                             int patience,
                             const Model& start) {
        Model::requires(df);

        auto current_model = start;
        auto best_model = start;

        VectorXd local_validation(current_model.num_nodes());
        for (auto n = 0; n < current_model.num_nodes(); ++n) {
            local_validation(n) = validation_score.local_score(current_model, n);
        }

        op_pool.cache_scores(current_model);
        int p = 0;
        double validation_offset = 0;

        OperatorTabuSet<Model> tabu_set(current_model);
        
        auto iter = 0;
        while(iter < max_iters) {
            auto best_op = op_pool.find_max(current_model, tabu_set);
            if (!best_op || best_op->delta() <= epsilon) {
                break;
            }

            std::cout << "Best op: " << best_op->ToString(current_model) << std::endl;

            double validation_delta = validation_score.delta_score(current_model, best_op.get(), local_validation);
            
            best_op->apply(current_model);
            if ((validation_delta + validation_offset) > 0) {
                p = 0;
                validation_offset = 0;
                best_model = current_model;
                tabu_set.clear();
            } else {
                if (++p >= patience)
                    break;
                validation_offset += validation_delta;
                tabu_set.insert(best_op->opposite());
            }

            op_pool.update_scores(current_model, best_op.get());
            ++iter;
        }

        std::cout << "Final score: " << op_pool.score(best_model) << std::endl;
        std::cout << "Validation score fun: " << validation_score.score(best_model) << std::endl;
        std::cout << "Final model: " << best_model << std::endl;
    }


}