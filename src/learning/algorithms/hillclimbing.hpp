#ifndef PGM_DATASET_HILLCLIMBING_HPP
#define PGM_DATASET_HILLCLIMBING_HPP

#include <pybind11/pybind11.h>

#include <dataset/dataset.hpp>
#include <graph/dag.hpp>
#include <util/validate_dtype.hpp>


namespace py = pybind11; 

using dataset::DataFrame;
using graph::DagType;
using util::ArcVector;

namespace learning::algorithms {

    // DagType check_valid_dag_string(std::string& dag_type);
    // BayesianNetworkType check_valid_bn_string(std::string& bn_type);
    // ScoreType check_valid_score_string(std::string& score);
    // std::vector<OperatorSetType> check_valid_operators_string(std::vector<std::string>& operators);
    // void check_valid_score(BayesianNetworkType bn_type, ScoreType score);
    // void check_valid_operators(BayesianNetworkType bn_type, std::vector<OperatorSetType>& operators);


    // TODO: Include start graph.
    void hc(py::handle data, std::string bn_str, std::string score_str, std::vector<std::string> operators_str,
            std::vector<py::tuple> arc_blacklist, std::vector<py::tuple> arc_whitelist, std::vector<py::tuple> type_whitelist,
                  int max_indegree, int max_iters, double epsilon, int patience, std::string dag_type_str);

    class GreedyHillClimbing {

    public:
        template<typename OperatorPool, typename Model>
        void estimate(const DataFrame& df, OperatorPool& op_pool, int max_iters, double epsilon, const Model& start);

        template<typename OperatorPool, typename ValidationScore, typename Model>
        void estimate_validation(const DataFrame& df, 
                                 OperatorPool& op_pool, 
                                 ValidationScore& validation_score, 
                                 int max_iters,
                                 double epsilon, 
                                 int patience,
                                 const Model& start);
    };
}




#endif //PGM_DATASET_HILLCLIMBING_HPP