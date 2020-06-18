#ifndef PGM_DATASET_HILLCLIMBING_HPP
#define PGM_DATASET_HILLCLIMBING_HPP

#include <pybind11/pybind11.h>

#include <dataset/dataset.hpp>
#include <graph/dag.hpp>

#include <learning/operators/operators.hpp>


namespace py = pybind11; 

using dataset::DataFrame;
using graph::arc_vector, graph::DagType;
using models::BayesianNetworkType;
using learning::scores::ScoreType;
using learning::operators::OperatorSetType;

namespace learning::algorithms {

    // DagType check_valid_dag_string(std::string& dag_type);
    // BayesianNetworkType check_valid_bn_string(std::string& bn_type);
    // ScoreType check_valid_score_string(std::string& score);
    // std::vector<OperatorSetType> check_valid_operators_string(std::vector<std::string>& operators);
    // void check_valid_score(BayesianNetworkType bn_type, ScoreType score);
    // void check_valid_operators(BayesianNetworkType bn_type, std::vector<OperatorSetType>& operators);


    // TODO: Include start graph.
    void hc(py::handle data, std::string bn_type, std::string score, std::vector<std::string> operators, 
            std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, 
                int max_indegree, double epsilon, std::string dag_type);

    template<typename Model>
    class GreedyHillClimbing {

    public:
        template<typename Operators>
        void estimate(Operators& op_pool, double epsilon, const Model& start);
    };
}




#endif //PGM_DATASET_HILLCLIMBING_HPP