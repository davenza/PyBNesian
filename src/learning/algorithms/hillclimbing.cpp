#include <iostream>
#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_dtype.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/operators/operators.hpp>

using namespace dataset;

using Eigen::VectorXd;
using models::GaussianNetwork;
using learning::scores::BIC;
using graph::arc_vector;

namespace learning::algorithms {




    
    // TODO: Include start model.
    void estimate(py::handle data, std::string str_score, 
                  std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, 
                  int max_indegree, double epsilon) {


        auto rb = dataset::to_record_batch(data);
        auto df = DataFrame(rb);

        auto blacklist_cpp = util::check_edge_list(df, blacklist);
        auto whitelist_cpp = util::check_edge_list(df, whitelist);

        GreedyHillClimbing<GaussianNetwork> hc;

        GaussianNetwork gbn = (whitelist_cpp.size() > 0) ? GaussianNetwork(df.column_names(), whitelist_cpp) :
                                                           GaussianNetwork(df.column_names());

        


        
        if (str_score == "bic") {
            BIC<GaussianNetwork> score;
            hc.estimate(df, score, blacklist_cpp, whitelist_cpp, max_indegree, epsilon, gbn);
        }
         else {
            throw std::invalid_argument("Wrong score \"" + str_score + "\". Currently supported scores: \"bic\".");
         }
    }
    
    template<typename Model>
    template<typename Score>
    void GreedyHillClimbing<Model>::estimate(const DataFrame& df, 
                                              Score score,
                                              arc_vector blacklist, 
                                              arc_vector whitelist, 
                                              int max_indegree, 
                                              double epsilon,
                                              const Model& start) {


        Model::requires(df);

    }


}