#include <iostream>
#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_dtype.hpp>

#include <dataset/dataset.hpp>

using namespace dataset;

using Eigen::VectorXd;

namespace learning::algorithms {



    
    void estimate(py::handle data, double score, std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, int max_indegree, double epsilon) {

        GreedyHillClimbing<int> hc;
        
        hc.estimate<double>(data, score, blacklist, whitelist, max_indegree, epsilon);
    }
    
    template<typename Model>
    template<typename Score>
    Model GreedyHillClimbing<Model>::estimate(py::handle data, 
                                              Score score, 
                                              std::vector<py::tuple> blacklist, 
                                              std::vector<py::tuple> whitelist, 
                                              int max_indegree, 
                                              double epsilon) {
        auto rb = dataset::to_record_batch(data);
        auto df = DataFrame(rb);

        // TODO: Check names not repeated in df.
        util::check_edge_list(df, blacklist);
        util::check_edge_list(df, whitelist);


    }


}