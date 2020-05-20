#ifndef PGM_DATASET_HILLCLIMBING_HPP
#define PGM_DATASET_HILLCLIMBING_HPP

#include <pybind11/pybind11.h>

#include <dataset/dataset.hpp>
#include <graph/dag.hpp>


namespace py = pybind11; 

using dataset::DataFrame;
using graph::arc_vector;

namespace learning::algorithms {


    // TODO: Include start graph.
    void estimate(py::handle data, std::string score, std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, int max_indegree, double epsilon);

    template<typename Model>
    class GreedyHillClimbing {

    public:
        template<typename Operators>
        void estimate(const DataFrame& data, Operators op_pool, arc_vector blacklist, 
                      arc_vector whitelist, int max_indegree, double epsilon, const Model& start);
    };
}




#endif //PGM_DATASET_HILLCLIMBING_HPP