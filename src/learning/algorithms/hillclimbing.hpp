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
    void hc(py::handle data, std::string score, std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, int max_indegree, double epsilon);

    template<typename Model>
    class GreedyHillClimbing {

    public:
        template<typename Operators>
        void estimate(Operators& op_pool, double epsilon, const Model& start);
    };
}




#endif //PGM_DATASET_HILLCLIMBING_HPP