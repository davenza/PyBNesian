#ifndef PGM_DATASET_HILLCLIMBING_HPP
#define PGM_DATASET_HILLCLIMBING_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11; 

namespace learning::algorithms {


    void estimate(py::handle data, double score, std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, int max_indegree, double epsilon);

    template<typename Model>
    class GreedyHillClimbing {

    public:
        template<typename Score>
        Model estimate(py::handle data, Score score, std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, int max_indegree, double epsilon);
    };
}




#endif //PGM_DATASET_HILLCLIMBING_HPP