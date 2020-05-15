#ifndef PGM_DATASET_VALIDATEDTYPE_HPP
#define PGM_DATASET_VALIDATEDTYPE_HPP

#include <pybind11/pybind11.h>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>

namespace py = pybind11;

using namespace dataset;
using graph::arc_vector;

namespace util {

    void check_df(const DataFrame& df);

    arc_vector check_edge_list(const DataFrame& df, const std::vector<py::tuple>& list);

}

#endif //PGM_DATASET_VALIDATEDTYPE_HPP