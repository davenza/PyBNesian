#ifndef PGM_DATASET_VALIDATE_DTYPE_HPP
#define PGM_DATASET_VALIDATE_DTYPE_HPP

#include <pybind11/pybind11.h>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>
#include <factors/factors.hpp>
#include <util/util_types.hpp>

namespace py = pybind11;

using namespace dataset;
using factors::FactorType;
using util::ArcVector, util::FactorTypeVector;

namespace util {


    void check_df(const DataFrame& df);

    ArcVector check_edge_list(const DataFrame& df, const std::vector<py::tuple>& list);
    FactorTypeVector check_node_type_list(const DataFrame& df, const std::vector<py::tuple>& list);

}

#endif //PGM_DATASET_VALIDATE_DTYPE_HPP