#ifndef PGM_DATASET_VALIDATEDTYPE_HPP
#define PGM_DATASET_VALIDATEDTYPE_HPP

#include <pybind11/pybind11.h>
#include <dataset/dataset.hpp>
#include <graph/dag.hpp>
#include <models/SemiparametricBN_NodeType.hpp>
#include <util/util_types.hpp>

namespace py = pybind11;

using namespace dataset;
using models::NodeType;
using util::ArcVector, util::NodeTypeVector;

namespace util {


    void check_df(const DataFrame& df);

    ArcVector check_edge_list(const DataFrame& df, const std::vector<py::tuple>& list);
    NodeTypeVector check_node_type_list(const DataFrame& df, const std::vector<py::tuple>& list);

}

#endif //PGM_DATASET_VALIDATEDTYPE_HPP