#ifndef PGM_DATASET_VALIDATEDTYPE_HPP
#define PGM_DATASET_VALIDATEDTYPE_HPP

#include <pybind11/pybind11.h>
#include <dataset/dataset.hpp>

namespace py = pybind11;

using namespace dataset;

namespace util {

    void check_df(const DataFrame& df);
    void check_edge_list(const DataFrame& df, const std::vector<py::tuple>& list);

}

#endif //PGM_DATASET_VALIDATEDTYPE_HPP