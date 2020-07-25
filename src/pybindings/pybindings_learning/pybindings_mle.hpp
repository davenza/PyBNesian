#ifndef PGM_DATASET_PYBINDINGS_MLE_HPP
#define PGM_DATASET_PYBINDINGS_MLE_HPP

#include <pybind11/pybind11.h>
#include <factors/factors.hpp>

namespace py = pybind11;

using factors::FactorType;

namespace pybindings::learning::parameters {

    py::object mle_python_wrapper(FactorType n);
}

#endif //PGM_DATASET_PYBINDINGS_MLE_HPP
