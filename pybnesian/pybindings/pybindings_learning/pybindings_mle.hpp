#ifndef PYBNESIAN_PYBINDINGS_MLE_HPP
#define PYBNESIAN_PYBINDINGS_MLE_HPP

#include <pybind11/pybind11.h>
#include <factors/factors.hpp>

namespace py = pybind11;

using factors::FactorType;

namespace pybindings::learning::parameters {

py::object mle_python_wrapper(std::shared_ptr<FactorType>& n);

}

#endif  // PYBNESIAN_PYBINDINGS_MLE_HPP
