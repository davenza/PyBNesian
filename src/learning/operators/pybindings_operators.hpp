#ifndef PGM_DATASET_PYBINDINGS_OPERATORS_HPP
#define PGM_DATASET_PYBINDINGS_OPERATORS_HPP

#include <pybind11/pybind11.h>
#include <models/BayesianNetwork.hpp>


namespace py = pybind11;

using models::BayesianNetworkType;

namespace learning::operators {

    py::object addarc_wrapper_constructor(BayesianNetworkType bn, std::string source, std::string dest, double delta);
    py::object removearc_wrapper_constructor(BayesianNetworkType bn, std::string source, std::string dest, double delta);
    py::object fliparc_wrapper_constructor(BayesianNetworkType bn, std::string source, std::string dest, double delta);
    py::object changenodetype_wrapper_constructor(BayesianNetworkType bn, std::string node, FactorType new_factor, double delta);
}

#endif //PGM_DATASET_PYBINDINGS_OPERATORS_HPP