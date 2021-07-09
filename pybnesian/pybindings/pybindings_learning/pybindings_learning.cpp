#include <pybind11/pybind11.h>

namespace py = pybind11;

void pybindings_scores(py::module& root);
void pybindings_independence_tests(py::module& root);
void pybindings_parameters(py::module& root);
void pybindings_operators(py::module& root);
void pybindings_algorithms(py::module& root);

void pybindings_learning(py::module& root) {
    pybindings_scores(root);
    pybindings_independence_tests(root);
    pybindings_parameters(root);
    pybindings_operators(root);
    pybindings_algorithms(root);
}