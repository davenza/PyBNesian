#include <pybind11/pybind11.h>

namespace py = pybind11;

void pybindings_scores(py::module& root);
void pybindings_independence_tests(py::module& root);
void pybindings_parameters(py::module& root);
void pybindings_operators(py::module& root);
void pybindings_algorithms(py::module& root);

void pybindings_learning(py::module& root) {
    auto learning = root.def_submodule("learning", R"doc(The pybnesian.learning module implements different types to
learn Bayesian networks from data. This includes the parameter learning and the structure learning.
)doc");

    pybindings_scores(learning);
    pybindings_independence_tests(learning);
    pybindings_parameters(learning);
    pybindings_operators(learning);
    pybindings_algorithms(learning);
}