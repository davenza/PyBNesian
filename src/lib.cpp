#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>

namespace py = pybind11;
namespace pyarrow = arrow::py;

void pybindings_dataset(py::module& root);
void pybindings_factors(py::module& root);
void pybindings_graph(py::module& root);
void pybindings_models(py::module& root);
void pybindings_learning(py::module& root);
void pybindings_distributions(py::module& root);


PYBIND11_MODULE(pgm_dataset, m) {
    pyarrow::import_pyarrow();

    m.doc() = "pybind11 data plugin"; // optional module docstring

    pybindings_dataset(m);
    pybindings_factors(m);
    pybindings_graph(m);
    pybindings_models(m);
    pybindings_learning(m);
    pybindings_distributions(m);
}