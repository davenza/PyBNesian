#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>

namespace py = pybind11;
namespace pyarrow = arrow::py;

void pybindings_dataset(py::module& root);
void pybindings_factors(py::module& root);
void pybindings_graph(py::module& root);
void pybindings_models(py::module& root);
void pybindings_learning(py::module& root);

py::object load(const std::string& name) {
    auto open = py::module::import("io").attr("open");
    auto file = open(name, "rb");
    auto bn = py::module::import("pickle").attr("load")(file);
    file.attr("close")();
    return bn;
}

PYBIND11_MODULE(pybnesian, m) {
    pyarrow::import_pyarrow();

    m.doc() = "pybind11 data plugin"; // optional module docstring

    m.def("load", &load);

    
    pybindings_dataset(m);
    pybindings_factors(m);
    pybindings_graph(m);
    pybindings_models(m);
    pybindings_learning(m);
}