#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>
#include <arrow/python/platform.h>
#include <arrow/api.h>

#define STRINGIFY(x)       #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

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

/*This module is needed to trick the MSVC linker, so a PyInit___init__() method exists.*/
#ifdef _MSC_VER
PYBIND11_MODULE(__init__, m) {}
#endif

PYBIND11_MODULE(pybnesian, m) {
    pyarrow::import_pyarrow();

    m.doc() = "pybind11 data plugin";  // optional module docstring

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#endif

    m.def("load", &load);

    pybindings_dataset(m);
    pybindings_factors(m);
    pybindings_graph(m);
    pybindings_models(m);
    pybindings_learning(m);
}