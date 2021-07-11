#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>
#include <arrow/python/platform.h>
#include <arrow/api.h>
#include <util/pickle.hpp>

#define STRINGIFY(x)       #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace pyarrow = arrow::py;

void pybindings_dataset(py::module& root);
void pybindings_kde(py::module& root);
void pybindings_factors(py::module& root);
void pybindings_graph(py::module& root);
void pybindings_models(py::module& root);
void pybindings_learning(py::module& root);

/*This module is needed to trick the MSVC linker, so a PyInit___init__() method exists.*/
#ifdef _MSC_VER
PYBIND11_MODULE(__init__, m) {}
#endif

PYBIND11_MODULE(pybnesian, m) {
    auto pa_version = py::module_::import("pyarrow").attr("__version__").cast<std::string>();

    std::string lib_pa_version{MACRO_STRINGIFY(PYARROW_VERSION_INFO)};
    if (pa_version != lib_pa_version) {
        throw std::runtime_error("PyBNesian was compiled with pyarrow " + lib_pa_version +
                                 ". To fix this error you can:\n"
                                 "1) Install pyarrow " +
                                 lib_pa_version +
                                 "."
                                 " Run \"pip install pyarrow==" +
                                 lib_pa_version +
                                 "\".\n"
                                 "2) If you want to keep pyarrow " +
                                 pa_version +
                                 " you"
                                 " must compile PyBNesian from source.");
    }

    pyarrow::import_pyarrow();

    m.doc() = R"doc(
- **PyBNesian** is a Python package that implements Bayesian networks. Currently, it is mainly dedicated to learning Bayesian networks.

- **PyBNesian** is implemented in C++, to achieve significant performance gains. It uses `Apache Arrow <https://arrow.apache.org>`_ to enable fast interoperability between Python and C++. In addition, some parts are implemented in OpenCL to achieve GPU acceleration.

- **PyBNesian** allows extending its functionality using Python code, so new research can be easily developed.

.. currentmodule:: pybnesian

)doc";  // optional module docstring

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#endif

    m.def("load", &util::load, py::arg("filename"), R"doc(
Load the saved object (a :class:`Factor <pybnesian.Factor>`, a graph, a :class:`BayesianNetworkBase <pybnesian.BayesianNetworkBase>`, etc...) in ``filename``.

:param filename: File name.
:returns: The object saved in the file.
)doc");

    pybindings_dataset(m);
    pybindings_kde(m);
    pybindings_factors(m);
    pybindings_graph(m);
    pybindings_models(m);
    pybindings_learning(m);
}