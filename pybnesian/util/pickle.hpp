#ifndef PYBNESIAN_UTIL_PICKLE_HPP
#define PYBNESIAN_UTIL_PICKLE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace util {

template <typename OBJ>
void save_object(const OBJ& obj, std::string name) {
    auto open = py::module_::import("io").attr("open");

    if (name.size() < 7 || name.substr(name.size() - 7) != ".pickle") name += ".pickle";

    auto file = open(name, "wb");
    py::module_::import("pickle").attr("dump")(py::cast(&obj), file, 2);
    file.attr("close")();
}

py::object load(const std::string& name);

}  // namespace util

#endif  // PYBNESIAN_UTIL_PICKLE_HPP