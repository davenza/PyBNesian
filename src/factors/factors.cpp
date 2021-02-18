#include <factors/factors.hpp>

namespace factors{
    py::object load_factor(const std::string& name) {
        auto open = py::module::import("io").attr("open");
        auto file = open(name, "rb");
        auto graph = py::module::import("pickle").attr("load")(file);
        file.attr("close")();
        return graph;
    }
}