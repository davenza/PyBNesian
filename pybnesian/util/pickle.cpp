#include <util/pickle.hpp>

namespace util {

py::object load(const std::string& name) {
    auto open = py::module::import("io").attr("open");
    auto file = open(name, "rb");
    auto bn = py::module::import("pickle").attr("load")(file);
    file.attr("close")();
    return bn;
}

}  // namespace util