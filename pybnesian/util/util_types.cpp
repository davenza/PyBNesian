#include <util/util_types.hpp>

namespace util {

FactorTypeVector& keep_FactorTypeVector_python_alive(FactorTypeVector& v) {
    for (auto& f : v) {
        FactorType::keep_python_alive(f.second);
    }

    return v;
}

FactorTypeVector keep_FactorTypeVector_python_alive(const FactorTypeVector& v) {
    FactorTypeVector fv;
    fv.reserve(v.size());

    for (const auto& f : v) {
        fv.push_back({f.first, FactorType::keep_python_alive(f.second)});
    }

    return fv;
}

}  // namespace util