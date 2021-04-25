#ifndef PYBNESIAN_UTIL_UTIL_TYPES_HPP
#define PYBNESIAN_UTIL_UTIL_TYPES_HPP

#include <optional>
#include <vector>
#include <factors/factors.hpp>
#include <graph/graph_types.hpp>

using factors::FactorType;
using graph::Arc, graph::ArcHash, graph::Edge, graph::EdgeHash, graph::EdgeEqualTo;

namespace util {

using ArcStringVector = std::vector<std::pair<std::string, std::string>>;
using EdgeStringVector = std::vector<std::pair<std::string, std::string>>;
using FactorTypeVector = std::vector<std::pair<std::string, std::shared_ptr<FactorType>>>;

using ArcSet = std::unordered_set<Arc, ArcHash>;
using EdgeSet = std::unordered_set<Edge, EdgeHash, EdgeEqualTo>;

class random_seed_arg {
public:
    random_seed_arg() : m_value(std::random_device{}()) {}
    random_seed_arg(unsigned int arg) : m_value(arg) {}
    random_seed_arg(std::optional<unsigned int> arg) {
        if (arg) {
            m_value = *arg;
        } else {
            m_value = std::random_device{}();
        }
    }

    operator unsigned int() { return m_value; }

private:
    unsigned int m_value;
};

}  // namespace util

#endif  // PYBNESIAN_UTIL_UTIL_TYPES_HPP