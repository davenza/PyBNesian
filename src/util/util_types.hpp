#ifndef PYBNESIAN_UTIL_UTIL_TYPES_HPP
#define PYBNESIAN_UTIL_UTIL_TYPES_HPP

#include <vector>
#include <factors/factors.hpp>
#include <graph/graph_types.hpp>

using factors::NodeType;
using graph::Arc, graph::ArcHash, graph::Edge, graph::EdgeHash, graph::EdgeEqualTo;


namespace util {
    using ArcStringVector = std::vector<std::pair<std::string, std::string>>;
    using EdgeStringVector = std::vector<std::pair<std::string, std::string>>;
    using FactorStringTypeVector = std::vector<std::pair<std::string, NodeType>>;

    using ArcSet = std::unordered_set<Arc, ArcHash>;
    using EdgeSet = std::unordered_set<Edge, EdgeHash, EdgeEqualTo>;
}

#endif //PYBNESIAN_UTIL_UTIL_TYPES_HPP