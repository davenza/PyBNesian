#include <util/validate_whitelists.hpp>

using graph::PartiallyDirectedGraph;

namespace util {

void check_arc_list(const DataFrame& df, const ArcStringVector& list) {
    auto schema = df->schema();

    for (auto pair : list) {
        if (!schema->GetFieldByName(pair.first))
            throw std::invalid_argument("Node " + pair.first + " not present in the data set.");

        if (!schema->GetFieldByName(pair.second))
            throw std::invalid_argument("Node " + pair.second + " not present in the data set.");
    }
}

void check_edge_list(const DataFrame& df, const EdgeStringVector& list) { check_arc_list(df, list); }

void check_node_type_list(const DataFrame& df, const FactorTypeVector& list) {
    auto schema = df->schema();

    for (auto pair : list) {
        if (!schema->GetFieldByName(pair.first))
            throw std::invalid_argument("Node " + pair.first + " not present in the data set.");
    }
}

}  // namespace util
