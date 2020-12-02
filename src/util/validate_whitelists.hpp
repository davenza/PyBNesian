#ifndef PYBNESIAN_UTIL_VALIDATE_WHITELISTS_HPP
#define PYBNESIAN_UTIL_VALIDATE_WHITELISTS_HPP

#include <dataset/dataset.hpp>
#include <graph/generic_graph.hpp>
#include <util/util_types.hpp>

using dataset::DataFrame;
using util::ArcSet, util::EdgeSet;
using graph::PartiallyDirectedGraph;

namespace util {

    struct ListRestrictions {
        ArcSet arc_blacklist;
        ArcSet arc_whitelist;
        EdgeSet edge_blacklist;
        EdgeSet edge_whitelist;
    };

    ListRestrictions validate_restrictions(const PartiallyDirectedGraph& g,
                                      const ArcStringVector& varc_blacklist, 
                                      const ArcStringVector& varc_whitelist,
                                      const EdgeStringVector& vedge_blacklist,
                                      const EdgeStringVector& vedge_whitelist);

    void check_arc_list(const PartiallyDirectedGraph& g, const ArcStringVector& list);
    void check_edge_list(const PartiallyDirectedGraph& g, const EdgeStringVector& list);
    void check_node_type_list(const PartiallyDirectedGraph& g, const FactorStringTypeVector& list);

    void check_arc_list(const DataFrame& df, const ArcStringVector& list);
    void check_edge_list(const DataFrame& df, const EdgeStringVector& list);
    void check_node_type_list(const DataFrame& df, const FactorStringTypeVector& list);
}

#endif //PYBNESIAN_UTIL_VALIDATE_WHITELISTS_HPP