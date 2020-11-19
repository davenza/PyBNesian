#ifndef PYBNESIAN_UTIL_VALIDATE_WHITELISTS_HPP
#define PYBNESIAN_UTIL_VALIDATE_WHITELISTS_HPP

#include <util/util_types.hpp>
#include <graph/generic_graph.hpp>

using util::ArcSet, util::EdgeSet;
using graph::PartiallyDirectedGraph;

namespace util {

    struct ListRestrictions {
        ArcSet arc_blacklist;
        ArcSet arc_whitelist;
        EdgeSet edge_blacklist;
        EdgeSet edge_whitelist;
    };

    ListRestrictions check_whitelists(const PartiallyDirectedGraph& g,
                                      const ArcStringVector& varc_blacklist, 
                                      const ArcStringVector& varc_whitelist,
                                      const EdgeStringVector& vedge_blacklist,
                                      const EdgeStringVector& vedge_whitelist);
}

#endif //PYBNESIAN_UTIL_VALIDATE_WHITELISTS_HPP