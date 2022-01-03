#ifndef PYBNESIAN_UTIL_VALIDATE_WHITELISTS_HPP
#define PYBNESIAN_UTIL_VALIDATE_WHITELISTS_HPP

#include <dataset/dataset.hpp>
#include <graph/generic_graph.hpp>
#include <util/util_types.hpp>
#include <util/bn_traits.hpp>

using dataset::DataFrame;
using graph::PartiallyDirectedGraph;
using util::ArcSet, util::EdgeSet;

namespace util {

struct ListRestrictions {
    ArcSet arc_blacklist;
    ArcSet arc_whitelist;
    EdgeSet edge_blacklist;
    EdgeSet edge_whitelist;
};

template <typename Model, util::enable_if_not_conditionalbn_t<Model, int> = 0>
void check_arc_list(const Model& g, const ArcStringVector& list) {
    for (auto pair : list) {
        if (!g.contains_node(pair.first))
            throw std::invalid_argument("Node " + pair.first + " not present in the graph.");

        if (!g.contains_node(pair.second))
            throw std::invalid_argument("Node " + pair.second + " not present in the graph.");
    }
}

template <typename Model, util::enable_if_conditionalbn_t<Model, int> = 0>
void check_arc_list(const Model& g, const ArcStringVector& list) {
    for (auto pair : list) {
        if (!g.contains_joint_node(pair.first))
            throw std::invalid_argument("Node " + pair.first + " not present in the graph.");

        if (!g.contains_node(pair.second))
            throw std::invalid_argument("Node " + pair.second + " not present in the graph.");
    }
}

template <typename Model, util::enable_if_not_conditionalbn_t<Model, int> = 0>
void check_edge_list(const Model& g, const EdgeStringVector& list) {
    return check_arc_list(g, list);
}

template <typename Model, util::enable_if_conditionalbn_t<Model, int> = 0>
void check_edge_list(const Model& g, const EdgeStringVector& list) {
    for (auto pair : list) {
        if (!g.contains_joint_node(pair.first))
            throw std::invalid_argument("Node " + pair.first + " not present in the graph.");

        if (!g.contains_joint_node(pair.second))
            throw std::invalid_argument("Node " + pair.second + " not present in the graph.");
    }
}

template <typename Model>
void check_node_type_list(const Model& g, const FactorTypeVector& list) {
    for (auto pair : list) {
        if (!g.contains_node(pair.first))
            throw std::invalid_argument("Node " + pair.first + " not present in the graph.");
    }
}

void check_arc_list(const DataFrame& df, const ArcStringVector& list);
void check_edge_list(const DataFrame& df, const EdgeStringVector& list);
void check_node_type_list(const DataFrame& df, const FactorTypeVector& list);

template <typename Model>
ListRestrictions validate_restrictions(const Model& g,
                                       const ArcStringVector& varc_blacklist,
                                       const ArcStringVector& varc_whitelist,
                                       const EdgeStringVector& vedge_blacklist,
                                       const EdgeStringVector& vedge_whitelist) {
    check_arc_list(g, varc_blacklist);
    check_arc_list(g, varc_whitelist);
    check_edge_list(g, vedge_blacklist);
    check_edge_list(g, vedge_whitelist);

    ListRestrictions r;

    for (const auto& edge : vedge_blacklist) {
        r.edge_blacklist.insert({g.index(edge.first), g.index(edge.second)});
    }

    for (const auto& edge : vedge_whitelist) {
        auto e1 = g.index(edge.first);
        auto e2 = g.index(edge.second);

        // Edge blacklist + Edge whitelist = Not possible
        if (r.edge_blacklist.count({e1, e2}) > 0) {
            throw std::invalid_argument("Edge " + edge.first + " -- " + edge.second + " in blacklist and whitelist");
        }

        r.edge_whitelist.insert({e1, e2});
    }

    for (const auto& arc : varc_whitelist) {
        auto s = g.index(arc.first);
        auto t = g.index(arc.second);

        // Edge blacklist + Arc whitelist =  Not possible
        if (r.edge_blacklist.count({s, t}) > 0) {
            throw std::invalid_argument("Edge blacklist " + arc.first + " -- " + arc.second +
                                        " is incompatible with arc whitelist" + arc.first + " -> " + arc.second);
        }

        // Edge whitelist + Arc whitelist = Arc whitelist
        if (r.edge_whitelist.count({s, t}) > 0) {
            r.edge_whitelist.erase({s, t});
        }

        r.arc_whitelist.insert({s, t});
    }

    for (const auto& arc : varc_blacklist) {
        auto s = g.index(arc.first);
        auto t = g.index(arc.second);

        // Arc whitelist + Arc blacklist = Not possible
        if (r.arc_whitelist.count({s, t}) > 0) {
            throw std::invalid_argument("Arc " + arc.first + " -> " + arc.second + " in blacklist and whitelist");
        }

        // Edge whitelist + Arc blacklist = Arc whitelist in opposite direction.
        if (r.edge_whitelist.count({s, t}) > 0) {
            r.arc_whitelist.insert({t, s});
            r.edge_whitelist.erase({s, t});
        }

        // Edge blacklist + Arc blacklist  = Edge blacklist -> do nothing.
        if (r.edge_blacklist.count({s, t}) == 0) r.arc_blacklist.insert({s, t});
    }

    for (auto it = r.arc_blacklist.begin(), end = r.arc_blacklist.end(); it != end;) {
        auto arc = *it;

        // Arc blacklist + Arc blacklist in opposite direction = Edge blacklist
        if (r.arc_blacklist.count({arc.second, arc.first}) > 0) {
            r.edge_blacklist.insert(arc);
            r.arc_blacklist.erase({arc.second, arc.first});
            it = r.arc_blacklist.erase(it);
        } else {
            ++it;
        }
    }

    return r;
}

template <typename Model>
ListRestrictions validate_restrictions(const Model& g,
                                       const ArcStringVector& varc_blacklist,
                                       const ArcStringVector& varc_whitelist) {
    check_arc_list(g, varc_blacklist);
    check_arc_list(g, varc_whitelist);

    ListRestrictions r;

    for (const auto& arc : varc_whitelist) {
        auto s = g.index(arc.first);
        auto t = g.index(arc.second);
        r.arc_whitelist.insert({s, t});
    }

    for (const auto& arc : varc_blacklist) {
        auto s = g.index(arc.first);
        auto t = g.index(arc.second);

        // Arc whitelist + Arc blacklist = Not possible
        if (r.arc_whitelist.count({s, t}) > 0) {
            throw std::invalid_argument("Arc " + arc.first + " -> " + arc.second + " in blacklist and whitelist");
        }

        r.arc_blacklist.insert({s, t});
    }

    return r;
}

template <typename Model>
void validate_type_restrictions(const Model& g,
                                const FactorTypeVector& type_blacklist,
                                const FactorTypeVector& type_whitelist) {
    if (type_blacklist.empty() || type_whitelist.empty()) {
        const auto& non_empty_list = (type_blacklist.empty()) ? type_whitelist : type_blacklist;
        std::string name_list = (type_blacklist.empty()) ? "whitelist" : "blacklist";

        for (const auto& [name, _] : non_empty_list) {
            if (!g.contains_node(name)) {
                throw std::invalid_argument("Node in the " + name_list + " (" + name + "), not present in the model.");
            }
        }

        return;
    }

    std::unordered_map<std::string, std::shared_ptr<FactorType>> whitelist_set;

    for (const auto& [name, type] : type_whitelist) {
        if (!g.contains_node(name)) {
            throw std::invalid_argument("Node in the whitelist (" + name + "), not present in the model.");
        }

        auto [it, inserted] = whitelist_set.insert({name, type});
        if (!inserted && *it->second != *type) {
            throw std::invalid_argument("Node " + name + " has two FactorType in the whitelist: " +
                                        it->second->ToString() + " and " + type->ToString() + ".");
        }
    }

    for (const auto& [name, type] : type_blacklist) {
        if (!g.contains_node(name)) {
            throw std::invalid_argument("Node in the blacklist (" + name + "), not present in the model.");
        }

        auto it = whitelist_set.find(name);

        if (it != whitelist_set.end() && *it->second == *type) {
            throw std::invalid_argument("Node " + name + " has a FactorType " + type->ToString() +
                                        " in blacklist and whitelist.");
        }
    }
}

}  // namespace util

#endif  // PYBNESIAN_UTIL_VALIDATE_WHITELISTS_HPP