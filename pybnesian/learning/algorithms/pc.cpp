#include <optional>
#include <graph/graph_types.hpp>
#include <learning/algorithms/pc.hpp>
#include <learning/algorithms/constraint.hpp>
#include <util/combinations.hpp>
#include <util/validate_whitelists.hpp>
#include <util/progress.hpp>
#include <util/vector.hpp>

using graph::PartiallyDirectedGraph, graph::UndirectedGraph, graph::Arc, graph::ArcHash, graph::Edge, graph::EdgeHash,
    graph::EdgeEqualTo;

using util::Combinations, util::Combinations2Sets, util::ProgressBar;

namespace learning::algorithms {

template <typename G>
bool max_cardinality(const G& g, int max_cardinality) {
    for (int i = 0; i < g.num_nodes(); ++i) {
        if ((g.num_neighbors_unsafe(i) + g.num_parents_unsafe(i)) > max_cardinality) return false;
    }
    return true;
}

template <typename G>
void remove_edges(G& g, const std::vector<Edge>& edges) {
    for (auto edge : edges) {
        g.remove_edge(edge.first, edge.second);
    }
}

template <typename G>
void filter_marginal_skeleton(G& skeleton,
                              const IndependenceTest& test,
                              SepSet& sepset,
                              double alpha,
                              EdgeSet& edge_whitelist,
                              util::BaseProgressBar& progress) {
    int nnodes = skeleton.num_nodes();
    if constexpr (graph::is_unconditional_graph_v<G>)
        progress.set_max_progress((nnodes * (nnodes - 1) / 2) - edge_whitelist.size());
    else if constexpr (graph::is_conditional_graph_v<G>) {
        progress.set_max_progress((nnodes * (nnodes - 1) / 2) + nnodes * skeleton.num_interface_nodes() -
                                  edge_whitelist.size());
    } else
        static_assert(util::always_false<G>, "Wrong graph class");

    progress.set_text("No sepset");
    progress.set_progress(0);

    const auto& nodes = skeleton.nodes();

    for (int i = 0; i < nnodes - 1; ++i) {
        auto index = skeleton.index(nodes[i]);
        for (int j = i + 1; j < nnodes; ++j) {
            auto other_index = skeleton.index(nodes[j]);

            if (skeleton.has_edge_unsafe(index, other_index) && edge_whitelist.count({index, other_index}) == 0) {
                double pvalue = test.pvalue(nodes[i], nodes[j]);
                if (pvalue > alpha) {
                    skeleton.remove_edge_unsafe(index, other_index);
                    sepset.insert({index, other_index}, {}, pvalue);
                }
                progress.tick();
            }
        }
    }

    if constexpr (graph::is_conditional_graph_v<G>) {
        const auto& interface_nodes = skeleton.interface_nodes();

        for (const auto& node : nodes) {
            auto nindex = skeleton.index(node);
            for (const auto& inode : interface_nodes) {
                auto iindex = skeleton.index(inode);

                if (skeleton.has_edge_unsafe(nindex, iindex) && edge_whitelist.count({nindex, iindex}) == 0) {
                    double pvalue = test.pvalue(node, inode);
                    if (pvalue > alpha) {
                        skeleton.remove_edge_unsafe(nindex, iindex);
                        sepset.insert({nindex, iindex}, {}, pvalue);
                    }

                    progress.tick();
                }
            }
        }
    }
}

template <typename G>
std::optional<std::pair<int, double>> find_univariate_sepset(const G& g,
                                                             const Edge& edge,
                                                             double alpha,
                                                             const IndependenceTest& test) {
    std::unordered_set<int> u;
    const auto& n1 = g.raw_node(edge.first);
    const auto& n2 = g.raw_node(edge.second);

    u.insert(n1.neighbors().begin(), n1.neighbors().end());
    u.insert(n1.parents().begin(), n1.parents().end());
    u.insert(n2.neighbors().begin(), n2.neighbors().end());
    u.insert(n2.parents().begin(), n2.parents().end());

    u.erase(edge.first);
    u.erase(edge.second);

    const auto& first_name = g.name(edge.first);
    const auto& second_name = g.name(edge.second);
    for (auto cond : u) {
        double pvalue = test.pvalue(first_name, second_name, g.name(cond));
        if (pvalue > alpha) {
            return std::optional<std::pair<int, double>>(std::make_pair(cond, pvalue));
        }
    }

    return {};
}

template <typename G>
void filter_univariate_skeleton(G& skeleton,
                                const IndependenceTest& test,
                                SepSet& sepset,
                                double alpha,
                                EdgeSet& edge_whitelist,
                                util::BaseProgressBar& progress) {
    progress.set_max_progress(skeleton.num_edges() - edge_whitelist.size());
    progress.set_text("Sepset Order 1");
    progress.set_progress(0);

    std::vector<Edge> edges_to_remove;

    for (const auto& edge : skeleton.edge_indices()) {
        if (edge_whitelist.count({edge.first, edge.second}) == 0) {
            auto indep = find_univariate_sepset(skeleton, edge, alpha, test);
            if (indep) {
                edges_to_remove.push_back(edge);
                sepset.insert(edge, {indep->first}, indep->second);
            }
            progress.tick();
        }
    }

    remove_edges(skeleton, edges_to_remove);
}

template <typename G, typename Comb>
std::optional<std::pair<std::unordered_set<int>, double>> evaluate_multivariate_sepset(
    const G& g, const Edge& edge, Comb& comb, const IndependenceTest& test, double alpha) {
    const auto& first_name = g.name(edge.first);
    const auto& second_name = g.name(edge.second);
    for (const auto& sepset : comb) {
        double pvalue = test.pvalue(first_name, second_name, sepset);
        if (pvalue > alpha) {
            std::unordered_set<int> indices;
            std::transform(
                sepset.begin(), sepset.end(), std::inserter(indices, indices.begin()), [&g](const std::string& name) {
                    return g.index(name);
                });

            return std::optional<std::pair<std::unordered_set<int>, double>>(
                std::make_pair<std::unordered_set<int>, double>(std::move(indices), std::move(pvalue)));
        }
    }

    return {};
}

template <typename G>
std::optional<std::pair<std::unordered_set<int>, double>> find_multivariate_sepset(
    const G& g, const Edge& edge, int sep_size, const IndependenceTest& test, double alpha) {
    const auto& nbr1 = g.neighbor_set(edge.first);
    const auto& pa1 = g.parent_set(edge.first);
    const auto& nbr2 = g.neighbor_set(edge.second);
    const auto& pa2 = g.parent_set(edge.second);

    bool set1_valid = static_cast<int>(nbr1.size() + pa1.size()) > sep_size;
    bool set2_valid = static_cast<int>(nbr2.size() + pa2.size()) > sep_size;

    if (!set1_valid && !set2_valid) {
        return {};
    }

    std::vector<std::string> u1;
    if (set1_valid) {
        u1.reserve(nbr1.size() + pa1.size());
        for (auto nbr : nbr1) {
            if (nbr != edge.second) u1.push_back(g.name(nbr));
        }

        std::transform(pa1.begin(), pa1.end(), std::inserter(u1, u1.end()), [&g](int pa) { return g.name(pa); });
    }

    std::vector<std::string> u2;
    if (set2_valid) {
        u2.reserve(nbr2.size() + pa2.size());
        for (auto nbr : nbr2) {
            if (nbr != edge.first) u2.push_back(g.name(nbr));
        }

        std::transform(pa2.begin(), pa2.end(), std::inserter(u2, u2.end()), [&g](int pa) { return g.name(pa); });
    }

    if (set1_valid) {
        if (set2_valid) {
            Combinations2Sets comb(std::move(u1), std::move(u2), sep_size);
            return evaluate_multivariate_sepset(g, edge, comb, test, alpha);
        } else {
            Combinations comb(std::move(u1), sep_size);
            return evaluate_multivariate_sepset(g, edge, comb, test, alpha);
        }
    } else {
        if (set2_valid) {
            Combinations comb(std::move(u2), sep_size);
            return evaluate_multivariate_sepset(g, edge, comb, test, alpha);
        }
    }

    return {};
}

template <typename G>
SepSet find_skeleton(
    G& g, const IndependenceTest& test, double alpha, EdgeSet& edge_whitelist, util::BaseProgressBar& progress) {
    if (static_cast<size_t>(g.num_edges()) == edge_whitelist.size()) {
        return SepSet{};
    }

    SepSet sepset;

    filter_marginal_skeleton(g, test, sepset, alpha, edge_whitelist, progress);

    if (static_cast<size_t>(g.num_edges()) == edge_whitelist.size() || max_cardinality(g, 1)) {
        return sepset;
    }

    filter_univariate_skeleton(g, test, sepset, alpha, edge_whitelist, progress);

    std::vector<Edge> edges_to_remove;
    auto limit = 2;
    while (static_cast<size_t>(g.num_edges()) > edge_whitelist.size() && !max_cardinality(g, limit)) {
        progress.set_max_progress(g.num_edges() - edge_whitelist.size());
        progress.set_text("Sepset Order " + std::to_string(limit));
        progress.set_progress(0);

        for (auto& edge : g.edge_indices()) {
            if (edge_whitelist.count({edge.first, edge.second}) == 0) {
                auto indep = find_multivariate_sepset(g, edge, limit, test, alpha);
                if (indep) {
                    edges_to_remove.push_back(edge);
                    sepset.insert(edge, std::move(indep->first), indep->second);
                }
                progress.tick();
            }
        }

        remove_edges(g, edges_to_remove);
        edges_to_remove.clear();
        ++limit;
    }

    return sepset;
}

template <typename G>
void estimate(G& skeleton,
              const IndependenceTest& test,
              const ArcStringVector& varc_blacklist,
              const ArcStringVector& varc_whitelist,
              const EdgeStringVector& vedge_blacklist,
              const EdgeStringVector& vedge_whitelist,
              double alpha,
              bool use_sepsets,
              double ambiguous_threshold,
              bool allow_bidirected,
              int verbose) {
    auto restrictions =
        util::validate_restrictions(skeleton, varc_blacklist, varc_whitelist, vedge_blacklist, vedge_whitelist);

    for (const auto& e : restrictions.edge_blacklist) {
        skeleton.remove_edge(e.first, e.second);
    }

    for (const auto& a : restrictions.arc_whitelist) {
        skeleton.direct(a.first, a.second);
    }

    // A cycle cannot be generated with less than 2 arcs.
    if (restrictions.arc_whitelist.size() > 2) {
        try {
            skeleton.to_dag();
        } catch (std::invalid_argument&) {
            throw std::invalid_argument(
                "The selected blacklist/whitelist configuration "
                "does not allow an acyclic graph.");
        }
    }

    auto progress = util::progress_bar(verbose);
    auto sepset = find_skeleton(skeleton, test, alpha, restrictions.edge_whitelist, *progress);

    if constexpr (graph::is_conditional_graph_v<G>) {
        skeleton.direct_interface_edges();
        remove_interface_arcs_blacklist(skeleton, restrictions.arc_blacklist);
    }

    direct_arc_blacklist(skeleton, restrictions.arc_blacklist);
    direct_unshielded_triples(skeleton,
                              test,
                              restrictions.arc_blacklist,
                              restrictions.arc_whitelist,
                              alpha,
                              sepset,
                              use_sepsets,
                              ambiguous_threshold,
                              allow_bidirected,
                              *progress);

    progress->set_max_progress(3);
    progress->set_text("Applying Meek rules");

    bool changed = true;
    while (changed) {
        changed = false;
        progress->set_progress(0);

        changed |= MeekRules::rule1(skeleton);
        progress->tick();
        changed |= MeekRules::rule2(skeleton);
        progress->tick();
        changed |= MeekRules::rule3(skeleton);
        progress->tick();
    }

    progress->mark_as_completed("Finished PC!");
}

PartiallyDirectedGraph PC::estimate(const IndependenceTest& test,
                                    const std::vector<std::string>& nodes,
                                    const ArcStringVector& varc_blacklist,
                                    const ArcStringVector& varc_whitelist,
                                    const EdgeStringVector& vedge_blacklist,
                                    const EdgeStringVector& vedge_whitelist,
                                    double alpha,
                                    bool use_sepsets,
                                    double ambiguous_threshold,
                                    bool allow_bidirected,
                                    int verbose) const {
    if (alpha <= 0 || alpha >= 1) throw std::invalid_argument("alpha must be a number between 0 and 1.");
    if (ambiguous_threshold < 0 || ambiguous_threshold > 1)
        throw std::invalid_argument("ambiguous_threshold must be a number between 0 and 1.");

    PartiallyDirectedGraph skeleton;
    if (nodes.empty())
        skeleton = PartiallyDirectedGraph::CompleteUndirected(test.variable_names());
    else {
        if (!test.has_variables(nodes))
            throw std::invalid_argument("IndependenceTest do not contain all the variables in nodes list.");

        skeleton = PartiallyDirectedGraph::CompleteUndirected(nodes);
    }

    learning::algorithms::estimate(skeleton,
                                   test,
                                   varc_blacklist,
                                   varc_whitelist,
                                   vedge_blacklist,
                                   vedge_whitelist,
                                   alpha,
                                   use_sepsets,
                                   ambiguous_threshold,
                                   allow_bidirected,
                                   verbose);
    return skeleton;
}

ConditionalPartiallyDirectedGraph PC::estimate_conditional(const IndependenceTest& test,
                                                           const std::vector<std::string>& nodes,
                                                           const std::vector<std::string>& interface_nodes,
                                                           const ArcStringVector& varc_blacklist,
                                                           const ArcStringVector& varc_whitelist,
                                                           const EdgeStringVector& vedge_blacklist,
                                                           const EdgeStringVector& vedge_whitelist,
                                                           double alpha,
                                                           bool use_sepsets,
                                                           double ambiguous_threshold,
                                                           bool allow_bidirected,
                                                           int verbose) const {
    if (alpha <= 0 || alpha >= 1) throw std::invalid_argument("alpha must be a number between 0 and 1.");
    if (ambiguous_threshold < 0 || ambiguous_threshold > 1)
        throw std::invalid_argument("ambiguous_threshold must be a number between 0 and 1.");

    if (nodes.empty()) throw std::invalid_argument("Node list cannot be empty to train a Conditional graph.");
    if (interface_nodes.empty())
        return PC::estimate(test,
                            nodes,
                            varc_blacklist,
                            varc_whitelist,
                            vedge_blacklist,
                            vedge_whitelist,
                            alpha,
                            use_sepsets,
                            ambiguous_threshold,
                            allow_bidirected,
                            verbose)
            .conditional_graph();

    if (!test.has_variables(nodes) || !test.has_variables(interface_nodes))
        throw std::invalid_argument(
            "IndependenceTest do not contain all the variables in nodes/interface_nodes lists.");

    auto skeleton = ConditionalPartiallyDirectedGraph::CompleteUndirected(nodes, interface_nodes);

    learning::algorithms::estimate(skeleton,
                                   test,
                                   varc_blacklist,
                                   varc_whitelist,
                                   vedge_blacklist,
                                   vedge_whitelist,
                                   alpha,
                                   use_sepsets,
                                   ambiguous_threshold,
                                   allow_bidirected,
                                   verbose);
    return skeleton;
}

}  // namespace learning::algorithms
