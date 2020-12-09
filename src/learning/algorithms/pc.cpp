#include <optional>
#include <indicators/cursor_control.hpp>
#include <graph/graph_types.hpp>
#include <learning/algorithms/pc.hpp>
#include <learning/algorithms/constraint.hpp>
#include <util/combinations.hpp>
#include <util/validate_whitelists.hpp>
#include <util/progress.hpp>
#include <util/vector.hpp>

using graph::PartiallyDirectedGraph, graph::UndirectedGraph, 
      graph::Arc, graph::ArcHash, graph::Edge, graph::EdgeHash, graph::EdgeEqualTo;

using util::Combinations, util::Combinations2Sets, util::ProgressBar;

namespace learning::algorithms {

    bool max_cardinality(const PartiallyDirectedGraph& g, int max_cardinality) {
        for (int i = 0; i < g.num_nodes(); ++i) {
            if ((g.num_neighbors_unsafe(i) + g.num_parents_unsafe(i)) > max_cardinality)
                return false;
        }
        return true;
    }

    void remove_edges(PartiallyDirectedGraph& g, const std::vector<Edge>& edges) {
        for (auto edge : edges) {
            g.remove_edge(edge.first, edge.second);
        }
    }

    std::optional<std::pair<int, double>> find_univariate_sepset(const PartiallyDirectedGraph& g, 
                                                                 const Edge& edge,
                                                                 double alpha,
                                                                 const IndependenceTest& test) {

        std::unordered_set<int> u;
        const auto& n1 = g.node(edge.first);
        const auto& n2 = g.node(edge.second);

        u.insert(n1.neighbors().begin(), n1.neighbors().end());
        u.insert(n1.parents().begin(), n1.parents().end());
        u.insert(n2.neighbors().begin(), n2.neighbors().end());
        u.insert(n2.parents().begin(), n2.parents().end());

        u.erase(edge.first);
        u.erase(edge.second);

        for (auto cond : u) {
            double pvalue = test.pvalue(edge.first, edge.second, cond);
            if (pvalue > alpha) {
                return std::optional<std::pair<int, double>>(std::make_pair(cond, pvalue));
            }
        }

        return {};
    }

    template<typename Comb>
    std::optional<std::pair<std::unordered_set<int>, double>> evaluate_multivariate_sepset(const Edge& edge,
                                                                        Comb& comb,
                                                                        const IndependenceTest& test,
                                                                        double alpha) {
        for (auto& sepset : comb) {
            double pvalue = test.pvalue(edge.first, edge.second, sepset.begin(), sepset.end());
            if (pvalue > alpha) {
                return std::optional<std::pair<std::unordered_set<int>, double>>(
                    std::make_pair<std::unordered_set<int>, double>({sepset.begin(), sepset.end()}, std::move(pvalue))
                );
            }
        }

        return {};
    }

    std::optional<std::pair<std::unordered_set<int>, double>> find_multivariate_sepset(const PartiallyDirectedGraph& g, 
                                                                                       const Edge& edge, 
                                                                                       int sep_size,
                                                                                       const IndependenceTest& test,
                                                                                       double alpha) {

        const auto& nbr1 = g.neighbor_set(edge.first);
        const auto& pa1 = g.parent_set(edge.first);
        const auto& nbr2 = g.neighbor_set(edge.second);
        const auto& pa2 = g.parent_set(edge.second);

        bool set1_valid = static_cast<int>(nbr1.size() + pa1.size()) > sep_size;
        bool set2_valid = static_cast<int>(nbr2.size() + pa2.size()) > sep_size;

        if (!set1_valid && !set2_valid) {
            return {};
        }

        std::vector<int> u1;
        if (set1_valid) {
            u1.reserve(nbr1.size() + pa1.size());
            u1.insert(u1.end(), nbr1.begin(), nbr1.end());
            u1.insert(u1.end(), pa1.begin(), pa1.end());
            util::swap_remove_v(u1, edge.second);
        }

        std::vector<int> u2;
        if (set2_valid) {
            u2.reserve(nbr2.size() + pa2.size());
            u2.insert(u2.end(), nbr2.begin(), nbr2.end());
            u2.insert(u2.end(), pa2.begin(), pa2.end());
            util::swap_remove_v(u2, edge.first);
        }

        if (set1_valid) {
            if (set2_valid) {
                Combinations2Sets comb(std::move(u1), std::move(u2), sep_size);
                return evaluate_multivariate_sepset(edge, comb, test, alpha);
            } else {
                Combinations comb(std::move(u1), sep_size);
                return evaluate_multivariate_sepset(edge, comb, test, alpha);
            }
        } else {
            if (set2_valid) {
                Combinations comb(std::move(u2), sep_size);
                return evaluate_multivariate_sepset(edge, comb, test, alpha);
            }
        }

        return {};
    }

    SepSet find_skeleton(PartiallyDirectedGraph& g,
                         const IndependenceTest& test, 
                         double alpha,
                         EdgeSet& edge_whitelist,
                         util::BaseProgressBar& progress) {

        if (static_cast<size_t>(g.num_edges()) == edge_whitelist.size()) {
            return SepSet{};
        }

        SepSet sepset;

        int nnodes = g.num_nodes();

        progress.set_max_progress((nnodes*(nnodes-1) / 2) - edge_whitelist.size());
        progress.set_text("No sepset");
        progress.set_progress(0);

        for (int i = 0; i < nnodes-1; ++i) {
            for (int j = i+1; j < nnodes; ++j) {
                if (g.has_edge_unsafe(i, j) && edge_whitelist.count({i, j}) == 0) {                  
                    double pvalue = test.pvalue(i, j);
                    if (pvalue > alpha) {
                        g.remove_edge_unsafe(i, j);
                        sepset.insert({i,j}, {}, pvalue);
                    }
                    progress.tick();
                }
            }
        }

        if (static_cast<size_t>(g.num_edges()) == edge_whitelist.size() || max_cardinality(g, 1)) {
            return sepset;
        }

        std::vector<Edge> edges_to_remove;

        progress.set_max_progress(g.num_edges() - edge_whitelist.size());
        progress.set_text("Sepset Order 1");
        progress.set_progress(0);

        for (auto& edge : g.edge_indices()) {
            if (edge_whitelist.count({edge.first, edge.second}) == 0) {
                auto indep = find_univariate_sepset(g, edge, alpha, test);
                if (indep) {
                    edges_to_remove.push_back(edge);
                    sepset.insert(edge, {indep->first}, indep->second);
                }
                progress.tick();
            }
        }

        remove_edges(g, edges_to_remove);

        auto limit = 2;
        while(static_cast<size_t>(g.num_edges()) > edge_whitelist.size() && !max_cardinality(g, limit)) {
            edges_to_remove.clear();

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
            ++limit;
        }

        return sepset;
    }

    PartiallyDirectedGraph PC::estimate(const IndependenceTest& test,
                        const ArcStringVector& varc_blacklist, 
                        const ArcStringVector& varc_whitelist,
                        const EdgeStringVector& vedge_blacklist,
                        const EdgeStringVector& vedge_whitelist,
                        double alpha,
                        bool use_sepsets,
                        double ambiguous_threshold,
                        bool allow_bidirected,
                        int verbose) const {
        
        auto skeleton = PartiallyDirectedGraph::CompleteUndirected(test.column_names());
        
        auto restrictions = util::validate_restrictions(skeleton, 
                                                        varc_blacklist,
                                                        varc_whitelist,
                                                        vedge_blacklist,
                                                        vedge_whitelist);

        for (const auto& e : restrictions.edge_blacklist) {
            skeleton.remove_edge(e.first, e.second);
        }

        for (const auto& a : restrictions.arc_whitelist) {
            skeleton.direct(a.first, a.second);
        }

        // A cycle can not be generated with less than 2 arcs.
        if (restrictions.arc_whitelist.size() > 2) {
            try {
                skeleton.to_dag();
            } catch (std::invalid_argument&) {
                throw std::invalid_argument("The selected blacklist/whitelist configuration "
                                            "does not allow an acyclic graph.");
            }
        }

        indicators::show_console_cursor(false);
        auto progress = util::progress_bar(verbose);
        auto sepset = find_skeleton(skeleton, test, alpha, restrictions.edge_whitelist, *progress);

        direct_arc_blacklist(skeleton, restrictions.arc_blacklist);

        direct_unshielded_triples(skeleton, test, restrictions.arc_blacklist, restrictions.arc_whitelist,
                                    alpha, sepset, use_sepsets, ambiguous_threshold, allow_bidirected, *progress);


        progress->set_max_progress(3);
        progress->set_text("Applying Meek rules");

        bool changed = true;
        while(changed) {
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
        indicators::show_console_cursor(true);
        return skeleton;
    }
}
