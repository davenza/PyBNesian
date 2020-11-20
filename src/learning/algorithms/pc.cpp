#include <optional>
#include <indicators/cursor_control.hpp>
#include <graph/graph_types.hpp>
#include <learning/algorithms/pc.hpp>
#include <util/combinations.hpp>
#include <util/validate_whitelists.hpp>
#include <util/progress.hpp>

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
            std::iter_swap(std::find(u1.begin(), u1.end(), edge.second), u1.end() - 1); 
            u1.pop_back();
        }

        std::vector<int> u2;
        if (set2_valid) {
            u2.reserve(nbr2.size() + pa2.size());
            u2.insert(u2.end(), nbr2.begin(), nbr2.end());
            u2.insert(u2.end(), pa2.begin(), pa2.end());
            std::iter_swap(std::find(u2.begin(), u2.end(), edge.first), u2.end() - 1); 
            u2.pop_back();
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

    class SepSet {
    public:
        void insert(Edge e, const std::unordered_set<int>& s, double pvalue) {
            m_sep.insert(std::make_pair(e, std::make_pair(s, pvalue)));
        }

        void insert(Edge e, std::unordered_set<int>&& s, double pvalue) {
            m_sep.insert(std::make_pair(e, std::make_pair(std::move(s), pvalue)));
        }

        const std::pair<std::unordered_set<int>, double>& sepset(Edge e) const {
            auto f = m_sep.find(e);
            if (f == m_sep.end()) {
                throw std::out_of_range("Edge (" + std::to_string(e.first) + ", " + std::to_string(e.second) + ") not found in sepset.");
            }

            return f->second;
        }

        auto begin() { return m_sep.begin(); }
        auto end() { return m_sep.end(); }

    private:
        std::unordered_map<Edge, std::pair<std::unordered_set<int>, double>, EdgeHash, EdgeEqualTo> m_sep;
    };


    SepSet find_skeleton(PartiallyDirectedGraph& g,
                         const IndependenceTest& test, 
                         double alpha,
                         EdgeSet& edge_whitelist,
                         util::BaseProgressBar* progress) {

        if (static_cast<size_t>(g.num_edges()) == edge_whitelist.size()) {
            return SepSet{};
        }

        SepSet sepset;

        int nnodes = g.num_nodes();

        progress->set_max_progress((nnodes*(nnodes-1) / 2) - edge_whitelist.size());
        progress->set_text("Non sepset");
        progress->set_progress(0);

        for (int i = 0; i < nnodes-1; ++i) {
            for (int j = i+1; j < nnodes; ++j) {
                if (g.has_edge_unsafe(i, j) && edge_whitelist.count({i, j}) == 0) {                  
                    double pvalue = test.pvalue(i, j);
                    if (pvalue > alpha) {
                        g.remove_edge_unsafe(i, j);
                        sepset.insert({i,j}, {}, pvalue);
                    }
                    progress->tick();
                }
            }
        }

        if (static_cast<size_t>(g.num_edges()) == edge_whitelist.size() || max_cardinality(g, 1)) {
            return sepset;
        }

        std::vector<Edge> edges_to_remove;

        progress->set_max_progress(g.num_edges() - edge_whitelist.size());
        progress->set_text("Sepset Order 1");
        progress->set_progress(0);

        for (auto& edge : g.edge_indices()) {
            if (edge_whitelist.count({edge.first, edge.second}) == 0) {
                auto indep = find_univariate_sepset(g, edge, alpha, test);
                if (indep) {
                    edges_to_remove.push_back(edge);
                    sepset.insert(edge, {indep->first}, indep->second);
                }
                progress->tick();
            }
        }

        remove_edges(g, edges_to_remove);

        auto limit = 2;
        while(static_cast<size_t>(g.num_edges()) > edge_whitelist.size() && !max_cardinality(g, limit)) {
            edges_to_remove.clear();

            progress->set_max_progress(g.num_edges() - edge_whitelist.size());
            progress->set_text("Sepset Order " + std::to_string(limit));
            progress->set_progress(0);

            for (auto& edge : g.edge_indices()) {
                if (edge_whitelist.count({edge.first, edge.second}) == 0) {
                    auto indep = find_multivariate_sepset(g, edge, limit, test, alpha);
                    if (indep) {
                        edges_to_remove.push_back(edge);
                        sepset.insert(edge, std::move(indep->first), indep->second);
                    }
                    progress->tick();
                }
            }

            remove_edges(g, edges_to_remove);
            ++limit;
        }

        return sepset;
    }

    struct vstructure {
        int p1;
        int p2;
        int children;
    };

    inline bool is_unshielded_triple(const PartiallyDirectedGraph& g, const vstructure& vs) {
        if (!g.has_connection(vs.p1, vs.p2)) {
            return true;
        }

        return false;
    }

    std::pair<int, int> count_univariate_sepsets(const PartiallyDirectedGraph& g, 
                                                 const vstructure& vs,
                                                 const IndependenceTest& test,
                                                 double alpha) {

        int indep_sepsets = 0;
        int children_in_sepsets = 0;

        if (test.pvalue(vs.p1, vs.p2, vs.children) > alpha) {
            ++indep_sepsets;
            ++children_in_sepsets;
        }
        
        std::unordered_set<int> possible_sepset;
        const auto& np1 = g.node(vs.p1);
        const auto& np2 = g.node(vs.p2);

        possible_sepset.insert(np1.neighbors().begin(), np1.neighbors().end());
        possible_sepset.insert(np1.parents().begin(), np1.parents().end());
        possible_sepset.insert(np2.neighbors().begin(), np2.neighbors().end());
        possible_sepset.insert(np2.parents().begin(), np2.parents().end());
        possible_sepset.erase(vs.children);

        for (auto sp : possible_sepset) {
            if (test.pvalue(vs.p1, vs.p2, sp) > alpha) {
                ++indep_sepsets;
            }
        }

        return std::make_pair(indep_sepsets, children_in_sepsets);
    }

    template<typename Comb>
    std::pair<int, int> count_multivariate_sepsets(const vstructure& vs,
                                                   Comb& comb,
                                                   const IndependenceTest& test,
                                                   double alpha) {

        int indep_sepsets = 0;
        int children_in_sepsets = 0;

        for (auto& sepset : comb) {
            double pvalue = test.pvalue(vs.p1, vs.p2, sepset.begin(), sepset.end());
            if (pvalue > alpha) {
                ++indep_sepsets;
                if(std::find(sepset.begin(), sepset.end(), vs.children) != sepset.end()) {
                    ++children_in_sepsets;
                }
            }
        }

        return std::make_pair(indep_sepsets, children_in_sepsets);
    }

    bool is_unambiguous_vstructure(const PartiallyDirectedGraph& g, 
                                   const vstructure& vs,
                                   const IndependenceTest& test,
                                   double alpha,
                                   double ambiguous_threshold) {
        size_t max_sepset = std::max(g.num_neighbors(vs.p1) + g.num_parents(vs.p1), 
                                        g.num_neighbors(vs.p2) + g.num_parents(vs.p2));

        double marg_pvalue = test.pvalue(vs.p1, vs.p2);

        int indep_sepsets = 0;
        int children_in_sepsets = 0;
        if (marg_pvalue > alpha) {
            ++indep_sepsets;
        }

        auto univariate_counts = count_univariate_sepsets(g, vs, test, alpha);

        indep_sepsets += univariate_counts.first;
        children_in_sepsets += univariate_counts.second;
        
        if (max_sepset >= 2) {
            std::pair<int, int> multivariate_counts;

            const auto& nbr1 = g.neighbor_set(vs.p1);
            const auto& pa1 = g.parent_set(vs.p1);
            const auto& nbr2 = g.neighbor_set(vs.p2);
            const auto& pa2 = g.parent_set(vs.p2);

            std::vector<int> u1;
            if ((nbr1.size() + pa1.size()) >= 2) {
                u1.reserve(nbr1.size() + pa1.size());
                u1.insert(u1.end(), nbr1.begin(), nbr1.end());
                u1.insert(u1.end(), pa1.begin(), pa1.end());
            }

            std::vector<int> u2;
            if ((nbr2.size() + pa2.size()) >= 2) {
                u2.reserve(nbr2.size() + pa2.size());
                u2.insert(u2.end(), nbr2.begin(), nbr2.end());
                u2.insert(u2.end(), pa2.begin(), pa2.end());
            }
            
            for (size_t i = 2; i <= max_sepset; ++i) {
                bool set1_valid = u1.size() >= i;
                bool set2_valid = u2.size() >= i;

                if (set1_valid) {
                    if (set2_valid) {
                        Combinations2Sets comb(u1.begin(), u1.end(), u2.begin(), u2.end(), i);
                        multivariate_counts = count_multivariate_sepsets(vs, comb, test, alpha);
                    } else {
                        Combinations comb(u1.begin(), u1.end(), i);
                        multivariate_counts = count_multivariate_sepsets(vs, comb, test, alpha);
                    }
                } else {
                    Combinations comb(u2.begin(), u2.end(), i);
                    multivariate_counts = count_multivariate_sepsets(vs, comb, test, alpha);
                }

                indep_sepsets += multivariate_counts.first;
                children_in_sepsets += multivariate_counts.second;
            }
        }

        if (indep_sepsets > 0) {
            double ratio = static_cast<double>(children_in_sepsets) / indep_sepsets;
            return ratio < ambiguous_threshold || ratio == 0;
        }

        return false;
    }


    bool is_vstructure(const PartiallyDirectedGraph& g, 
                                   const vstructure& vs,
                                   const IndependenceTest& test,
                                   double alpha,
                                   const SepSet& sepset,
                                   bool use_sepsets,
                                   double ambiguous_threshold) {

        if (is_unshielded_triple(g, vs)) {
            if (use_sepsets) {
                const auto& s = sepset.sepset({vs.p1, vs.p2});
                return s.first.count(vs.children) == 0;
            } else {
                return is_unambiguous_vstructure(g, vs, test, alpha, ambiguous_threshold);
            }
        }

        return false;
    }

    std::vector<vstructure> evaluate_vstructures_at_node(const PartiallyDirectedGraph& g, 
                                                         const PDNode& node,
                                                         const IndependenceTest& test,
                                                         double alpha,
                                                         const SepSet& sepset,
                                                         bool use_sepsets,
                                                         double ambiguous_threshold) {
        const auto& nbr = node.neighbors();
        std::vector<int> v {nbr.begin(), nbr.end()};
        std::vector<vstructure> res;

        if (v.size() > 1) {
            // v-structures between neighbors.
            if (v.size() == 2) {
                vstructure vs { .p1 = v[0], .p2 = v[1], .children = node.index() };
                if (is_vstructure(g, vs, test, alpha, sepset, use_sepsets, ambiguous_threshold))
                    res.push_back(std::move(vs));
            } else if (v.size() == 3) {
                vstructure vs { .p1 = v[0], .p2 = v[1], .children = node.index() };
                if (is_vstructure(g, vs, test, alpha, sepset, use_sepsets, ambiguous_threshold))
                    res.push_back(std::move(vs));

                vs = { .p1 = v[0], .p2 = v[2], .children = node.index() };
                if (is_vstructure(g, vs, test, alpha, sepset, use_sepsets, ambiguous_threshold))
                    res.push_back(std::move(vs));
                
                vs = { .p1 = v[1], .p2 = v[2], .children = node.index() };
                if (is_vstructure(g, vs, test, alpha, sepset, use_sepsets, ambiguous_threshold))
                    res.push_back(std::move(vs));
            } else {
                Combinations comb(std::move(v), 2);
                for (const auto& parents : comb) {
                    vstructure vs { .p1 = parents[0], .p2 = parents[1], .children = node.index() };
                    if (is_vstructure(g, vs, test, alpha, sepset, use_sepsets, ambiguous_threshold))
                        res.push_back(std::move(vs));
                }
            }
        }

        // v-structures between a neighbor and a parent.
        const auto& parents = node.parents();

        if (parents.size() > 0) {
            // Remove already directed edges.
            std::unordered_set<int> remaining_neighbors{nbr.begin(), nbr.end()};
            for (const auto& found_vs : res) {
                remaining_neighbors.erase(found_vs.p1);
                remaining_neighbors.erase(found_vs.p2);
            }

            for (auto neighbor : remaining_neighbors) {
                for (auto parent : parents) {
                    vstructure vs { .p1 = neighbor, .p2 = parent, .children = node.index() };
                    if (is_vstructure(g, vs, test, alpha, sepset, use_sepsets, ambiguous_threshold))
                        res.push_back(std::move(vs));
                }
            }
        }

        return res;
    }


    void direct_unshielded_triples(PartiallyDirectedGraph& pdag,
                                   const IndependenceTest& test,
                                   const ArcSet& arc_blacklist,
                                   const ArcSet& arc_whitelist,
                                   double alpha,
                                   SepSet& sepset,
                                   bool use_sepsets,
                                   double ambiguous_threshold,
                                   bool allow_bidirected,
                                   util::BaseProgressBar* progress) {

        std::vector<vstructure> vs;

        progress->set_max_progress(pdag.num_nodes());
        progress->set_text("Finding v-structures");
        progress->set_progress(0);

        for (const auto& node : pdag.node_indices()) {
            if (node.neighbors().size() >= 1 && (node.parents().size() + node.neighbors().size()) >= 2) {
                auto tmp = evaluate_vstructures_at_node(pdag, node, test, alpha, sepset, use_sepsets, ambiguous_threshold);
                vs.insert(vs.end(), tmp.begin(), tmp.end());
            }
            progress->tick();
        }

        if (allow_bidirected) {
            for(const auto& vstructure : vs) {
                // If some arc of the structure is in blacklist, don't apply it.
                if (arc_blacklist.count({vstructure.p1, vstructure.children}) > 0 ||
                    arc_blacklist.count({vstructure.p2, vstructure.children}))
                    continue;

                pdag.direct(vstructure.p1, vstructure.children);
                pdag.direct(vstructure.p2, vstructure.children);
            }
        } else {
            for(const auto& vstructure : vs) {
                // If some arc of the structure is in blacklist, don't apply it.
                if (arc_blacklist.count({vstructure.p1, vstructure.children}) > 0 ||
                    arc_blacklist.count({vstructure.p2, vstructure.children}) > 0)
                    continue;
                
                // We cannot remove arcs in the whitelist.
                if ((pdag.has_arc_unsafe(vstructure.children, vstructure.p1) && 
                        arc_whitelist.count({vstructure.children, vstructure.p1}) > 0) ||
                    (pdag.has_arc_unsafe(vstructure.children, vstructure.p2) && 
                        arc_whitelist.count({vstructure.children, vstructure.p2}) > 0))
                    continue;

                pdag.direct(vstructure.p1, vstructure.children);
                pdag.direct(vstructure.p2, vstructure.children);

                if (pdag.has_arc_unsafe(vstructure.children, vstructure.p1))
                    pdag.remove_arc_unsafe(vstructure.children, vstructure.p1);
                if (pdag.has_arc_unsafe(vstructure.children, vstructure.p2))
                    pdag.remove_arc_unsafe(vstructure.children, vstructure.p2);
            }
        }
    }

    void direct_arc_blacklist(PartiallyDirectedGraph g, const ArcSet& arc_blacklist) {
        for (const auto& arc : arc_blacklist) {
            if (g.has_edge_unsafe(arc.first, arc.second)) {
                g.direct(arc.second, arc.first);
            }
        }
    }

    bool MeekRules::rule1(PartiallyDirectedGraph& pdag) {
        bool changed = false;
        for (const auto& arc : pdag.arc_indices()) {
            auto children = arc.second;

            for (const auto& neigh : pdag.neighbor_indices(children)) {
                if (!pdag.has_connection_unsafe(arc.first, neigh)) {
                    pdag.direct(arc.second, neigh);
                    changed = true;
                }
            }
        }

        return changed;
    }

    template<typename T>
    bool any_intersect(const std::unordered_set<T>& s1, const std::unordered_set<T>& s2) {
        const auto& [smaller_set, greater_set] = [&s1, &s2]() {
            if (s1.size() <= s2.size()) {
                return std::make_pair(s1, s2);
            } else {
                return std::make_pair(s2, s1);
            }
        }();

        for (const auto& el : smaller_set) {
            if (greater_set.count(el) > 0)
                return true;
        }

        return false;
    }

    template<typename T>
    std::unordered_set<T> intersect(const std::unordered_set<T>& s1, const std::unordered_set<T>& s2) {
        std::unordered_set<T> res;

        const auto& [smaller_set, greater_set] = [&s1, &s2]() {
            if (s1.size() <= s2.size()) {
                return std::make_pair(s1, s2);
            } else {
                return std::make_pair(s2, s1);
            }
        }();

        for (const auto& el : smaller_set) {
            if (greater_set.count(el) > 0)
                res.insert(el);
        }

        return res;
    }

    bool MeekRules::rule2(PartiallyDirectedGraph& pdag) {
        bool changed = false;
        for (const auto& edge : pdag.edge_indices()) {
            const auto& n1 = pdag.node(edge.first);
            const auto& n2 = pdag.node(edge.second);

            const auto& children1 = n1.children();
            const auto& parents2 = n2.parents();
            
            if (any_intersect(parents2, children1)) {
                changed = true;
                pdag.direct(edge.first, edge.second);
            }

            const auto& parents1 = n1.parents();
            const auto& children2 = n2.children();

            if (any_intersect(parents1, children2)) {
                changed = true;
                pdag.direct(edge.second, edge.first);
            }            
        }

        return changed;
    }

    bool rule3_at_node(PartiallyDirectedGraph& pdag, const PDNode& n) {
        const auto& nbr = n.neighbors();
        const auto& parents = n.parents();

        bool changed = false;
        for (const auto& neigh : nbr) {
            const auto& nbr_of_neigh = pdag.neighbor_set(neigh);
            auto intersection = intersect(nbr_of_neigh, parents);

            if (intersection.size() >= 2) {
                Combinations comb(intersection.begin(), intersection.end(), 2);

                for (const auto& p : comb) {
                    if (!pdag.has_connection(p[0], p[1])) {
                        pdag.direct(neigh, n.index());
                        changed = true;
                    }
                }
            }
        }

        return changed;
    }

    bool MeekRules::rule3(PartiallyDirectedGraph& pdag) {
        bool changed = false;

        for (const auto& node : pdag.node_indices()) {
            if (node.parents().size() >= 2 && node.neighbors().size() >= 1) {
                changed |= rule3_at_node(pdag, node);
            }
        }
        return changed;
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
                        int verbose) {
        
        auto skeleton = PartiallyDirectedGraph::CompleteUndirected(test.column_names());

        auto restrictions = util::check_whitelists(skeleton, 
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
            } catch (std::invalid_argument) {
                throw std::invalid_argument("The selected blacklist/whitelist configuration "
                                            "does not allow an acyclic graph.");
            }
        }

        indicators::show_console_cursor(false);
        auto progress = util::progress_bar(verbose);
        auto sepset = find_skeleton(skeleton, test, alpha, restrictions.edge_whitelist, progress.get());

        direct_arc_blacklist(skeleton, restrictions.arc_blacklist);

        direct_unshielded_triples(skeleton, test, restrictions.arc_blacklist, restrictions.arc_whitelist,
                                    alpha, sepset, use_sepsets, ambiguous_threshold, allow_bidirected, progress.get());


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
