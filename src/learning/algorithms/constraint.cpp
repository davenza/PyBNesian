#include <learning/algorithms/constraint.hpp>
#include <util/combinations.hpp>

using util::Combinations, util::Combinations2Sets;

namespace learning::algorithms {

    void direct_arc_blacklist(PartiallyDirectedGraph& g, const ArcSet& arc_blacklist) {
        for (const auto& arc : arc_blacklist) {
            if (g.has_edge_unsafe(arc.first, arc.second)) {
                g.direct(arc.second, arc.first);
            }
        }
    }

    struct vstructure {
        int p1;
        int p2;
        int children;
    };

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

        if (ambiguous_threshold == 0 && children_in_sepsets > 0) return false; 
        
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

    inline bool is_unshielded_triple(const PartiallyDirectedGraph& g, const vstructure& vs) {
        if (!g.has_connection(vs.p1, vs.p2)) {
            return true;
        }

        return false;
    }

    bool is_vstructure(const PartiallyDirectedGraph& g, 
                                   const vstructure& vs,
                                   const IndependenceTest& test,
                                   double alpha,
                                   const std::optional<SepSet>& sepset,
                                   bool use_sepsets,
                                   double ambiguous_threshold) {

        if (is_unshielded_triple(g, vs)) {
            if (use_sepsets) {
                if (sepset) {
                    const auto& s = sepset->sepset({vs.p1, vs.p2});
                    return s.first.count(vs.children) == 0;
                } else {
                    // Search for a subset S, such that p1 _|_ p2 | S.
                    return is_unambiguous_vstructure(g, vs, test, alpha, 0);
                }
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
                                                         const std::optional<SepSet>& sepset,
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
                                   const std::optional<SepSet>& sepset,
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
                    arc_blacklist.count({vstructure.p2, vstructure.children}) > 0)
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

        // Iterate over a copy because we are making changes.
        std::vector<Edge> edge_indices(pdag.edge_indices().begin(), pdag.edge_indices().end());
        for (const auto& edge : edge_indices) {
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
}
