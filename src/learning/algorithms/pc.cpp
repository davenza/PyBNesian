#include <optional>

#include <learning/algorithms/pc.hpp>
#include <graph/undirected.hpp>
#include <util/combinations.hpp>

using graph::PartiallyDirectedGraph, graph::UndirectedGraph, graph::Edge, graph::EdgeHash, graph::EdgeEqualTo;

using util::Combinations, util::Combinations2Sets;

namespace learning::algorithms {

    bool max_cardinality(UndirectedGraph& g, int max_cardinality) {
        for (int i = 0; i < g.num_nodes(); ++i) {
            if (g.num_neighbors_unsafe(i) > max_cardinality)
                return false;
        }
        return true;
    }

    void remove_edges(UndirectedGraph& g, const std::vector<Edge>& edges) {
        for (auto edge : edges) {
            g.remove_edge(edge.first, edge.second);
        }
    }

    std::optional<std::pair<int, double>> find_univariate_sepset(UndirectedGraph& g, 
                                                const Edge& edge, 
                                                double alpha, 
                                                const IndependenceTest& test) {

        std::unordered_set<int> nbr;
        const auto& n1 = g.node(edge.first);
        const auto& n2 = g.node(edge.second);

        nbr.insert(n1.neighbors().begin(), n1.neighbors().end());
        nbr.insert(n2.neighbors().begin(), n2.neighbors().end());

        nbr.erase(edge.first);
        nbr.erase(edge.second);

        for (auto cond : nbr) {
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

    std::optional<std::pair<std::unordered_set<int>, double>> find_multivariate_sepset(UndirectedGraph& g, 
                                                                   const Edge& edge, 
                                                                   int sep_size,
                                                                   const IndependenceTest& test,
                                                                   double alpha) {

        auto nbr = g.neighbor_indices(edge.first);
        auto nbr2 = g.neighbor_indices(edge.second);
        
        bool set1_valid = static_cast<int>(nbr.size()) > sep_size;
        bool set2_valid = static_cast<int>(nbr2.size()) > sep_size;

        if (!set1_valid && !set2_valid) {
            return {};
        }

        nbr.erase(edge.second);
        nbr2.erase(edge.first);

        if (set1_valid) {
            if (set2_valid) {
                Combinations2Sets comb(nbr.begin(), nbr.end(), nbr2.begin(), nbr2.end(), sep_size);
                return evaluate_multivariate_sepset(edge, comb, test, alpha);
            } else {
                Combinations comb(nbr.begin(), nbr.end(), sep_size);
                return evaluate_multivariate_sepset(edge, comb, test, alpha);
            }
        } else {
            if (set2_valid) {
                Combinations comb(nbr2.begin(), nbr2.end(), sep_size);
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


    std::pair<UndirectedGraph, SepSet> find_skeleton(const DataFrame& df, 
                                                   const IndependenceTest& test, 
                                                   double alpha) {
        SepSet sepset;

        auto g = UndirectedGraph::Complete(df.column_names());
        std::cout << "Cardinality 0" << std::endl;
        int nnodes = g.num_nodes();
        for (int i = 0; i < nnodes-1; ++i) {
            for (int j = i+1; j < nnodes; ++j) {
                double pvalue = test.pvalue(i, j);

                if (pvalue > alpha) {
                    g.remove_edge(i, j);
                    sepset.insert({i,j}, {}, pvalue);
                }
            }
        }
        
        if (max_cardinality(g, 1)) {
            return std::make_pair(g, sepset);
        }

        std::cout << "Cardinality 1" << std::endl;

        std::vector<Edge> edges_to_remove;

        for (auto& edge : g.edge_indices()) {
            auto indep = find_univariate_sepset(g, edge, alpha, test);
            if (indep) {
                edges_to_remove.push_back(edge);
                sepset.insert(edge, {indep->first}, indep->second);
            }
        }

        remove_edges(g, edges_to_remove);

        auto limit = 2;
        while(!max_cardinality(g, limit)) {
            std::cout << "Cardinality " << limit << std::endl;
            edges_to_remove.clear();
            
            for (auto& edge : g.edge_indices()) {
                auto indep = find_multivariate_sepset(g, edge, limit, test, alpha);
                if (indep) {
                    edges_to_remove.push_back(edge);
                    sepset.insert(edge, std::move(indep->first), indep->second);
                }
            }

            remove_edges(g, edges_to_remove);
            ++limit;
        }

        return std::make_pair(g, sepset);
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

        // std::cout << "\tS = ["  << g.name(vs.children) << "] pvalue: " << test.pvalue(vs.p1, vs.p2, vs.children) << std::endl; 
        if (test.pvalue(vs.p1, vs.p2, vs.children) > alpha) {
            // std::cout << "[vstructure] "  << g.name(vs.p1) << " _|_ " << g.name(vs.p2) << " | " << g.name(vs.children) << std::endl; 
            ++indep_sepsets;
            ++children_in_sepsets;
        }
        
        std::unordered_set<int> possible_sepset;
        const auto& np1 = g.node(vs.p1);
        const auto& np2 = g.node(vs.p2);

        possible_sepset.insert(np1.neighbors().begin(), np1.neighbors().end());
        possible_sepset.insert(np2.neighbors().begin(), np2.neighbors().end());
        possible_sepset.erase(vs.children);

        for (auto sp : possible_sepset) {
            // std::cout << "\tS = ["  << g.name(sp) << "] pvalue: " << test.pvalue(vs.p1, vs.p2, sp) << std::endl; 
            if (test.pvalue(vs.p1, vs.p2, sp) > alpha) {
                // std::cout << "[vstructure] "  << g.name(vs.p1) << " _|_ " << g.name(vs.p2) << " | " << g.name(sp) << std::endl; 
                ++indep_sepsets;
            }
        }

        return std::make_pair(indep_sepsets, children_in_sepsets);
    }

    template<typename Comb>
    std::pair<int, int> count_multivariate_sepsets(const PartiallyDirectedGraph& g, 
                                                   const vstructure& vs,
                                                   Comb& comb,
                                                   const IndependenceTest& test,
                                                   double alpha) {

        int indep_sepsets = 0;
        int children_in_sepsets = 0;

        for (auto& sepset : comb) {
            double pvalue = test.pvalue(vs.p1, vs.p2, sepset.begin(), sepset.end());

            // std::cout << "\tS = [" << g.name(sepset[0]);
            // for(auto it = ++sepset.begin(), end = sepset.end(); it != end; ++it) {
            //     std::cout << ", " << g.name(*it);
            // }
            // std::cout << "]; pvalue: " << pvalue << std::endl;


            if (pvalue > alpha) {
                // std::cout << "[vstructure] "  << g.name(vs.p1) << " _|_ " << g.name(vs.p2) << " | " << g.name(sepset[0]);
                // for(auto it = ++sepset.begin(), end = sepset.end(); it != end; ++it) {
                //     std::cout << ", " << g.name(*it);
                // }
                // std::cout << std::endl;


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
                                   double ambiguous_threshold, 
                                   double ambiguous_slack) {

        if (is_unshielded_triple(g, vs)) {

            int max_sepset = std::max(g.num_neighbors(vs.p1), g.num_neighbors(vs.p2));
            
            std::cout << "==============================" << std::endl;
            std::cout << "[vstructure] Unshielded triple " << g.name(vs.p1) << " - " << g.name(vs.children) << " - " << g.name(vs.p2) << std::endl;
            std::cout << "[vstructure] max sepset: " << max_sepset << std::endl; 
            std::cout << "==============================" << std::endl;
            
            double marg_pvalue = test.pvalue(vs.p1, vs.p2);

            int indep_sepsets = 0;
            int children_in_sepsets = 0;
            // std::cout << "\tS = []; pvalue: " << marg_pvalue << std::endl;
            if (marg_pvalue > alpha) {
                // std::cout << "[vstructure] "  << g.name(vs.p1) << " _|_ " << g.name(vs.p2) << " | ()" << std::endl; 
                ++indep_sepsets;
            }

            auto univariate_counts = count_univariate_sepsets(g, vs, test, alpha);

            indep_sepsets += univariate_counts.first;
            children_in_sepsets += univariate_counts.second;
            
            if (max_sepset >= 2) {
                std::pair<int, int> multivariate_counts;

                const auto& nbr1 = g.neighbor_indices(vs.p1);
                const auto& nbr2 = g.neighbor_indices(vs.p2);

                for (auto i = 2; i <= max_sepset; ++i) {
                    bool set1_valid = nbr1.size() >= i;
                    bool set2_valid = nbr2.size() >= i;
                    if (set1_valid) {
                        if (set2_valid) {
                            Combinations2Sets comb(nbr1.begin(), nbr1.end(), nbr2.begin(), nbr2.end(), i);
                            multivariate_counts = count_multivariate_sepsets(g, vs, comb, test, alpha);
                        } else {
                            Combinations comb(nbr1.begin(), nbr1.end(), i);
                            multivariate_counts = count_multivariate_sepsets(g, vs, comb, test, alpha);
                        }
                    } else {
                        Combinations comb(nbr2.begin(), nbr2.end(), i);
                        multivariate_counts = count_multivariate_sepsets(g, vs, comb, test, alpha);
                    }

                    indep_sepsets += multivariate_counts.first;
                    children_in_sepsets += multivariate_counts.second;
                }
            }

            double ratio = static_cast<double>(children_in_sepsets) / indep_sepsets;
            // std::cout << "[vstructure] Sepsets: " << indep_sepsets << ", Children in sepsets: " << children_in_sepsets 
            //             << ", Ratio: " << ratio << std::endl;
            if (ratio == 0 || ratio < (ambiguous_threshold - ambiguous_slack)) {
                std::cout << "[vstructure] Valid vstructure  " << g.name(vs.p1) << " - " << g.name(vs.children) << " - " << g.name(vs.p2) << std::endl;
                return true;
            }
        }

        return false;
    }
    
    std::vector<vstructure> evaluate_vstructures_at_node(const PartiallyDirectedGraph& g, 
                                                         const PDNode& node, 
                                                         const SepSet& sepset,
                                                         const IndependenceTest& test,
                                                         double alpha,
                                                         double ambiguous_threshold, 
                                                         double ambiguous_slack) {
        const auto& nbr = node.neighbors();

        std::vector<int> v {nbr.begin(), nbr.end()};

        std::vector<vstructure> res;
        if (v.size() == 2) {
            vstructure vs { .p1 = v[0], .p2 = v[1], .children = node.index() };
            if (is_unambiguous_vstructure(g, vs, test, alpha, ambiguous_threshold, ambiguous_slack))
                res.push_back(std::move(vs));
        } else if (v.size() == 3) {
            vstructure vs { .p1 = v[0], .p2 = v[1], .children = node.index() };
            if (is_unambiguous_vstructure(g, vs, test, alpha, ambiguous_threshold, ambiguous_slack))
                res.push_back(std::move(vs));

            vs = { .p1 = v[0], .p2 = v[2], .children = node.index() };
            if (is_unambiguous_vstructure(g, vs, test, alpha, ambiguous_threshold, ambiguous_slack))
                res.push_back(std::move(vs));
            
            vs = { .p1 = v[1], .p2 = v[2], .children = node.index() };
            if (is_unambiguous_vstructure(g, vs, test, alpha, ambiguous_threshold, ambiguous_slack))
                res.push_back(std::move(vs));
        } else {
            Combinations comb(std::move(v), 2);
            for (const auto& parents : comb) {
                vstructure vs { .p1 = parents[0], .p2 = parents[1], .children = node.index() };
                if (is_unambiguous_vstructure(g, vs, test, alpha, ambiguous_threshold, ambiguous_slack))
                    res.push_back(std::move(vs));
            }
        }

        return res;
    }


    void direct_unshielded_triples(PartiallyDirectedGraph& pdag, 
                                   const SepSet& sepset, 
                                   const IndependenceTest& test,
                                   double alpha,
                                   double ambiguous_threshold, 
                                   double ambiguous_slack) {
        std::cout << "Starting searching vstructures" << std::endl;

        std::vector<vstructure> vs;
        for (const auto& node : pdag.node_indices()) {
            if (node.neighbors().size() >= 2) {
                std::cout << "Evaluating vstructures at node " << node.name() << std::endl; 
                auto tmp = evaluate_vstructures_at_node(pdag, node, sepset, test, alpha, ambiguous_threshold, ambiguous_slack);

                vs.insert(vs.end(), tmp.begin(), tmp.end());
            }
        }
        std::cout << "Ending searching vstructures" << std::endl; 

        for(const auto& vstructure : vs) {
            std::cout << "[vstructure] Apply vstructure  " << pdag.name(vstructure.p1) << " -> " 
                      << pdag.name(vstructure.children) << " <- " << pdag.name(vstructure.p2) << std::endl;
            pdag.direct(vstructure.p1, vstructure.children);
            pdag.direct(vstructure.p2, vstructure.children);
        }
    }

    bool MeekRules::rule1(PartiallyDirectedGraph& pdag) {
        bool changed = false;
        for (const auto& arc : pdag.arc_indices()) {
            auto children = arc.second;

            for (const auto& neigh : pdag.neighbor_indices(children)) {
                if (!pdag.has_connection_unsafe(arc.first, neigh)) {
                    std::cout << "[Rule1] Direct " << pdag.name(arc.second) << " -> " << pdag.name(neigh) << std::endl;
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
                std::cout << "[Rule2] Direct " << pdag.name(edge.first) << " -> " << pdag.name(edge.second) << std::endl;
            }

            const auto& parents1 = n1.parents();
            const auto& children2 = n2.children();

            if (any_intersect(parents1, children2)) {
                changed = true;
                pdag.direct(edge.second, edge.first);
                std::cout << "[Rule2] Direct " << pdag.name(edge.first) << " -> " << pdag.name(edge.second) << std::endl;
            }            
        }

        return changed;
    }

    bool rule3_at_node(PartiallyDirectedGraph& pdag, const PDNode& n) {
        const auto& nbr = n.neighbors();
        const auto& parents = n.parents();

        bool changed = false;
        for (const auto& neigh : nbr) {
            const auto& nbr_of_neigh = pdag.neighbor_indices(neigh);
            auto intersection = intersect(nbr_of_neigh, parents);

            if (intersection.size() >= 2) {
                Combinations comb(intersection.begin(), intersection.end(), 2);

                for (const auto& p : comb) {
                    if (!pdag.has_connection(p[0], p[1])) {
                        pdag.direct(neigh, n.index());
                        changed = true;
                        std::cout << "[Rule3] Direct " << pdag.name(neigh) << " -> " << pdag.name(n.index()) << std::endl;
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

    void show_skeleton(UndirectedGraph& g) {

        auto edges = g.edges();

        if (!edges.empty()) {
            std::cout << "Skeleton: [(\"" << edges[0].first << "\", \"" << edges[0].second << "\")";

            for(auto it = ++edges.begin(), end = edges.end(); it != end; ++it) {
                std::cout << ", (\"" << it->first << "\", \"" << it->second << "\")";
            }

            std::cout << "]" << std::endl;
            
        } else {
            std::cout << "Skeleton: []" << std::endl;
        }
    }

    PartiallyDirectedGraph PC::estimate(const DataFrame& df, 
                        ArcVector& arc_blacklist, 
                        ArcVector& arc_whitelist, 
                        const IndependenceTest& test,
                        double alpha,
                        double ambiguous_threshold,
                        double ambiguous_slack) {

        GaussianNetwork::requires(df);

        auto [skeleton, sepset] = find_skeleton(df, test, alpha);
        show_skeleton(skeleton);

        PartiallyDirectedGraph pdag(std::move(skeleton));

        direct_unshielded_triples(pdag, sepset, test, alpha, ambiguous_threshold, ambiguous_slack);

        bool changed = true;
        while(changed) {
            changed = false;

            changed |= MeekRules::rule1(pdag);
            changed |= MeekRules::rule2(pdag);
            changed |= MeekRules::rule3(pdag);
        }

        return pdag;
    }

}