#include <optional>

#include <learning/algorithms/pc.hpp>
#include <graph/undirected.hpp>
#include <util/combinations.hpp>

using graph::UndirectedGraph, graph::Edge, graph::EdgeHash, graph::EdgeEqualTo;

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

    std::optional<int> find_univariate_sepset(UndirectedGraph& g, 
                                                const Edge& edge, 
                                                double alpha, 
                                                const IndependenceTest& test) {

        auto nbr = g.neighbor_indices(edge.first);
        nbr.erase(edge.second);
        auto nbr2 = g.neighbor_indices(edge.second);
        nbr2.erase(edge.first);

        nbr.merge(nbr2);

        for (auto cond : nbr) {
            double pvalue = test.pvalue(edge.first, edge.second, cond);

            if (pvalue > alpha) {
                return std::optional<int>(cond);
            }
        }

        return {};
    }

    template<typename Comb>
    std::optional<std::unordered_set<int>> evaluate_multivariate_sepset(const Edge& edge,
                                                                         Comb& comb,
                                                                         const IndependenceTest& test,
                                                                         double alpha) {
        for (auto& sepset : comb) {
            double pvalue = test.pvalue(edge.first, edge.second, sepset.begin(), sepset.end());

            if (pvalue > alpha) {
                return std::optional<std::unordered_set<int>>({sepset.begin(), sepset.end()});
            }
        }

        return {};
    }

    std::optional<std::unordered_set<int>> find_multivariate_sepset(UndirectedGraph& g, 
                                                                   const Edge& edge, 
                                                                   int sep_size,
                                                                   const IndependenceTest& test,
                                                                   double alpha) {

        auto nbr = g.neighbor_indices(edge.first);
        nbr.erase(edge.second);
        auto nbr2 = g.neighbor_indices(edge.second);
        nbr2.erase(edge.first);

        bool set1_valid = static_cast<int>(nbr.size()) >= sep_size;
        bool set2_valid = static_cast<int>(nbr2.size()) >= sep_size;

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
            } else {
                return {};
            }
        }
    }

    class SepSet {
    public:
        void insert(Edge e, const std::unordered_set<int>& s) {
            m_sep.insert(std::make_pair(e, s));
        }

        void insert(Edge e, std::unordered_set<int>&& s) {
            m_sep.insert(std::make_pair(e, std::move(s)));
        }

        const std::unordered_set<int>& sepset(Edge e) {
            auto f = m_sep.find(e);
            if (f == m_sep.end()) {
                throw std::out_of_range("Edge (" + std::to_string(e.first) + ", " + std::to_string(e.second) + ") not found in sepset.");
            }

            return f->second;
        }
    private:
        std::unordered_map<Edge, std::unordered_set<int>, EdgeHash, EdgeEqualTo> m_sep;
    };


    std::pair<UndirectedGraph, SepSet> find_sepset(const DataFrame& df, 
                                                   const IndependenceTest& test, 
                                                   double alpha) {
        SepSet sepset;

        auto g = UndirectedGraph::Complete(df.column_names());
        

        int nnodes = g.num_nodes();
        for (int i = 0; i < nnodes-1; ++i) {
            for (int j = i+1; j < nnodes; ++j) {
                double pvalue = test.pvalue(i, j);

                if (pvalue > alpha) {
                    g.remove_edge(i, j);
                    sepset.insert({i,j}, {});
                }
            }
        }
        
        if (max_cardinality(g, 1)) {
            return std::make_pair(g, sepset);
        }

        std::vector<Edge> edges_to_remove;

        for (auto& edge : g.edge_indices()) {
            auto indep = find_univariate_sepset(g, edge, alpha, test);
            if (indep) {
                edges_to_remove.push_back(edge);
                sepset.insert(edge, {*indep});
            }
        }

        remove_edges(g, edges_to_remove);

        auto limit = 2;
        while(!max_cardinality(g, limit)) {
            edges_to_remove.clear();
            
            for (auto& edge : g.edge_indices()) {
                auto indep = find_multivariate_sepset(g, edge, limit, test, alpha);
                if (indep) {
                    edges_to_remove.push_back(edge);
                    sepset.insert(edge, std::move(*indep));
                }
            }

            remove_edges(g, edges_to_remove);
            ++limit;
        }

        return std::make_pair(g, sepset);
    }


    void PC::estimate(const DataFrame& df, 
                        ArcVector& arc_blacklist, 
                        ArcVector& arc_whitelist, 
                        const IndependenceTest& test,
                        double alpha) {
        GaussianNetwork::requires(df);


        auto [skeleton, sepset] = find_sepset(df, test, alpha);

    }


}