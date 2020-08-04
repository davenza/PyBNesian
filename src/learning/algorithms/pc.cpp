#include <optional>

#include <learning/algorithms/pc.hpp>
#include <graph/undirected.hpp>
#include <util/combinations.hpp>

using graph::UndirectedGraph, graph::UEdge;

using util::Combinations;

namespace learning::algorithms {


    bool max_cardinality(UndirectedGraph& g, int max_cardinality) {
        for (int i = 0; i < g.num_nodes(); ++i) {
            if (g.num_neighbors_unsafe(i) > max_cardinality)
                return false;
        }
        return true;
    }

    std::optional<std::unordered_set<int>> find_conditional_subset(UndirectedGraph& g, 
                                                                   const UEdge& edge, 
                                                                   int sep_size,
                                                                   double alpha,
                                                                   const IndependenceTest& test) {
        auto nbr = g.neighbor_indices(edge.first);
        nbr.erase(edge.second);
        if (nbr.size() >= sep_size) {
            Combinations comb(nbr.begin(), nbr.end(), sep_size);

            for (auto& condset : comb) {
                double pvalue = test.pvalue(edge.first, edge.second, condset.begin(), condset.end());

                if (pvalue > alpha) {
                    // Is independent. Remove arc and update sepset.
                }
            }
        }


        auto nbr2 = g.neighbor_indices(edge.second);
        nbr2.erase(edge.first);

        if (sep_size > 0 && nbr2.size() >= sep_size && nbr != nbr2) {
            if (sep_size == 1) {
                
            }
        }



        // if (g.num_neighbors(edge.second) > sep_size) {

        // }           

        return std::optional<std::unordered_set<int>>{};
    }


    void PC::estimate(const DataFrame& df, 
                        ArcVector& arc_blacklist, 
                        ArcVector& arc_whitelist, 
                        const IndependenceTest& test,
                        double alpha) {
        GaussianNetwork::requires(df);

        auto g = UndirectedGraph::Complete(df.column_names());


        auto limit = 0;

        while(!max_cardinality(g, limit)) {

            for (auto& edge : g.edge_indices()) {
                find_conditional_subset(g, edge, limit, alpha, test);
            }
            
            // for (int node = 0; node < g.num_nodes(); ++node) {
            //     if (g.num_neighbors(node) > limit) {
            //         for (auto neighbor : g.neighbor_indices(node)) {
            //             auto subset = find_conditional_subset(g, node, neighbor, limit);
            //         }
            //     }
            // }



            ++limit;
        }

    }


}