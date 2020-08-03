#include <optional>

#include <learning/algorithms/pc.hpp>
#include <graph/undirected.hpp>

using graph::UndirectedGraph;

namespace learning::algorithms {


    bool max_cardinality(UndirectedGraph& g, int max_cardinality) {
        for (int i = 0; i < g.num_nodes(); ++i) {
            if (g.num_neighbors_unsafe(i) > max_cardinality)
                return false;
        }
        return true;
    }

    std::optional<std::unordered_set<int>> find_conditional_subset(UndirectedGraph& g, int var1, int var2, int sep_size) {
    
        
        

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
            
            for (int node = 0; node < g.num_nodes(); ++node) {
                if (g.num_neighbors(node) > limit) {
                    for (auto neighbor : g.neighbor_indices(node)) {
                        auto subset = find_conditional_subset(g, node, neighbor, limit);
                    }
                }
            }



            ++limit;
        }

    }


}