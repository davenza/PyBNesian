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


    void PC::estimate(const DataFrame& df, 
                                 ArcVector& arc_blacklist, 
                                 ArcVector& arc_whitelist, 
                                 const IndependenceTest& test) {
        GaussianNetwork::requires(df);

        auto g = UndirectedGraph::Complete(df.column_names());


        auto limit = 0;

        while(!max_cardinality(g, limit + 1)) {
            
            // for (int node = 0; node < )



            ++limit;
        }

    }


}