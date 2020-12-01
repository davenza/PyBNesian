#ifndef PYBNESIAN_LEARNING_ALGORITHMS_MMPC_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_MMPC_HPP

#include <graph/generic_graph.hpp>
#include <learning/independences/independence.hpp>

using graph::PartiallyDirectedGraph;
using learning::independences::IndependenceTest;
using util::ArcStringVector, util::EdgeStringVector;

namespace learning::algorithms {

    class MMPC {
    public:
        PartiallyDirectedGraph estimate(const IndependenceTest& test,
                        const ArcStringVector& arc_blacklist, 
                        const ArcStringVector& arc_whitelist,
                        const EdgeStringVector& edge_blacklist,
                        const EdgeStringVector& edge_whitelist,
                        double alpha,
                        double ambiguous_threshold,
                        bool allow_bidirected,
                        int verbose) const;
    };

}

#endif //PYBNESIAN_LEARNING_ALGORITHMS_MMPC_HPP