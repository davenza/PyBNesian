#ifndef PYBNESIAN_LEARNING_ALGORITHMS_PC_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_PC_HPP

#include <util/util_types.hpp>
#include <graph/generic_graph.hpp>
#include <models/GaussianNetwork.hpp>
#include <learning/independences/independence.hpp>

using util::ArcStringVector;
using graph::PartiallyDirectedGraph;
using models::GaussianNetwork;
using learning::independences::IndependenceTest;

namespace learning::algorithms {
    

    class PC {
    public:
        PartiallyDirectedGraph estimate(const IndependenceTest& test,
                        const ArcStringVector& arc_blacklist, 
                        const ArcStringVector& arc_whitelist,
                        const EdgeStringVector& edge_blacklist,
                        const EdgeStringVector& edge_whitelist,
                        double alpha,
                        bool use_sepsets,
                        double ambiguous_threshold,
                        bool allow_bidirected,
                        int verbose);

    };

    class MeekRules {
    public:
        static bool rule1(PartiallyDirectedGraph& pdag);
        static bool rule2(PartiallyDirectedGraph& pdag);
        static bool rule3(PartiallyDirectedGraph& pdag);
    };
}

#endif //PYBNESIAN_LEARNING_ALGORITHMS_PC_HPP