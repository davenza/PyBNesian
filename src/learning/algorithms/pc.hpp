#ifndef PGM_DATASET_PC_HPP
#define PGM_DATASET_PC_HPP

#include <util/util_types.hpp>
#include <graph/pdag.hpp>
#include <models/GaussianNetwork.hpp>
#include <learning/independences/independence.hpp>

using util::ArcVector;
using graph::PartiallyDirectedGraph;
using models::GaussianNetwork;
using learning::independences::IndependenceTest;

namespace learning::algorithms {
    

    class PC {
    public:
        PartiallyDirectedGraph estimate(const DataFrame& df, 
                        ArcVector& arc_blacklist, 
                        ArcVector& arc_whitelist, 
                        const IndependenceTest& test,
                        double alpha,
                        double ambiguous_threshold,
                        double ambiguous_slack);

    };

    class MeekRules {
    public:
        static bool rule1(PartiallyDirectedGraph& pdag);
        static bool rule2(PartiallyDirectedGraph& pdag);
        static bool rule3(PartiallyDirectedGraph& pdag);
    };
}

#endif //PGM_DATASET_PC_HPP