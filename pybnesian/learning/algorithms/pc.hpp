#ifndef PYBNESIAN_LEARNING_ALGORITHMS_PC_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_PC_HPP

#include <graph/generic_graph.hpp>
#include <learning/independences/independence.hpp>

using graph::PartiallyDirectedGraph, graph::ConditionalPartiallyDirectedGraph;
using learning::independences::IndependenceTest;
using util::ArcStringVector;

namespace learning::algorithms {

class PC {
public:
    PartiallyDirectedGraph estimate(const IndependenceTest& test,
                                    const std::vector<std::string>& nodes,
                                    const ArcStringVector& arc_blacklist,
                                    const ArcStringVector& arc_whitelist,
                                    const EdgeStringVector& edge_blacklist,
                                    const EdgeStringVector& edge_whitelist,
                                    double alpha,
                                    bool use_sepsets,
                                    double ambiguous_threshold,
                                    bool allow_bidirected,
                                    int verbose) const;

    ConditionalPartiallyDirectedGraph estimate_conditional(const IndependenceTest& test,
                                                           const std::vector<std::string>& nodes,
                                                           const std::vector<std::string>& interface_nodes,
                                                           const ArcStringVector& arc_blacklist,
                                                           const ArcStringVector& arc_whitelist,
                                                           const EdgeStringVector& edge_blacklist,
                                                           const EdgeStringVector& edge_whitelist,
                                                           double alpha,
                                                           bool use_sepsets,
                                                           double ambiguous_threshold,
                                                           bool allow_bidirected,
                                                           int verbose) const;
};

}  // namespace learning::algorithms

#endif  // PYBNESIAN_LEARNING_ALGORITHMS_PC_HPP