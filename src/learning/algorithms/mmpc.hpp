#ifndef PYBNESIAN_LEARNING_ALGORITHMS_MMPC_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_MMPC_HPP

#include <graph/generic_graph.hpp>
#include <learning/independences/independence.hpp>
#include <util/progress.hpp>

using graph::PartiallyDirectedGraph;
using learning::independences::IndependenceTest;
using util::ArcStringVector, util::EdgeStringVector;

namespace learning::algorithms {

    std::unordered_set<int> mmpc_variable(const IndependenceTest& test,
                                          int variable,
                                          double alpha,
                                          const ArcSet& arc_whitelist,
                                          const EdgeSet& edge_blacklist,
                                          const EdgeSet& edge_whitelist,
                                          util::BaseProgressBar& progress);

    std::vector<std::unordered_set<int>> mmpc_all_variables(const IndependenceTest& test,
                                                            double alpha,
                                                            const ArcSet& arc_whitelist,
                                                            const EdgeSet& edge_blacklist,
                                                            const EdgeSet& edge_whitelist,
                                                            util::BaseProgressBar& progress);
                                                            
    void mmpc_all_variables(const IndependenceTest& test,
                            std::vector<std::unordered_set<int>>& cpcs,
                            std::vector<std::unordered_set<int>>& to_be_checked,
                            double alpha,
                            const ArcSet& arc_whitelist,
                            const EdgeSet& edge_blacklist,
                            const EdgeSet& edge_whitelist,
                            util::BaseProgressBar& progress);

    

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
        
        PartiallyDirectedGraph estimate_conditional(const IndependenceTest& test,
                                                    const std::vector<std::string>& nodes,
                                                    const std::vector<std::string>& interface_nodes,
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