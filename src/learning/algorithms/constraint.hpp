#ifndef PYBNESIAN_LEARNING_ALGORITHMS_CONSTRAINT_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_CONSTRAINT_HPP

#include <graph/generic_graph.hpp>
#include <util/progress.hpp>
#include <learning/independences/independence.hpp>

using graph::PartiallyDirectedGraph;
using learning::independences::IndependenceTest;
using util::BaseProgressBar;

namespace learning::algorithms {

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

    void direct_arc_blacklist(PartiallyDirectedGraph& g, const ArcSet& arc_blacklist);
    void direct_unshielded_triples(PartiallyDirectedGraph& pdag,
                                   const IndependenceTest& test,
                                   const ArcSet& arc_blacklist,
                                   const ArcSet& arc_whitelist,
                                   double alpha,
                                   const std::optional<SepSet>&  sepset,
                                   bool use_sepsets,
                                   double ambiguous_threshold,
                                   bool allow_bidirected,
                                   util::BaseProgressBar* progress);

    class MeekRules {
    public:
        static bool rule1(PartiallyDirectedGraph& pdag);
        static bool rule2(PartiallyDirectedGraph& pdag);
        static bool rule3(PartiallyDirectedGraph& pdag);
    };
}

#endif //PYBNESIAN_LEARNING_ALGORITHMS_CONSTRAINT_HPP