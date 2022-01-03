#include <memory>
#include <models/DynamicBayesianNetwork.hpp>
#include <learning/algorithms/mmhc.hpp>
#include <learning/algorithms/dmmhc.hpp>
#include <util/validate_options.hpp>

using models::ConditionalBayesianNetwork, models::ConditionalGaussianNetwork, models::ConditionalSemiparametricBN;
using models::DynamicBayesianNetwork, models::DynamicGaussianNetwork, models::DynamicSemiparametricBN;

namespace learning::algorithms {

ArcStringVector static_blacklist(const std::vector<std::string>& variables, int markovian_order) {
    if (markovian_order == 1) return ArcStringVector();

    ArcStringVector blacklist;
    blacklist.reserve(variables.size() * variables.size() * markovian_order * (markovian_order - 1) / 2);

    auto slice_names = util::temporal_slice_names(variables, 1, markovian_order);

    for (int i = 0, end = markovian_order - 1; i < end; ++i) {
        for (const auto& source : slice_names[i]) {
            for (auto j = i + 1; j < markovian_order; ++j) {
                for (const auto& dest : slice_names[j]) {
                    blacklist.push_back(std::make_pair(source, dest));
                }
            }
        }
    }

    return blacklist;
};

std::shared_ptr<DynamicBayesianNetworkBase> DMMHC::estimate(const DynamicIndependenceTest& test,
                                                            OperatorSet& op_set,
                                                            DynamicScore& score,
                                                            const std::vector<std::string>& variables,
                                                            const BayesianNetworkType& bn_type,
                                                            int markovian_order,
                                                            const std::shared_ptr<Callback> static_callback,
                                                            const std::shared_ptr<Callback> transition_callback,
                                                            int max_indegree,
                                                            int max_iters,
                                                            double epsilon,
                                                            int patience,
                                                            double alpha,
                                                            int verbose) {
    std::vector<std::string> vars;
    if (variables.empty())
        vars = test.variable_names();
    else {
        if (!test.has_variables(variables))
            throw std::invalid_argument("DynamicIndependenceTest do not contain all the variables in nodes lists.");
        vars = variables;
    }

    if (!score.has_variables(vars))
        throw std::invalid_argument("Score do not contain all the variables in nodes list.");

    MMHC mmhc;

    auto static_nodes = util::temporal_names(vars, 1, markovian_order);
    auto static_arc_blacklist = static_blacklist(vars, markovian_order);
    const auto& static_tests = test.static_tests();

    auto& static_score = score.static_score();

    ArcStringVector arc_blacklist;
    ArcStringVector arc_whitelist;
    EdgeStringVector edge_blacklist;
    EdgeStringVector edge_whitelist;
    FactorTypeVector type_blacklist;
    FactorTypeVector type_whitelist;

    auto g0 = mmhc.estimate(static_tests,
                            op_set,
                            static_score,
                            static_nodes,
                            bn_type,
                            static_arc_blacklist,
                            arc_whitelist,
                            edge_blacklist,
                            edge_whitelist,
                            type_blacklist,
                            type_whitelist,
                            static_callback,
                            max_indegree,
                            max_iters,
                            epsilon,
                            patience,
                            alpha,
                            verbose);

    auto transition_nodes = util::temporal_names(vars, 0, 0);
    const auto& transition_tests = test.transition_tests();
    auto& transition_score = score.transition_score();

    auto gt = mmhc.estimate_conditional(transition_tests,
                                        op_set,
                                        transition_score,
                                        transition_nodes,
                                        static_nodes,
                                        bn_type,
                                        arc_blacklist,
                                        arc_whitelist,
                                        edge_blacklist,
                                        edge_whitelist,
                                        type_blacklist,
                                        type_whitelist,
                                        transition_callback,
                                        max_indegree,
                                        max_iters,
                                        epsilon,
                                        patience,
                                        alpha,
                                        verbose);

    return std::make_shared<DynamicBayesianNetwork>(vars, markovian_order, std::move(g0), std::move(gt));
}

}  // namespace learning::algorithms
