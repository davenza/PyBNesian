#include <memory>
#include <learning/algorithms/mmhc.hpp>
#include <learning/algorithms/dmmhc.hpp>
#include <util/validate_options.hpp>

using models::ConditionalBayesianNetwork, models::ConditionalGaussianNetwork,
      models::ConditionalSemiparametricBN;
using models::DynamicBayesianNetwork, models::DynamicGaussianNetwork,
      models::DynamicSemiparametricBN;

namespace learning::algorithms {

    ArcStringVector static_blacklist(const std::vector<std::string>& variables, int markovian_order) {
        if (markovian_order == 1)
            return ArcStringVector();

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

    std::unique_ptr<DynamicBayesianNetworkBase> DMMHC::estimate(const DynamicIndependenceTest& test,
                                                                OperatorSet& op_set,
                                                                DynamicScore& score,
                                                                DynamicScore* validation_score,
                                                                const std::vector<std::string>& variables,
                                                                const std::string& bn_str,
                                                                int markovian_order,
                                                                int max_indegree,
                                                                int max_iters, 
                                                                double epsilon,
                                                                int patience,
                                                                double alpha,
                                                                int verbose) {
        auto bn_type = util::check_valid_bn_string(bn_str);
        if (!test.has_variables(variables))
            throw std::invalid_argument("DynamicIndependenceTest do not contain all the variables in nodes lists.");
        if (!score.has_variables(variables) || (validation_score && !validation_score->has_variables(variables)))
            throw std::invalid_argument("Score do not contain all the variables in nodes list.");

        std::vector<std::string> vars;
        if (variables.empty())
            vars = test.variable_names();
        else
            vars = variables;


        MMHC mmhc;

        auto static_nodes = util::temporal_names(vars, 1, markovian_order);
        auto static_arc_blacklist = static_blacklist(vars, markovian_order);
        const auto& static_tests = test.static_tests();

        auto& static_score = score.static_score();
        Score* validation_static_score = nullptr;
        if (validation_score)
            validation_static_score = &validation_score->static_score();

        auto g0 = mmhc.estimate(static_tests,
                                op_set,
                                static_score,
                                validation_static_score,
                                static_nodes,
                                bn_str,
                                static_arc_blacklist,
                                ArcStringVector(),
                                EdgeStringVector(),
                                EdgeStringVector(),
                                FactorStringTypeVector(),
                                max_indegree,
                                max_iters,
                                epsilon,
                                patience,
                                alpha,
                                verbose);
    
        auto transition_nodes = util::temporal_names(vars, 0, 0);
        const auto& transition_tests = test.transition_tests();

        auto& transition_score = score.transition_score();
        Score* validation_transition_score = nullptr;
        if (validation_score)
            validation_transition_score = &validation_score->transition_score();

        auto gt = mmhc.estimate_conditional(transition_tests,
                                            op_set,
                                            transition_score,
                                            validation_transition_score,
                                            transition_nodes,
                                            static_nodes,
                                            bn_str,
                                            ArcStringVector(),
                                            ArcStringVector(),
                                            EdgeStringVector(),
                                            EdgeStringVector(),
                                            FactorStringTypeVector(),
                                            max_indegree,
                                            max_iters,
                                            epsilon,
                                            patience,
                                            alpha,
                                            verbose);
        
        auto dynamic_bn = [bn_type, &vars, markovian_order, &g0, &gt]() -> std::unique_ptr<DynamicBayesianNetworkBase> {
            switch (bn_type) {
                case BayesianNetworkType::Gaussian:
                    return std::make_unique<DynamicGaussianNetwork>(
                                    vars,
                                    markovian_order,
                                    *static_cast<GaussianNetwork*>(g0.release()), 
                                    *static_cast<ConditionalGaussianNetwork*>(gt.release()));
                case BayesianNetworkType::Semiparametric:
                    return std::make_unique<DynamicSemiparametricBN>(
                                    vars,
                                    markovian_order,
                                    *static_cast<SemiparametricBN*>(g0.release()),
                                    *static_cast<ConditionalSemiparametricBN*>(gt.release()));
                default:
                    throw std::invalid_argument("Wrong BayesianNetwork type. Unreachable code!");
            }
        }();

        return dynamic_bn;
    }

}