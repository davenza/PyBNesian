#include <memory>
#include <learning/algorithms/mmhc.hpp>
#include <learning/algorithms/dmmhc.hpp>
#include <util/validate_options.hpp>

using models::DynamicBayesianNetwork;
namespace learning::algorithms {


    std::unique_ptr<DynamicBayesianNetworkBase> DMMHC::estimate(const DynamicIndependenceTest& test,
                                                                OperatorSet& op_set,
                                                                DynamicScore& score,
                                                                DynamicScore* validation_score,
                                                                const std::string& bn_str,
                                                                // const ArcStringVector& varc_blacklist,
                                                                // const ArcStringVector& varc_whitelist,
                                                                // const EdgeStringVector& vedge_blacklist,
                                                                // const EdgeStringVector& vedge_whitelist,
                                                                // const FactorStringTypeVector& type_whitelist,
                                                                int max_indegree,
                                                                int max_iters, 
                                                                double epsilon,
                                                                int patience,
                                                                double alpha,
                                                                int verbose) {

        MMHC mmhc;

        const auto& static_tests = test.static_tests();
        auto static_blacklist = test.static_blacklist();
        auto& static_score = score.static_score();
        Score* validation_static_score = nullptr;
        if (validation_score)
            validation_static_score = &validation_score->static_score();

        auto g0 = mmhc.estimate(static_tests,
                                op_set,
                                static_score,
                                validation_static_score,
                                bn_str,
                                static_blacklist,
                                ArcStringVector(),
                                EdgeStringVector(),
                                EdgeStringVector(),
                                FactorStringTypeVector(),
                                // varc_blacklist,
                                // varc_whitelist,
                                // vedge_blacklist,
                                // vedge_whitelist,
                                // type_whitelist,
                                max_indegree,
                                max_iters,
                                epsilon,
                                patience,
                                alpha,
                                verbose);
    
        const auto& transition_tests = test.transition_tests();
        auto transition_blacklist = test.transition_blacklist();
        auto& transition_score = score.transition_score();
        Score* validation_transition_score = nullptr;
        if (validation_score)
            validation_transition_score = &validation_score->transition_score();

        auto gt = mmhc.estimate(transition_tests,
                                op_set,
                                transition_score,
                                validation_transition_score,
                                bn_str,
                                transition_blacklist,
                                ArcStringVector(),
                                EdgeStringVector(),
                                EdgeStringVector(),
                                FactorStringTypeVector(),
                                // varc_blacklist,
                                // varc_whitelist,
                                // vedge_blacklist,
                                // vedge_whitelist,
                                // type_whitelist,
                                max_indegree,
                                max_iters,
                                epsilon,
                                patience,
                                alpha,
                                verbose);
        

        auto bn_type = util::check_valid_bn_string(bn_str);
        // auto dynamic_bn = [bn_type, &g0, &gt]() -> std::unique_ptr<DynamicBayesianNetworkBase> {
        //     switch (bn_type) {
        //         case BayesianNetworkType::GBN:
        //             return std::make_unique<DynamicBayesianNetwork<GaussianNetwork>>(
        //                             *static_cast<GaussianNetwork*>(g0.release()), 
        //                             *static_cast<GaussianNetwork*>(gt.release()));
        //         case BayesianNetworkType::SPBN:
        //             return std::make_unique<DynamicBayesianNetwork<SemiparametricBN>>(
        //                             *static_cast<SemiparametricBN*>(g0.release()),
        //                             *static_cast<SemiparametricBN*>(gt.release()));
        //         default:
        //             throw std::invalid_argument("Wrong BayesianNetwork type. Unreachable code!");
        //     }
        // }();

        // return dynamic_bn;
    }

}