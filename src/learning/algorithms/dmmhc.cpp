#include <learning/algorithms/mmhc.hpp>
#include <learning/algorithms/dmmhc.hpp>

namespace learning::algorithms {


    std::unique_ptr<DynamicBayesianNetworkBase> DMMHC::estimate(const IndependenceTest& test,
                                                                OperatorSet& op_set,
                                                                Score& score,
                                                                Score* validation_score,
                                                                const std::string& bn_str,
                                                                const ArcStringVector& varc_blacklist,
                                                                const ArcStringVector& varc_whitelist,
                                                                const EdgeStringVector& vedge_blacklist,
                                                                const EdgeStringVector& vedge_whitelist,
                                                                const FactorStringTypeVector& type_whitelist,
                                                                int max_indegree,
                                                                int max_iters, 
                                                                double epsilon,
                                                                int patience,
                                                                double alpha,
                                                                int markovian_order,
                                                                int verbose) {

        MMHC mmhc;
        
        auto g0 = mmhc.estimate(test,
                                op_set,
                                score,
                                validation_score,
                                bn_str,
                                varc_blacklist,
                                varc_whitelist,
                                vedge_blacklist,
                                vedge_whitelist,
                                type_whitelist,
                                max_indegree,
                                max_iters,
                                epsilon,
                                patience,
                                alpha,
                                verbose);
    





        



    }

}