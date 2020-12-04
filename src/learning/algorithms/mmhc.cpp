#include <indicators/cursor_control.hpp>
#include <learning/algorithms/mmhc.hpp>
#include <learning/algorithms/mmpc.hpp>
#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_whitelists.hpp>
#include <util/validate_options.hpp>

namespace learning::algorithms {

    void remove_asymmetries(std::vector<std::unordered_set<int>> cpcs) {
        for (size_t i = 0; i < cpcs.size(); ++i) {
            for (auto it = cpcs[i].begin(), end = cpcs[i].end(); it != end; ++it) {
                if (cpcs[*it].count(i) == 0) {
                    it = cpcs[i].erase(it);
                }
            }
        }
    }

    ArcSet create_hc_blacklist(std::vector<std::unordered_set<int>> cpcs) {
        ArcSet blacklist;

        for (int i = 0, i_end = cpcs.size()-1; i < i_end; ++i) {
            for(int j = i + 1, j_end = cpcs.size(); j < j_end; ++j) {
                if (cpcs[i].count(j) == 0) {
                    blacklist.insert({i, j});
                    blacklist.insert({j, i});
                }
            }
        }

        return blacklist;
    }
    
    std::unique_ptr<BayesianNetworkBase> MMHC::estimate(const IndependenceTest& test,
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
                                                        int verbose) {

        auto bn_type = util::check_valid_bn_string(bn_str);

        std::unique_ptr<BayesianNetworkBase> skeleton = [bn_type, &test]() -> std::unique_ptr<BayesianNetworkBase> {
            switch (bn_type) {
                case BayesianNetworkType::GBN:
                    return std::make_unique<GaussianNetwork>(test.column_names());
                case BayesianNetworkType::SPBN:
                    return std::make_unique<SemiparametricBN>(test.column_names());
                default:
                    throw std::invalid_argument("Wrong BayesianNetwork type. Unreachable code!");
            }
        }();
        
        auto restrictions = util::validate_restrictions(*skeleton, 
                                                        varc_blacklist,
                                                        varc_whitelist,
                                                        vedge_blacklist,
                                                        vedge_whitelist);
        indicators::show_console_cursor(false);
        auto progress = util::progress_bar(verbose);
    
        auto cpcs = mmpc_all_variables(test, alpha, restrictions.arc_whitelist, 
                                restrictions.edge_blacklist, restrictions.edge_whitelist, *progress);
        
        remove_asymmetries(cpcs);

        auto hc_blacklist = create_hc_blacklist(cpcs);

        auto score_type = score.type();

        if (score_type == ScoreType::PREDICTIVE_LIKELIHOOD) {
            if (!validation_score)
                throw std::invalid_argument("A validation score is needed if predictive likelihood is used as score.");
            return learning::algorithms::estimate_validation_hc(op_set,
                                                                score,
                                                                *validation_score,
                                                                *skeleton,
                                                                hc_blacklist,
                                                                restrictions.arc_whitelist,
                                                                type_whitelist,
                                                                max_indegree,
                                                                max_iters,
                                                                epsilon,
                                                                patience,
                                                                verbose);
        } else {
            return learning::algorithms::estimate_hc(op_set,
                                                     score,
                                                     *skeleton,
                                                     hc_blacklist,
                                                     restrictions.arc_whitelist,
                                                     max_indegree,
                                                     max_iters,
                                                     epsilon,
                                                     verbose);
        }

        indicators::show_console_cursor(true);

    }
}