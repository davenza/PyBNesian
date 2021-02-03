#include <indicators/cursor_control.hpp>
#include <learning/algorithms/mmhc.hpp>
#include <learning/algorithms/mmpc.hpp>
#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_whitelists.hpp>
#include <util/validate_options.hpp>

using models::ConditionalGaussianNetwork, models::ConditionalSemiparametricBN;

namespace learning::algorithms {

    void remove_asymmetries(std::vector<std::unordered_set<int>>& cpcs) {
        for (size_t i = 0; i < cpcs.size(); ++i) {
            for (auto it = cpcs[i].begin(), end = cpcs[i].end(); it != end;) {
                if (cpcs[*it].count(i) == 0) {
                    it = cpcs[i].erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    ArcSet create_hc_blacklist(BayesianNetworkBase& bn,
                               const std::vector<std::unordered_set<int>>& cpcs) {
        ArcSet blacklist;

        const auto& nodes = bn.nodes();

        for (auto i = 0, i_end = bn.num_nodes()-1; i < i_end; ++i) {
            auto index = bn.index(nodes[i]);
            for (auto j = i+1, j_end = bn.num_nodes(); j < j_end; ++j) {
                auto other_index = bn.index(nodes[j]);

                if (!cpcs[index].count(other_index)) {
                    blacklist.insert({index, other_index});
                    blacklist.insert({other_index, index});
                }
            }
        }


        return blacklist;
    }

    ArcSet create_conditional_hc_blacklist(ConditionalBayesianNetworkBase& bn,
                                           const std::vector<std::unordered_set<int>>& cpcs) {
        ArcSet blacklist;

        const auto& nodes = bn.nodes();

        for (auto i = 0, i_end = bn.num_nodes()-1; i < i_end; ++i) {
            auto index = bn.index(nodes[i]);
            for (auto j = i+1, j_end = bn.num_nodes(); j < j_end; ++j) {
                auto other_index = bn.index(nodes[j]);

                if (!cpcs[index].count(other_index)) {
                    blacklist.insert({index, other_index});
                    blacklist.insert({other_index, index});
                }
            }
        }

        for (const auto& node : bn.nodes()) {
            auto nindex = bn.index(node);
            for (const auto& inode : bn.interface_nodes()) {
                auto iindex = bn.index(inode);

                if (!cpcs[nindex].count(iindex)) {
                    blacklist.insert({iindex, nindex});
                }
            }
        }

        return blacklist;                                 
    }
    
    std::unique_ptr<BayesianNetworkBase> MMHC::estimate(const IndependenceTest& test,
                                                        OperatorSet& op_set,
                                                        Score& score,
                                                        Score* validation_score,
                                                        const std::vector<std::string>& nodes,
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

        auto score_type = score.type();
        if (score_type == ScoreType::PREDICTIVE_LIKELIHOOD && !validation_score)
            throw std::invalid_argument("A validation score is needed if predictive likelihood is used as score.");


        auto create_bn = [bn_type](const std::vector<std::string>& nodes) -> std::unique_ptr<BayesianNetworkBase> {
            switch (bn_type) {
                case BayesianNetworkType::Gaussian:
                    return std::make_unique<GaussianNetwork>(nodes);
                case BayesianNetworkType::Semiparametric:
                    return std::make_unique<SemiparametricBN>(nodes);
                default:
                    throw std::invalid_argument("Wrong BayesianNetwork type. Unreachable code!");
            }
        };

        PartiallyDirectedGraph skeleton;
        std::unique_ptr<BayesianNetworkBase> bn;
        if (nodes.empty()) {
            skeleton = PartiallyDirectedGraph(test.variable_names());
            bn = create_bn(test.variable_names());
            
            if (!score.has_variables(skeleton.nodes()) || (validation_score && !validation_score->has_variables(skeleton.nodes())))
                throw std::invalid_argument("Score do not contain all the variables in nodes list.");

        } else {
            if (!test.has_variables(nodes))
                throw std::invalid_argument("IndependenceTest do not contain all the variables in nodes list.");
            if (!score.has_variables(nodes) || (validation_score && !validation_score->has_variables(nodes)))
                throw std::invalid_argument("Score do not contain all the variables in nodes list.");

            skeleton = PartiallyDirectedGraph(nodes);
            bn = create_bn(nodes);
        }

        auto restrictions = util::validate_restrictions(skeleton, 
                                                        varc_blacklist,
                                                        varc_whitelist,
                                                        vedge_blacklist,
                                                        vedge_whitelist);

        indicators::show_console_cursor(false);
        auto progress = util::progress_bar(verbose);
        auto cpcs = mmpc_all_variables(test, skeleton, alpha, restrictions.arc_whitelist, 
                                       restrictions.edge_blacklist, restrictions.edge_whitelist, *progress);

        remove_asymmetries(cpcs);
        auto hc_blacklist = create_hc_blacklist(*bn, cpcs);
        indicators::show_console_cursor(true);

        if (score_type == ScoreType::PREDICTIVE_LIKELIHOOD) {
            return learning::algorithms::estimate_validation_hc(op_set,
                                                                score,
                                                                *validation_score,
                                                                *bn,
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
                                                     *bn,
                                                     hc_blacklist,
                                                     restrictions.arc_whitelist,
                                                     max_indegree,
                                                     max_iters,
                                                     epsilon,
                                                     verbose);
        }
    }

    std::unique_ptr<ConditionalBayesianNetworkBase> MMHC::estimate_conditional(const IndependenceTest& test,
                                                                               OperatorSet& op_set,
                                                                               Score& score,
                                                                               Score* validation_score,
                                                                               const std::vector<std::string>& nodes,
                                                                               const std::vector<std::string>& interface_nodes,
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

        if (nodes.empty())
            throw std::invalid_argument("Node list cannot be empty to train a Conditional Bayesian network.");
        if (interface_nodes.empty())
            return MMHC::estimate(test, op_set, score, validation_score, nodes, bn_str, 
                                  varc_blacklist, varc_whitelist, vedge_blacklist, vedge_whitelist,
                                  type_whitelist, max_indegree, max_iters, epsilon, patience, alpha, verbose)->conditional_bn();

        if (!test.has_variables(nodes) || !test.has_variables(interface_nodes))
            throw std::invalid_argument("IndependenceTest do not contain all the variables in nodes/interface_nodes lists.");
        if (!score.has_variables(nodes) || !score.has_variables(interface_nodes) || 
            (validation_score && (!validation_score->has_variables(nodes) || !validation_score->has_variables(interface_nodes))))
            throw std::invalid_argument("Score do not contain all the variables in nodes list.");

        auto score_type = score.type();
        if (score_type == ScoreType::PREDICTIVE_LIKELIHOOD && !validation_score)
            throw std::invalid_argument("A validation score is needed if predictive likelihood is used as score.");


        ConditionalPartiallyDirectedGraph skeleton(nodes, interface_nodes);

        auto bn_type = util::check_valid_bn_string(bn_str);
        auto bn = [bn_type, &nodes, &interface_nodes]() 
                    -> std::unique_ptr<ConditionalBayesianNetworkBase> {
            switch (bn_type) {
                case BayesianNetworkType::Gaussian:
                    return std::make_unique<ConditionalGaussianNetwork>(nodes, interface_nodes);
                case BayesianNetworkType::Semiparametric:
                    return std::make_unique<ConditionalSemiparametricBN>(nodes, interface_nodes);
                default:
                    throw std::invalid_argument("Wrong ConditionalBayesianNetwork type. Unreachable code!");
            }
        }();

        auto restrictions = util::validate_restrictions(skeleton, 
                                                varc_blacklist,
                                                varc_whitelist,
                                                vedge_blacklist,
                                                vedge_whitelist);

        indicators::show_console_cursor(false);
        auto progress = util::progress_bar(verbose);
        auto cpcs = mmpc_all_variables(test, skeleton, alpha, restrictions.arc_whitelist, 
                                       restrictions.edge_blacklist, restrictions.edge_whitelist, *progress);
        remove_asymmetries(cpcs);
        auto hc_blacklist = create_conditional_hc_blacklist(*bn, cpcs);
        indicators::show_console_cursor(true);

        if (score_type == ScoreType::PREDICTIVE_LIKELIHOOD) {
            return learning::algorithms::estimate_validation_hc(op_set,
                                                                score,
                                                                *validation_score,
                                                                *bn,
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
                                                     *bn,
                                                     hc_blacklist,
                                                     restrictions.arc_whitelist,
                                                     max_indegree,
                                                     max_iters,
                                                     epsilon,
                                                     verbose);
        }
    }
}