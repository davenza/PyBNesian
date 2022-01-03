#include <learning/algorithms/mmhc.hpp>
#include <learning/algorithms/mmpc.hpp>
#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_whitelists.hpp>
#include <util/validate_options.hpp>

using models::BayesianNetwork, models::ConditionalBayesianNetwork, models::ConditionalGaussianNetwork,
    models::ConditionalSemiparametricBN;

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

ArcStringVector create_hc_blacklist(BayesianNetworkBase& bn, const std::vector<std::unordered_set<int>>& cpcs) {
    ArcStringVector blacklist;

    const auto& nodes = bn.nodes();

    for (auto i = 0, i_end = bn.num_nodes() - 1; i < i_end; ++i) {
        auto index = bn.index(nodes[i]);
        for (auto j = i + 1, j_end = bn.num_nodes(); j < j_end; ++j) {
            auto other_index = bn.index(nodes[j]);

            if (!cpcs[index].count(other_index)) {
                blacklist.push_back({nodes[index], nodes[other_index]});
                blacklist.push_back({nodes[other_index], nodes[index]});
            }
        }
    }

    return blacklist;
}

ArcStringVector create_conditional_hc_blacklist(ConditionalBayesianNetworkBase& bn,
                                                const std::vector<std::unordered_set<int>>& cpcs) {
    ArcStringVector blacklist;

    const auto& nodes = bn.nodes();

    for (auto i = 0, i_end = bn.num_nodes() - 1; i < i_end; ++i) {
        auto index = bn.index(nodes[i]);
        for (auto j = i + 1, j_end = bn.num_nodes(); j < j_end; ++j) {
            auto other_index = bn.index(nodes[j]);

            if (!cpcs[index].count(other_index)) {
                blacklist.push_back({nodes[index], nodes[other_index]});
                blacklist.push_back({nodes[other_index], nodes[index]});
            }
        }
    }

    for (const auto& node : nodes) {
        auto nindex = bn.index(node);
        for (const auto& inode : bn.interface_nodes()) {
            auto iindex = bn.index(inode);

            if (!cpcs[nindex].count(iindex)) {
                blacklist.push_back({inode, node});
            }
        }
    }

    return blacklist;
}

std::shared_ptr<BayesianNetworkBase> MMHC::estimate(const IndependenceTest& test,
                                                    OperatorSet& op_set,
                                                    Score& score,
                                                    const std::vector<std::string>& nodes,
                                                    const BayesianNetworkType& bn_type,
                                                    const ArcStringVector& varc_blacklist,
                                                    const ArcStringVector& varc_whitelist,
                                                    const EdgeStringVector& vedge_blacklist,
                                                    const EdgeStringVector& vedge_whitelist,
                                                    const FactorTypeVector& type_blacklist,
                                                    const FactorTypeVector& type_whitelist,
                                                    const std::shared_ptr<Callback> callback,
                                                    int max_indegree,
                                                    int max_iters,
                                                    double epsilon,
                                                    int patience,
                                                    double alpha,
                                                    int verbose) {
    PartiallyDirectedGraph skeleton;
    std::shared_ptr<BayesianNetworkBase> bn;
    if (nodes.empty()) {
        skeleton = PartiallyDirectedGraph(test.variable_names());
        bn = bn_type.new_bn(test.variable_names());
    } else {
        if (!test.has_variables(nodes))
            throw std::invalid_argument("IndependenceTest do not contain all the variables in nodes list.");
        skeleton = PartiallyDirectedGraph(nodes);
        bn = bn_type.new_bn(nodes);
    }

    if (!score.compatible_bn(*bn)) {
        throw std::invalid_argument("BayesianNetwork is not compatible with the score.");
    }

    if (!score.has_variables(skeleton.nodes()))
        throw std::invalid_argument("Score do not contain all the variables in nodes list.");

    auto restrictions =
        util::validate_restrictions(skeleton, varc_blacklist, varc_whitelist, vedge_blacklist, vedge_whitelist);
    util::validate_type_restrictions(skeleton, type_blacklist, type_whitelist);

    auto progress = util::progress_bar(verbose);
    auto cpcs = mmpc_all_variables(test,
                                   skeleton,
                                   alpha,
                                   restrictions.arc_whitelist,
                                   restrictions.edge_blacklist,
                                   restrictions.edge_whitelist,
                                   *progress);

    remove_asymmetries(cpcs);

    auto hc_blacklist = create_hc_blacklist(*bn, cpcs);
    hc_blacklist.insert(hc_blacklist.end(), varc_blacklist.begin(), varc_blacklist.end());
    ArcStringVector arc_whitelist;
    arc_whitelist.reserve(restrictions.arc_whitelist.size());

    for (auto& p : restrictions.arc_whitelist) {
        arc_whitelist.push_back({skeleton.name(p.first), skeleton.name(p.second)});
    }

    return learning::algorithms::estimate_downcast_score(op_set,
                                                         score,
                                                         *bn,
                                                         hc_blacklist,
                                                         arc_whitelist,
                                                         type_blacklist,
                                                         type_whitelist,
                                                         callback,
                                                         max_indegree,
                                                         max_iters,
                                                         epsilon,
                                                         patience,
                                                         verbose);
}

std::shared_ptr<ConditionalBayesianNetworkBase> MMHC::estimate_conditional(
    const IndependenceTest& test,
    OperatorSet& op_set,
    Score& score,
    const std::vector<std::string>& nodes,
    const std::vector<std::string>& interface_nodes,
    const BayesianNetworkType& bn_type,
    const ArcStringVector& varc_blacklist,
    const ArcStringVector& varc_whitelist,
    const EdgeStringVector& vedge_blacklist,
    const EdgeStringVector& vedge_whitelist,
    const FactorTypeVector& type_blacklist,
    const FactorTypeVector& type_whitelist,
    const std::shared_ptr<Callback> callback,
    int max_indegree,
    int max_iters,
    double epsilon,
    int patience,
    double alpha,
    int verbose) {
    if (nodes.empty())
        throw std::invalid_argument("Node list cannot be empty to train a Conditional Bayesian network.");
    if (interface_nodes.empty())
        return MMHC::estimate(test,
                              op_set,
                              score,
                              nodes,
                              bn_type,
                              varc_blacklist,
                              varc_whitelist,
                              vedge_blacklist,
                              vedge_whitelist,
                              type_blacklist,
                              type_whitelist,
                              callback,
                              max_indegree,
                              max_iters,
                              epsilon,
                              patience,
                              alpha,
                              verbose)
            ->conditional_bn();

    if (!test.has_variables(nodes) || !test.has_variables(interface_nodes))
        throw std::invalid_argument(
            "IndependenceTest do not contain all the variables in nodes/interface_nodes lists.");
    if (!score.has_variables(nodes) || !score.has_variables(interface_nodes))
        throw std::invalid_argument("Score do not contain all the variables in nodes list.");

    ConditionalPartiallyDirectedGraph skeleton(nodes, interface_nodes);

    auto bn = bn_type.new_cbn(nodes, interface_nodes);

    if (!score.compatible_bn(*bn)) {
        throw std::invalid_argument("BayesianNetwork is not compatible with the score.");
    }

    auto restrictions =
        util::validate_restrictions(skeleton, varc_blacklist, varc_whitelist, vedge_blacklist, vedge_whitelist);
    util::validate_type_restrictions(skeleton, type_blacklist, type_whitelist);

    auto progress = util::progress_bar(verbose);
    auto cpcs = mmpc_all_variables(test,
                                   skeleton,
                                   alpha,
                                   restrictions.arc_whitelist,
                                   restrictions.edge_blacklist,
                                   restrictions.edge_whitelist,
                                   *progress);
    remove_asymmetries(cpcs);
    auto hc_blacklist = create_conditional_hc_blacklist(*bn, cpcs);

    hc_blacklist.insert(hc_blacklist.end(), varc_blacklist.begin(), varc_blacklist.end());
    ArcStringVector arc_whitelist;
    arc_whitelist.reserve(restrictions.arc_whitelist.size());

    for (auto& p : restrictions.arc_whitelist) {
        arc_whitelist.push_back({skeleton.name(p.first), skeleton.name(p.second)});
    }

    return learning::algorithms::estimate_downcast_score(op_set,
                                                         score,
                                                         *bn,
                                                         hc_blacklist,
                                                         arc_whitelist,
                                                         type_blacklist,
                                                         type_whitelist,
                                                         callback,
                                                         max_indegree,
                                                         max_iters,
                                                         epsilon,
                                                         patience,
                                                         verbose);
}

}  // namespace learning::algorithms