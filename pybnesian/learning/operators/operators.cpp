#include <models/BayesianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <learning/scores/scores.hpp>
#include <learning/operators/operators.hpp>
#include <util/validate_whitelists.hpp>

using models::BayesianNetworkType, models::SemiparametricBNType;

namespace learning::operators {

std::shared_ptr<Operator> AddArc::opposite(const BayesianNetworkBase&) const {
    return std::make_shared<RemoveArc>(this->source(), this->target(), -this->delta());
}

std::shared_ptr<Operator> AddArc::opposite(const ConditionalBayesianNetworkBase& m) const {
    return opposite(static_cast<const BayesianNetworkBase&>(m));
}

void ArcOperatorSet::update_valid_ops(const BayesianNetworkBase& model) {
    int num_nodes = model.num_nodes();

    bool changed_size = delta.rows() != num_nodes || delta.cols() != num_nodes;
    if (changed_size) {
        delta = MatrixXd(num_nodes, num_nodes);
        valid_op = MatrixXb(num_nodes, num_nodes);
    }

    auto val_ptr = valid_op.data();
    std::fill(val_ptr, val_ptr + num_nodes * num_nodes, true);

    auto restrictions = util::validate_restrictions(model, m_blacklist, m_whitelist);

    auto valid_ops =
        (num_nodes * num_nodes) - 2 * restrictions.arc_whitelist.size() - restrictions.arc_blacklist.size() - num_nodes;

    for (const auto& whitelist_arc : restrictions.arc_whitelist) {
        int source_index = model.collapsed_from_index(whitelist_arc.first);
        int target_index = model.collapsed_from_index(whitelist_arc.second);

        valid_op(source_index, target_index) = false;
        valid_op(target_index, source_index) = false;
        delta(source_index, target_index) = std::numeric_limits<double>::lowest();
        delta(target_index, source_index) = std::numeric_limits<double>::lowest();
    }

    for (const auto& blacklist_arc : restrictions.arc_blacklist) {
        int source_index = model.collapsed_from_index(blacklist_arc.first);
        int target_index = model.collapsed_from_index(blacklist_arc.second);

        valid_op(source_index, target_index) = false;
        delta(source_index, target_index) = std::numeric_limits<double>::lowest();
    }

    for (int i = 0; i < num_nodes; ++i) {
        valid_op(i, i) = false;
        delta(i, i) = std::numeric_limits<double>::lowest();
    }

    sorted_idx.clear();
    sorted_idx.reserve(valid_ops);

    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (valid_op(i, j)) {
                sorted_idx.push_back(i + j * num_nodes);
            }
        }
    }
}

double cache_score_operation(const BayesianNetworkBase& model,
                             const Score& score,
                             const std::string& source,
                             const std::string& target,
                             std::vector<std::string>& parents_target,
                             double source_cached_score,
                             double target_cached_score) {
    if (model.has_arc(source, target)) {
        util::swap_remove_v(parents_target, source);
        auto d = score.local_score(model, target, parents_target) - target_cached_score;
        parents_target.push_back(source);
        return d;
    } else if (model.has_arc(target, source)) {
        auto new_parents_source = model.parents(source);
        util::swap_remove_v(new_parents_source, target);

        parents_target.push_back(source);
        double d = score.local_score(model, source, new_parents_source) +
                   score.local_score(model, target, parents_target) - source_cached_score - target_cached_score;
        parents_target.pop_back();
        return d;
    } else {
        parents_target.push_back(source);
        double d = score.local_score(model, target, parents_target) - target_cached_score;
        parents_target.pop_back();
        return d;
    }
}

void ArcOperatorSet::cache_scores(const BayesianNetworkBase& model, const Score& score) {
    if (!score.compatible_bn(model)) {
        throw std::invalid_argument("BayesianNetwork is not compatible with the score.");
    }

    initialize_local_cache(model);

    if (owns_local_cache()) {
        this->m_local_cache->cache_local_scores(model, score);
    }

    update_valid_ops(model);

    auto bn_type = model.type();
    for (const auto& target_node : model.nodes()) {
        std::vector<std::string> new_parents_target = model.parents(target_node);
        int target_collapsed = model.collapsed_index(target_node);
        for (const auto& source_node : model.nodes()) {
            int source_collapsed = model.collapsed_index(source_node);
            if (valid_op(source_collapsed, target_collapsed) &&
                bn_type->can_have_arc(model, source_node, target_node)) {
                delta(source_collapsed, target_collapsed) =
                    cache_score_operation(model,
                                          score,
                                          source_node,
                                          target_node,
                                          new_parents_target,
                                          m_local_cache->local_score(model, source_node),
                                          m_local_cache->local_score(model, target_node));
            }
        }
    }
}

double cache_score_interface(const ConditionalBayesianNetworkBase& model,
                             const Score& score,
                             const std::string& source,
                             const std::string& target,
                             std::vector<std::string>& parents_target,
                             double target_cached_score) {
    if (model.has_arc(source, target)) {
        util::swap_remove_v(parents_target, source);
        double d = score.local_score(model, target, parents_target) - target_cached_score;
        parents_target.push_back(source);
        return d;
    } else {
        parents_target.push_back(source);
        double d = score.local_score(model, target, parents_target) - target_cached_score;
        parents_target.pop_back();
        return d;
    }
}

void ArcOperatorSet::update_valid_ops(const ConditionalBayesianNetworkBase& model) {
    int num_nodes = model.num_nodes();
    int total_nodes = model.num_joint_nodes();

    bool changed_size = delta.rows() != total_nodes || delta.cols() != num_nodes;
    if (changed_size) {
        delta = MatrixXd(total_nodes, num_nodes);
        valid_op = MatrixXb(total_nodes, num_nodes);
    }

    auto val_ptr = valid_op.data();
    std::fill(val_ptr, val_ptr + total_nodes * num_nodes, true);

    auto restrictions = util::validate_restrictions(model, m_blacklist, m_whitelist);

    auto valid_ops = total_nodes * num_nodes - num_nodes;

    for (const auto& whitelist_arc : restrictions.arc_whitelist) {
        int source_joint_collapsed = model.joint_collapsed_from_index(whitelist_arc.first);
        int target_collapsed = model.collapsed_from_index(whitelist_arc.second);

        valid_op(source_joint_collapsed, target_collapsed) = false;
        delta(source_joint_collapsed, target_collapsed) = std::numeric_limits<double>::lowest();
        --valid_ops;
        if (!model.is_interface(model.name(whitelist_arc.first))) {
            int target_joint_collapsed = model.joint_collapsed_from_index(whitelist_arc.second);
            int source_collapsed = model.collapsed_from_index(whitelist_arc.first);
            valid_op(target_joint_collapsed, source_collapsed) = false;
            delta(target_joint_collapsed, source_collapsed) = std::numeric_limits<double>::lowest();
            --valid_ops;
        }
    }

    for (const auto& blacklist_arc : restrictions.arc_blacklist) {
        int source_joint_collapsed = model.joint_collapsed_from_index(blacklist_arc.first);
        int target_collapsed = model.collapsed_from_index(blacklist_arc.second);

        valid_op(source_joint_collapsed, target_collapsed) = false;
        delta(source_joint_collapsed, target_collapsed) = std::numeric_limits<double>::lowest();
        --valid_ops;
    }

    for (int i = 0; i < num_nodes; ++i) {
        auto joint_collapsed = model.joint_collapsed_from_index(model.index_from_collapsed(i));
        valid_op(joint_collapsed, i) = false;
        delta(joint_collapsed, i) = std::numeric_limits<double>::lowest();
    }

    sorted_idx.clear();
    sorted_idx.reserve(valid_ops);

    for (int i = 0; i < total_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (valid_op(i, j)) {
                sorted_idx.push_back(i + j * total_nodes);
            }
        }
    }
}

void ArcOperatorSet::cache_scores(const ConditionalBayesianNetworkBase& model, const Score& score) {
    if (!score.compatible_bn(model)) {
        throw std::invalid_argument("BayesianNetwork is not compatible with the score.");
    }

    initialize_local_cache(model);

    if (owns_local_cache()) {
        this->m_local_cache->cache_local_scores(model, score);
    }

    update_valid_ops(model);

    auto bn_type = model.type();
    for (const auto& target_node : model.nodes()) {
        auto target_collapsed = model.collapsed_index(target_node);
        auto new_parents_target = model.parents(target_node);

        for (const auto& source_node : model.joint_nodes()) {
            int source_joint_collapsed = model.joint_collapsed_index(source_node);
            if (valid_op(source_joint_collapsed, target_collapsed) &&
                bn_type->can_have_arc(model, source_node, target_node)) {
                if (model.is_interface(source_node)) {
                    delta(source_joint_collapsed, target_collapsed) =
                        cache_score_interface(model,
                                              score,
                                              source_node,
                                              target_node,
                                              new_parents_target,
                                              m_local_cache->local_score(model, target_node));
                } else {
                    delta(source_joint_collapsed, target_collapsed) =
                        cache_score_operation(model,
                                              score,
                                              source_node,
                                              target_node,
                                              new_parents_target,
                                              m_local_cache->local_score(model, source_node),
                                              m_local_cache->local_score(model, target_node));
                }
            }
        }
    }
}

std::shared_ptr<Operator> ArcOperatorSet::find_max(const BayesianNetworkBase& model) const {
    raise_uninitialized();

    if (max_indegree > 0)
        return find_max_indegree<true>(model);
    else
        return find_max_indegree<false>(model);
}

std::shared_ptr<Operator> ArcOperatorSet::find_max(const ConditionalBayesianNetworkBase& model) const {
    raise_uninitialized();

    if (max_indegree > 0)
        return find_max_indegree<true>(model);
    else
        return find_max_indegree<false>(model);
}

std::shared_ptr<Operator> ArcOperatorSet::find_max(const BayesianNetworkBase& model,
                                                   const OperatorTabuSet& tabu_set) const {
    raise_uninitialized();

    if (max_indegree > 0)
        return find_max_indegree<true>(model, tabu_set);
    else
        return find_max_indegree<false>(model, tabu_set);
}

std::shared_ptr<Operator> ArcOperatorSet::find_max(const ConditionalBayesianNetworkBase& model,
                                                   const OperatorTabuSet& tabu_set) const {
    raise_uninitialized();

    if (max_indegree > 0)
        return find_max_indegree<true>(model, tabu_set);
    else
        return find_max_indegree<false>(model, tabu_set);
}

void ArcOperatorSet::update_incoming_arcs_scores(const BayesianNetworkBase& model,
                                                 const Score& score,
                                                 const std::string& target_node) {
    auto target_collapsed = model.collapsed_index(target_node);
    auto parents = model.parents(target_node);

    auto bn_type = model.type();
    for (const auto& source_node : model.nodes()) {
        auto source_collapsed = model.collapsed_index(source_node);

        if (valid_op(source_collapsed, target_collapsed)) {
            if (model.has_arc(source_node, target_node)) {
                // Update remove arc: source_node -> target_node
                util::swap_remove_v(parents, source_node);
                double d = score.local_score(model, target_node, parents) -
                           this->m_local_cache->local_score(model, target_node);
                parents.push_back(source_node);
                delta(source_collapsed, target_collapsed) = d;

                // Update flip arc: source_node -> target_node
                if (valid_op(target_collapsed, source_collapsed) &&
                    bn_type->can_have_arc(model, target_node, source_node)) {
                    auto parents_source = model.parents(source_node);
                    parents_source.push_back(target_node);
                    delta(target_collapsed, source_collapsed) = d +
                                                                score.local_score(model, source_node, parents_source) -
                                                                this->m_local_cache->local_score(model, source_node);
                }
            } else if (model.has_arc(target_node, source_node) &&
                       bn_type->can_have_arc(model, source_node, target_node)) {
                // Update flip arc: target_node -> source_node
                auto parents_source = model.parents(source_node);
                util::swap_remove_v(parents_source, target_node);

                parents.push_back(source_node);
                double d = score.local_score(model, source_node, parents_source) +
                           score.local_score(model, target_node, parents) -
                           this->m_local_cache->local_score(model, source_node) -
                           this->m_local_cache->local_score(model, target_node);
                parents.pop_back();
                delta(source_collapsed, target_collapsed) = d;
            } else if (bn_type->can_have_arc(model, source_node, target_node)) {
                // Update add arc: source_node -> target_node
                parents.push_back(source_node);
                double d = score.local_score(model, target_node, parents) -
                           this->m_local_cache->local_score(model, target_node);
                parents.pop_back();
                delta(source_collapsed, target_collapsed) = d;
            }
        }
    }
}

void ArcOperatorSet::update_scores(const BayesianNetworkBase& model,
                                   const Score& score,
                                   const std::vector<std::string>& variables) {
    raise_uninitialized();

    if (owns_local_cache()) {
        for (const auto& n : variables) {
            m_local_cache->update_local_score(model, score, n);
        }
    }

    for (const auto& n : variables) {
        update_incoming_arcs_scores(model, score, n);
    }
}

void ArcOperatorSet::update_incoming_arcs_scores(const ConditionalBayesianNetworkBase& model,
                                                 const Score& score,
                                                 const std::string& target_node) {
    auto target_collapsed = model.collapsed_index(target_node);
    auto parents = model.parents(target_node);

    auto bn_type = model.type();
    for (const auto& source_node : model.joint_nodes()) {
        auto source_joint_collapsed = model.joint_collapsed_index(source_node);

        if (valid_op(source_joint_collapsed, target_collapsed)) {
            if (model.has_arc(source_node, target_node)) {
                // Update remove arc: source_node -> target_node
                util::swap_remove_v(parents, source_node);
                double d = score.local_score(model, target_node, parents) -
                           this->m_local_cache->local_score(model, target_node);
                parents.push_back(source_node);
                delta(source_joint_collapsed, target_collapsed) = d;

                if (!model.is_interface(source_node) && bn_type->can_have_arc(model, target_node, source_node)) {
                    // Update flip arc: source_node -> target_node
                    int target_joint_collapsed = model.joint_collapsed_index(target_node);
                    int source_collapsed = model.collapsed_index(source_node);

                    if (valid_op(target_joint_collapsed, source_collapsed)) {
                        auto parents_source = model.parents(source_node);
                        parents_source.push_back(target_node);

                        delta(target_joint_collapsed, source_collapsed) =
                            d + score.local_score(model, source_node, parents_source) -
                            this->m_local_cache->local_score(model, source_node);
                    }
                }
            } else if (!model.is_interface(source_node) && model.has_arc(target_node, source_node) &&
                       bn_type->can_have_arc(model, source_node, target_node)) {
                // Update flip arc: target_node -> source_node
                auto parents_source = model.parents(source_node);
                util::swap_remove_v(parents_source, target_node);

                parents.push_back(source_node);
                double d = score.local_score(model, source_node, parents_source) +
                           score.local_score(model, target_node, parents) -
                           this->m_local_cache->local_score(model, source_node) -
                           this->m_local_cache->local_score(model, target_node);
                parents.pop_back();
                delta(source_joint_collapsed, target_collapsed) = d;
            } else if (bn_type->can_have_arc(model, source_node, target_node)) {
                // Update add arc: source_node -> target_node
                parents.push_back(source_node);
                double d = score.local_score(model, target_node, parents) -
                           this->m_local_cache->local_score(model, target_node);
                parents.pop_back();
                delta(source_joint_collapsed, target_collapsed) = d;
            }
        }
    }
}

void ArcOperatorSet::update_scores(const ConditionalBayesianNetworkBase& model,
                                   const Score& score,
                                   const std::vector<std::string>& variables) {
    raise_uninitialized();

    if (owns_local_cache()) {
        for (const auto& n : variables) {
            m_local_cache->update_local_score(model, score, n);
        }
    }

    for (const auto& n : variables) {
        update_incoming_arcs_scores(model, score, n);
    }
}

void ChangeNodeTypeSet::cache_scores(const BayesianNetworkBase& model, const Score& score) {
    if (model.type_ref().is_homogeneous()) {
        throw std::invalid_argument("ChangeNodeTypeSet can only be used with non-homogeneous Bayesian networks.");
    }

    if (!score.compatible_bn(model)) {
        throw std::invalid_argument("BayesianNetwork is not compatible with the score.");
    }

    initialize_local_cache(model);

    if (owns_local_cache()) {
        this->m_local_cache->cache_local_scores(model, score);
    }

    delta.clear();
    update_whitelisted(model);

    auto bn_type = model.type();
    for (int i = 0; i < model.num_nodes(); ++i) {
        if (m_is_whitelisted(i)) continue;

        const auto& collapsed_name = model.collapsed_name(i);

        auto type = model.node_type(collapsed_name);
        if (*type == UnknownFactorType::get_ref()) {
            throw std::invalid_argument("Cannot calculate ChangeNodeType delta score for " + type->ToString() +
                                        ". Set appropiate node types for the model");
        }

        auto alt_node_types = bn_type->alternative_node_type(model, collapsed_name);

        if (alt_node_types.empty()) {
            delta.emplace_back();
        } else {
            delta.emplace_back(alt_node_types.size());

            double current_score = this->m_local_cache->local_score(model, collapsed_name);
            for (auto k = 0, k_end = static_cast<int>(alt_node_types.size()); k < k_end; ++k) {
                bool not_blacklisted =
                    m_type_blacklist.find(std::make_pair(collapsed_name, alt_node_types[k])) == m_type_blacklist.end();

                if (not_blacklisted && bn_type->compatible_node_type(model, collapsed_name, alt_node_types[k])) {
                    auto parents = model.parents(collapsed_name);
                    delta.back()(k) =
                        score.local_score(model, alt_node_types[k], collapsed_name, parents) - current_score;
                } else {
                    delta.back()(k) = std::numeric_limits<double>::lowest();
                }
            }
        }
    }
}

std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(const BayesianNetworkBase& model) const {
    raise_uninitialized();

    double max_score = std::numeric_limits<double>::lowest();
    int max_node = -1;
    int max_type = -1;
    for (auto i = 0, i_end = static_cast<int>(delta.size()); i < i_end; ++i) {
        if (!m_is_whitelisted(i) && delta[i].rows() > 0) {
            int local_max;
            delta[i].maxCoeff(&local_max);

            if (delta[i](local_max) > max_score) {
                max_score = delta[i](local_max);
                max_node = i;
                max_type = local_max;
            }
        }
    }

    if (max_score > std::numeric_limits<double>::lowest()) {
        const auto& node_name = model.collapsed_name(max_node);
        auto alt_node_types = model.type()->alternative_node_type(model, node_name);

        return std::make_shared<ChangeNodeType>(node_name, alt_node_types[max_type], delta[max_node](max_type));
    } else {
        return nullptr;
    }
}

std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(const BayesianNetworkBase& model,
                                                      const OperatorTabuSet& tabu_set) const {
    raise_uninitialized();

    double max_score = std::numeric_limits<double>::lowest();
    int max_node = -1;
    int max_type = -1;

    for (auto i = 0, i_end = static_cast<int>(delta.size()); i < i_end; ++i) {
        if (!m_is_whitelisted(i) && delta[i].rows() > 0) {
            const auto& collapsed_name = model.collapsed_name(i);
            auto alt_node_types = model.type()->alternative_node_type(model, collapsed_name);
            for (auto k = 0; k < delta[i].rows(); ++k) {
                if (delta[i](k) > max_score) {
                    auto op = std::make_shared<ChangeNodeType>(collapsed_name, alt_node_types[k], delta[i](k));
                    if (!tabu_set.contains(op)) {
                        max_score = delta[i](k);
                        max_node = i;
                        max_type = k;
                    }
                }
            }
        }
    }

    if (max_score > std::numeric_limits<double>::lowest()) {
        const auto& node_name = model.collapsed_name(max_node);
        auto alt_node_types = model.type()->alternative_node_type(model, node_name);

        return std::make_shared<ChangeNodeType>(node_name, alt_node_types[max_type], delta[max_node](max_type));
    } else {
        return nullptr;
    }
}

void ChangeNodeTypeSet::update_scores(const BayesianNetworkBase& model,
                                      const Score& score,
                                      const std::vector<std::string>& variables) {
    raise_uninitialized();

    if (owns_local_cache()) {
        for (const auto& n : variables) {
            m_local_cache->update_local_score(model, score, n);
        }
    }

    auto bn_type = model.type();
    for (const auto& n : variables) {
        auto collapsed_index = model.collapsed_index(n);

        if (m_is_whitelisted(collapsed_index)) continue;

        double current_score = this->m_local_cache->local_score(model, n);
        auto alt_node_types = model.type()->alternative_node_type(model, n);

        if (static_cast<size_t>(delta[collapsed_index].rows()) < alt_node_types.size()) {
            delta[collapsed_index] = VectorXd(alt_node_types.size());
        }

        if (static_cast<size_t>(delta[collapsed_index].rows()) > alt_node_types.size()) {
            std::fill(delta[collapsed_index].data() + alt_node_types.size(),
                      delta[collapsed_index].data() + delta[collapsed_index].rows(),
                      std::numeric_limits<double>::lowest());
        }

        for (auto k = 0, k_end = static_cast<int>(alt_node_types.size()); k < k_end; ++k) {
            bool not_blacklisted =
                m_type_blacklist.find(std::make_pair(n, alt_node_types[k])) == m_type_blacklist.end();

            if (bn_type->compatible_node_type(model, n, alt_node_types[k]) && not_blacklisted) {
                auto parents = model.parents(n);
                delta[collapsed_index](k) = score.local_score(model, alt_node_types[k], n, parents) - current_score;
            } else {
                delta[collapsed_index](k) = std::numeric_limits<double>::lowest();
            }
        }
    }
}

}  // namespace learning::operators
