#include <learning/operators/operators.hpp>

using learning::operators::AddArc;

namespace learning::operators {

    bool Operator::operator==(const Operator& a) const {
        if (m_type == a.m_type) {
            switch(m_type) {
                case OperatorType::ADD_ARC:
                case OperatorType::REMOVE_ARC:
                case OperatorType::FLIP_ARC: {
                    auto& this_dwn = dynamic_cast<const ArcOperator&>(*this);
                    auto& a_dwn = dynamic_cast<const ArcOperator&>(a);
                    return this_dwn.source() == a_dwn.source() && this_dwn.target() == a_dwn.target();
                }
                case OperatorType::CHANGE_NODE_TYPE: {
                    auto& this_dwn = dynamic_cast<const ChangeNodeType&>(*this);
                    auto& a_dwn = dynamic_cast<const ChangeNodeType&>(a);
                    return this_dwn.node() == a_dwn.node() && this_dwn.node_type() == a_dwn.node_type();
                }
                default:
                    throw std::runtime_error("Wrong operator type declared");
            }
        } else {
            return false;
        }
    }

    std::shared_ptr<Operator> AddArc::opposite() {
        return std::make_shared<RemoveArc>(this->source(), this->target(), -this->delta());
    }

    void ArcOperatorSet::update_listed_arcs(BayesianNetworkBase& model) {
        if (required_arclist_update) {
            int num_nodes = model.num_nodes();
            if (delta.rows() != num_nodes) {
                delta = MatrixXd(num_nodes, num_nodes);
                valid_op = MatrixXb(num_nodes, num_nodes);
            }

            for (const auto& arc : m_blacklist_names) {
                m_blacklist.insert({model.index(arc.first), model.index(arc.second)});
            }

            m_blacklist_names.clear();

            for (const auto& arc : m_whitelist_names) {
                m_whitelist.insert({model.index(arc.first), model.index(arc.second)});
            }

            m_whitelist_names.clear();

            auto val_ptr = valid_op.data();

            std::fill(val_ptr, val_ptr + num_nodes*num_nodes, true);

            auto valid_ops = (num_nodes * num_nodes) - 2*m_whitelist.size() - m_blacklist.size() - num_nodes;

            for(auto whitelist_arc : m_whitelist) {
                int source_index = whitelist_arc.first; 
                int target_index = whitelist_arc.second;

                valid_op(source_index, target_index) = false;
                valid_op(target_index, source_index) = false;
                delta(source_index, target_index) = std::numeric_limits<double>::lowest();
                delta(target_index, source_index) = std::numeric_limits<double>::lowest();
            }
            
            for(auto blacklist_arc : m_blacklist) {
                int source_index = blacklist_arc.first; 
                int target_index = blacklist_arc.second;

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

            required_arclist_update = false;
        }
    }

    double cache_score_operation(BayesianNetworkBase& model,
                                 Score& score,
                                 int source,
                                 int dest,
                                 std::vector<int>& parents_dest,
                                 double source_cached_score,
                                 double dest_cached_score) {
        if (model.has_arc(source, dest)) {            
            std::iter_swap(std::find(parents_dest.begin(), parents_dest.end(), source), parents_dest.end() - 1);
            return score.local_score(model, dest, parents_dest.begin(), parents_dest.end() - 1) - dest_cached_score;
        } else if (model.has_arc(dest, source)) {
            auto new_parents_source = model.parent_indices(source);
            util::swap_remove_v(new_parents_source, dest);
            
            parents_dest.push_back(source);
            double d = score.local_score(model, source, new_parents_source.begin(), new_parents_source.end()) + 
                        score.local_score(model, dest, parents_dest.begin(), parents_dest.end()) 
                        - source_cached_score - dest_cached_score;
            parents_dest.pop_back();
            return d;
        } else {
            parents_dest.push_back(source);
            double d = score.local_score(model, dest, parents_dest.begin(), parents_dest.end()) 
                        - dest_cached_score;
            parents_dest.pop_back();
            return d;
        }
    }

    void ArcOperatorSet::cache_scores(BayesianNetworkBase& model, Score& score) {
        initialize_local_cache(model);

        if (owns_local_cache()) {
            this->m_local_cache->cache_local_scores(model, score);
        }

        if (!util::compatible_score(model, score.type())) {
            throw std::invalid_argument("Invalid score " + score.ToString() + " for model type " + model.type().ToString() + ".");
        }

        update_listed_arcs(model);

        for (auto dest = 0; dest < model.num_nodes(); ++dest) {
            std::vector<int> new_parents_dest = model.parent_indices(dest);
            
            for (auto source = 0; source < model.num_nodes(); ++source) {
                if(valid_op(source, dest)) {
                    delta(source, dest) = cache_score_operation(model, score, source, dest, new_parents_dest,
                                                                m_local_cache->local_score(source),
                                                                m_local_cache->local_score(dest));
                }
            }
        }
    }

    double cache_score_interface(ConditionalBayesianNetworkBase& model,
                                 Score& score,
                                 int source,
                                 int dest,
                                 std::vector<int>& parents_dest,
                                 double dest_cached_score) {
        if (model.has_arc(source, dest)) {            
            std::iter_swap(std::find(parents_dest.begin(), parents_dest.end(), source), parents_dest.end() - 1);
            return score.local_score(model, dest, parents_dest.begin(), parents_dest.end() - 1) - dest_cached_score;
        } else {
            parents_dest.push_back(source);
            double d = score.local_score(model, dest, parents_dest.begin(), parents_dest.end()) 
                        - dest_cached_score;
            parents_dest.pop_back();
            return d;
        }
    }

    void ArcOperatorSet::update_listed_arcs(ConditionalBayesianNetworkBase& model) {
        if (required_arclist_update) {
            int num_nodes = model.num_nodes();
            int total_nodes = model.num_total_nodes();
            if (delta.rows() != total_nodes || delta.cols() != num_nodes) {
                delta = MatrixXd(total_nodes, num_nodes);
                valid_op = MatrixXb(total_nodes, num_nodes);
            }

            for (const auto& arc : m_blacklist_names) {
                m_blacklist.insert({model.index(arc.first), model.index(arc.second)});
            }

            m_blacklist_names.clear();

            for (const auto& arc : m_whitelist_names) {
                m_whitelist.insert({model.index(arc.first), model.index(arc.second)});
            }

            m_whitelist_names.clear();

            auto val_ptr = valid_op.data();

            std::fill(val_ptr, val_ptr + total_nodes*num_nodes, true);

            auto valid_ops = total_nodes * num_nodes - num_nodes;

            for(auto whitelist_arc : m_whitelist) {
                int source_index = whitelist_arc.first; 
                int target_index = whitelist_arc.second;

                int collapsed_target = model.collapsed_from_index(target_index);

                valid_op(source_index, collapsed_target) = false;
                delta(source_index, collapsed_target) = std::numeric_limits<double>::lowest();
                --valid_ops;
                if (!model.is_interface(source_index)) {
                    int collapsed_source = model.collapsed_from_index(source_index);
                    valid_op(target_index, collapsed_source) = false;
                    delta(target_index, collapsed_source) = std::numeric_limits<double>::lowest();
                    --valid_ops;
                }
            }
            
            for(auto blacklist_arc : m_blacklist) {
                int source_index = blacklist_arc.first; 
                int target_index = model.collapsed_from_index(blacklist_arc.second);

                valid_op(source_index, target_index) = false;
                delta(source_index, target_index) = std::numeric_limits<double>::lowest();
                --valid_ops;
            }

            for (int i = 0; i < num_nodes; ++i) {
                auto index = model.index_from_collapsed(i);
                valid_op(index, i) = false;
                delta(index, i) = std::numeric_limits<double>::lowest();
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

            required_arclist_update = false;
        }
    }

    void ArcOperatorSet::cache_scores(ConditionalBayesianNetworkBase& model, Score& score) {
        initialize_local_cache(model);

        if (owns_local_cache()) {
            this->m_local_cache->cache_local_scores(model, score);
        }

        if (!util::compatible_score(model, score.type())) {
            throw std::invalid_argument("Invalid score " + score.ToString() + " for model type " + model.type().ToString() + ".");
        }

        update_listed_arcs(model);

        for (auto dest_collapsed = 0; dest_collapsed < model.num_nodes(); ++dest_collapsed) {
            auto dest = model.index_from_collapsed(dest_collapsed);
            std::vector<int> new_parents_dest = model.parent_indices(dest);
            
            for (auto source = 0; source < model.num_total_nodes(); ++source) {
                if(valid_op(source, dest_collapsed)) {
                    if (model.is_interface(source)) {
                        delta(source, dest) = cache_score_interface(model, score, source, dest, new_parents_dest,
                                                                    m_local_cache->local_score(dest_collapsed));
                    } else {
                        auto source_collapsed = model.collapsed_from_index(source);
                        delta(source, dest) = cache_score_operation(model, score, source, dest, new_parents_dest,
                                                                    m_local_cache->local_score(source_collapsed),
                                                                    m_local_cache->local_score(dest_collapsed));
                    }
                }
            }
        }
    }

    std::shared_ptr<Operator> ArcOperatorSet::find_max(BayesianNetworkBase& model) {
        raise_uninitialized();

        if (max_indegree > 0)
            return find_max_indegree<true>(model);
        else
            return find_max_indegree<false>(model);
    }

    std::shared_ptr<Operator> ArcOperatorSet::find_max(ConditionalBayesianNetworkBase& model) {
        raise_uninitialized();

        if (max_indegree > 0)
            return find_max_indegree<true>(model);
        else
            return find_max_indegree<false>(model);
    }

    std::shared_ptr<Operator> ArcOperatorSet::find_max(BayesianNetworkBase& model, OperatorTabuSet& tabu_set) {
        raise_uninitialized();

        if (max_indegree > 0)
            return find_max_indegree<true>(model, tabu_set);
        else
            return find_max_indegree<false>(model, tabu_set);
    }

    std::shared_ptr<Operator> ArcOperatorSet::find_max(ConditionalBayesianNetworkBase& model, OperatorTabuSet& tabu_set) {
        raise_uninitialized();

        if (max_indegree > 0)
            return find_max_indegree<true>(model, tabu_set);
        else
            return find_max_indegree<false>(model, tabu_set);
    }

    void ArcOperatorSet::update_scores(BayesianNetworkBase& model, Score& score, Operator& op) {
        raise_uninitialized();

        if (owns_local_cache()) {
            m_local_cache->update_local_score(model, score, op);
        }

        switch(op.type()) {
            case OperatorType::ADD_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                update_node_arcs_scores(model, score, dwn_op.target());
            }
                break;
            case OperatorType::REMOVE_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);   
                // Update the cost of (AddArc: target -> source). New Add delta = Old Flip delta - Old Remove delta
                auto source_idx = model.index(dwn_op.source());
                auto target_idx = model.index(dwn_op.target());
                delta(target_idx, source_idx) = delta(target_idx, source_idx) - delta(source_idx, target_idx);

                update_node_arcs_scores(model, score, dwn_op.target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                update_node_arcs_scores(model, score, dwn_op.source());
                update_node_arcs_scores(model, score, dwn_op.target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto& dwn_op = dynamic_cast<ChangeNodeType&>(op);
                update_node_arcs_scores(model, score, dwn_op.node());
            }
                break;
        }
    }   

    void ArcOperatorSet::update_scores(ConditionalBayesianNetworkBase& model, Score& score, Operator& op) {
        raise_uninitialized();

        if (owns_local_cache()) {
            m_local_cache->update_local_score(model, score, op);
        }

        switch(op.type()) {
            case OperatorType::ADD_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                update_node_arcs_scores(model, score, dwn_op.target());
            }
                break;
            case OperatorType::REMOVE_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);   
                // Update the cost of (AddArc: target -> source). New Add delta = Old Flip delta - Old Remove delta
                if (!model.is_interface(dwn_op.source())) {
                    auto source_idx = model.index(dwn_op.source());
                    auto source_collapsed = model.collapsed_index(dwn_op.source());
                    auto target_idx = model.index(dwn_op.target());
                    auto target_collapsed = model.collapsed_index(dwn_op.target());
                    delta(target_idx, source_collapsed) = delta(target_idx, source_collapsed) - delta(source_idx, target_collapsed);
                }

                update_node_arcs_scores(model, score, dwn_op.target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                update_node_arcs_scores(model, score, dwn_op.source());
                update_node_arcs_scores(model, score, dwn_op.target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto& dwn_op = dynamic_cast<ChangeNodeType&>(op);
                update_node_arcs_scores(model, score, dwn_op.node());
            }
                break;
        }
    }

    void ArcOperatorSet::update_node_arcs_scores(BayesianNetworkBase& model, Score& score, const std::string& dest_node) {
        auto dest_idx = model.index(dest_node);
        auto parents = model.parent_indices(dest_idx);
        
        for (int i = 0; i < model.num_nodes(); ++i) {
            if (valid_op(i, dest_idx)) {
                if (model.has_arc(i, dest_idx)) {
                    // Update remove arc: i -> dest_idx
                    std::iter_swap(std::find(parents.begin(), parents.end(), i), parents.end() - 1);
                    double d = score.local_score(model, dest_idx, parents.begin(), parents.end() - 1) - 
                               this->m_local_cache->local_score(dest_idx);
                    delta(i, dest_idx) = d;

                    // Update flip arc: i -> dest_idx
                    if (valid_op(dest_idx, i)) {                       
                        auto new_parents_i = model.parent_indices(i);
                        new_parents_i.push_back(dest_idx);

                        delta(dest_idx, i) = d + score.local_score(model, i, new_parents_i.begin(), new_parents_i.end())
                                                - this->m_local_cache->local_score(i);
                    }
                } else if (model.has_arc(dest_idx, i)) {
                    // Update flip arc: dest_idx -> i
                    auto new_parents_i = model.parent_indices(i);
                    util::swap_remove_v(new_parents_i, dest_idx);

                    parents.push_back(i);
                    double d = score.local_score(model, i, new_parents_i.begin(), new_parents_i.end()) + 
                               score.local_score(model, dest_idx, parents.begin(), parents.end()) 
                                - this->m_local_cache->local_score(i) - this->m_local_cache->local_score(dest_idx);
                    parents.pop_back();
                    delta(i, dest_idx) = d;
                } else {
                    // Update add arc: i -> dest_idx
                    parents.push_back(i);
                    double d = score.local_score(model, dest_idx, parents.begin(), parents.end()) - 
                                this->m_local_cache->local_score(dest_idx);
                    parents.pop_back();
                    delta(i, dest_idx) = d;
                }
            }
        }
    }

    void ArcOperatorSet::update_node_arcs_scores(ConditionalBayesianNetworkBase& model, Score& score, const std::string& dest_node) {
        auto dest_idx = model.index(dest_node);
        auto dest_collapsed = model.collapsed_from_index(dest_idx);
        auto parents = model.parent_indices(dest_idx);
        
        for (int i = 0; i < model.num_total_nodes(); ++i) {
            if (valid_op(i, dest_collapsed)) {
                if (model.has_arc(i, dest_idx)) {
                    // Update remove arc: i -> dest_idx
                    std::iter_swap(std::find(parents.begin(), parents.end(), i), parents.end() - 1);
                    double d = score.local_score(model, dest_idx, parents.begin(), parents.end() - 1) - 
                               this->m_local_cache->local_score(dest_collapsed);
                    delta(i, dest_collapsed) = d;

                    // Update flip arc: i -> dest_idx
                    if (!model.is_interface(i)) {
                        auto i_collapsed = model.collapsed_from_index(i);
                        if (valid_op(dest_idx, i_collapsed)) {                       
                            auto new_parents_i = model.parent_indices(i);
                            new_parents_i.push_back(dest_idx);

                            delta(dest_idx, i_collapsed) = d + score.local_score(model, i, new_parents_i.begin(), new_parents_i.end())
                                                    - this->m_local_cache->local_score(i_collapsed);
                        }
                    }
                } else if (model.has_arc(dest_idx, i)) {
                    // Update flip arc: dest_idx -> i
                    auto i_collapsed = model.collapsed_from_index(i);
                    auto new_parents_i = model.parent_indices(i);
                    util::swap_remove_v(new_parents_i, dest_idx);

                    parents.push_back(i);
                    double d = score.local_score(model, i, new_parents_i.begin(), new_parents_i.end()) + 
                               score.local_score(model, dest_idx, parents.begin(), parents.end()) 
                                - this->m_local_cache->local_score(i_collapsed) - this->m_local_cache->local_score(dest_collapsed);
                    parents.pop_back();
                    delta(i, dest_collapsed) = d;
                } else {
                    // Update add arc: i -> dest_idx
                    parents.push_back(i);
                    double d = score.local_score(model, dest_idx, parents.begin(), parents.end()) - 
                                this->m_local_cache->local_score(dest_collapsed);
                    parents.pop_back();
                    delta(i, dest_collapsed) = d;
                }
            }
        }
    }

    void ChangeNodeTypeSet::cache_scores(BayesianNetworkBase& model, Score& score) {
        initialize_local_cache(model);

        if (owns_local_cache()) {
            this->m_local_cache->cache_local_scores(model, score);
        }

        if (model.type() != BayesianNetworkType::SPBN) {
            throw std::invalid_argument("ChangeNodeTypeSet can only be used with SemiparametricBN");
        }

        if (!util::compatible_score(model, score.type())) {
            throw std::invalid_argument("Invalid score " + score.ToString() + " for model type " + model.type().ToString() + ".");
        }
        
        update_whitelisted(model);

        for(int i = 0, num_nodes = model.num_nodes(); i < num_nodes; ++i) {
            if(valid_op(i)) {
                update_local_delta(model, score, i);
            }
        }
    }

    void ChangeNodeTypeSet::cache_scores(ConditionalBayesianNetworkBase& model, Score& score) {}

    std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(BayesianNetworkBase& model) {
        raise_uninitialized();

        auto delta_ptr = delta.data();
        auto max_element = std::max_element(delta_ptr, delta_ptr + model.num_nodes());
        int idx_max = std::distance(delta_ptr, max_element);
        auto& spbn = dynamic_cast<SemiparametricBNBase&>(model);
        auto node_type = spbn.node_type(idx_max);

        if(valid_op(idx_max))
            return std::make_shared<ChangeNodeType>(model.name(idx_max), node_type.opposite(), *max_element);
        else
            return nullptr;
    }

    std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(ConditionalBayesianNetworkBase& model) {}

    std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(BayesianNetworkBase& model, OperatorTabuSet& tabu_set) {
        raise_uninitialized();
        
        auto delta_ptr = delta.data();
        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });
        auto& spbn = dynamic_cast<SemiparametricBNBase&>(model);
        for(auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
            int idx_max = *it;
            auto node_type = spbn.node_type(idx_max);
            std::shared_ptr<Operator> op = std::make_shared<ChangeNodeType>(model.name(idx_max), node_type.opposite(), delta(idx_max));
            if (tabu_set.contains(op))
                return op;

        }

        return nullptr;
    }

    std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(ConditionalBayesianNetworkBase& model, OperatorTabuSet& tabu_set) {}

    void ChangeNodeTypeSet::update_scores(BayesianNetworkBase& model, Score& score, Operator& op) {
        raise_uninitialized();

        if (owns_local_cache()) {
            m_local_cache->update_local_score(model, score, op);
        }

        switch(op.type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                update_local_delta(model, score, dwn_op.target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
                update_local_delta(model, score, dwn_op.source());
                update_local_delta(model, score, dwn_op.target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto& dwn_op = dynamic_cast<ChangeNodeType&>(op);
                int index = model.index(dwn_op.node());
                delta(index) = -dwn_op.delta();
            }
                break;
        }
    }

    void ChangeNodeTypeSet::update_scores(ConditionalBayesianNetworkBase& model, Score& score, Operator& op) {}

    void OperatorPool::cache_scores(BayesianNetworkBase& model, Score& score) {
        initialize_local_cache(model);
        m_local_cache->cache_local_scores(model, score);

        for (auto& op_set : m_op_sets) {
            op_set->cache_scores(model, score);
        }
    }

    void OperatorPool::cache_scores(ConditionalBayesianNetworkBase& model, Score& score) {}

    std::shared_ptr<Operator> OperatorPool::find_max(BayesianNetworkBase& model) {
        raise_uninitialized();

        double max_delta = std::numeric_limits<double>::lowest();
        std::shared_ptr<Operator> max_op = nullptr;

        for (auto& op_set : m_op_sets) {
            auto new_op = op_set->find_max(model);
            if (new_op && new_op->delta() > max_delta) {
                max_op = std::move(new_op);
                max_delta = max_op->delta();
            }
        }

        return max_op;
    }

    std::shared_ptr<Operator> OperatorPool::find_max(ConditionalBayesianNetworkBase& model) {}

    std::shared_ptr<Operator> OperatorPool::find_max(BayesianNetworkBase& model, OperatorTabuSet& tabu_set) {
        raise_uninitialized();

        if (tabu_set.empty())
            return find_max(model);
        
        double max_delta = std::numeric_limits<double>::lowest();
        std::shared_ptr<Operator> max_op = nullptr;

        for (auto& op_set : m_op_sets) {
            auto new_op = op_set->find_max(model, tabu_set);
            if (new_op && new_op->delta() > max_delta) {
                max_op = std::move(new_op);
                max_delta = max_op->delta();
            }
        }

        return max_op;
    }

    std::shared_ptr<Operator> OperatorPool::find_max(ConditionalBayesianNetworkBase& model, OperatorTabuSet& tabu_set) {}

    void OperatorPool::update_scores(BayesianNetworkBase& model, Score& score, Operator& op) {
        raise_uninitialized();

        m_local_cache->update_local_score(model, score, op);
        for (auto& op_set : m_op_sets) {
            op_set->update_scores(model, score, op);
        }
    }

    void OperatorPool::update_scores(ConditionalBayesianNetworkBase& model, Score& score, Operator& op) {}
}