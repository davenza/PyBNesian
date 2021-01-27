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

    std::shared_ptr<Operator> AddArc::opposite() const {
        return std::make_shared<RemoveArc>(this->source(), this->target(), -this->delta());
    }

    void ArcOperatorSet::update_valid_ops(const BayesianNetworkBase& model) {
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
                int source_index = model.collapsed_from_index(whitelist_arc.first);
                int target_index = model.collapsed_from_index(whitelist_arc.second);

                valid_op(source_index, target_index) = false;
                valid_op(target_index, source_index) = false;
                delta(source_index, target_index) = std::numeric_limits<double>::lowest();
                delta(target_index, source_index) = std::numeric_limits<double>::lowest();
            }
            
            for(auto blacklist_arc : m_blacklist) {
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

            required_arclist_update = false;
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
                       score.local_score(model, target, parents_target) 
                        - source_cached_score - target_cached_score;
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
        if (!util::compatible_score(model, score.type())) {
            throw std::invalid_argument("Invalid score " + score.ToString() + " for model type " + 
                                        models::BayesianNetworkType_ToString(model.type()) + ".");
        }

        initialize_local_cache(model);

        if (owns_local_cache()) {
            this->m_local_cache->cache_local_scores(model, score);
        }

        update_valid_ops(model);

        for (const auto& target_node : model.nodes()) {
            std::vector<std::string> new_parents_target = model.parents(target_node);
            int target_collapsed = model.collapsed_index(target_node);
            for (const auto& source_node : model.nodes()) {
                int source_collapsed = model.collapsed_index(source_node);
                if(valid_op(source_collapsed, target_collapsed)) {
                    delta(source_collapsed, target_collapsed) = cache_score_operation(model,
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

                int source_joint_collapsed = model.joint_collapsed_from_index(source_index);
                int target_collapsed = model.collapsed_from_index(target_index);

                valid_op(source_joint_collapsed, target_collapsed) = false;
                delta(source_joint_collapsed, target_collapsed) = std::numeric_limits<double>::lowest();
                --valid_ops;
                if (!model.is_interface(source_index)) {
                    int target_joint_collapsed = model.joint_collapsed_from_index(target_index);
                    int source_collapsed = model.collapsed_from_index(source_index);
                    valid_op(target_joint_collapsed, source_collapsed) = false;
                    delta(target_joint_collapsed, source_collapsed) = std::numeric_limits<double>::lowest();
                    --valid_ops;
                }
            }
            
            for(auto blacklist_arc : m_blacklist) {
                int source_index = blacklist_arc.first; 
                int target_index = blacklist_arc.second;

                int source_joint_collapsed = model.joint_collapsed_from_index(source_index);
                int target_collapsed = model.collapsed_from_index(target_index);

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

            required_arclist_update = false;
        }
    }

    void ArcOperatorSet::cache_scores(const ConditionalBayesianNetworkBase& model, const Score& score) {
        if (!util::compatible_score(model, score.type())) {
            throw std::invalid_argument("Invalid score " + score.ToString() + " for model type " +
                                        models::BayesianNetworkType_ToString(model.type()) + ".");
        }

        initialize_local_cache(model);

        if (owns_local_cache()) {
            this->m_local_cache->cache_local_scores(model, score);
        }

        update_valid_ops(model);

        for (const auto& target_node : model.nodes()) {
            auto target_collapsed = model.collapsed_index(target_node);
            auto new_parents_target = model.parents(target_node);

            for (const auto& source_node : model.all_nodes()) {
                int source_joint_collapsed = model.joint_collapsed_index(source_node);
                if(valid_op(source_joint_collapsed, target_collapsed)) {
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
                    if (valid_op(target_collapsed, source_collapsed)) {                       
                        auto parents_source = model.parents(source_node);
                        parents_source.push_back(target_node);

                        delta(target_collapsed, source_collapsed) = d + score.local_score(model, source_node, parents_source)
                                                                    - this->m_local_cache->local_score(model, source_node);
                    }
                } else if (model.has_arc(target_node, source_node)) {
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
                } else {
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
                                       const Operator& op) {
        raise_uninitialized();

        if (owns_local_cache()) {
            m_local_cache->update_local_score(model, score, op);
        }

        switch(op.type()) {
            case OperatorType::ADD_ARC: {
                auto& dwn_op = dynamic_cast<const ArcOperator&>(op);

                // (REMOVE_ARC, source -> target) = - (ADD_ARC, source -> target)
                auto source_collapsed = model.collapsed_index(dwn_op.source());
                auto target_collapsed = model.collapsed_index(dwn_op.target());
                delta(source_collapsed, target_collapsed) = -dwn_op.delta();

                if (valid_op(target_collapsed, source_collapsed)) {
                    // (FLIP_ARC, source -> target) = (REMOVE_ARC, source -> target) + (ADD_ARC, target -> source)
                    delta(target_collapsed, source_collapsed) = delta(source_collapsed, target_collapsed) + 
                                                                delta(target_collapsed, source_collapsed);

                }

                // Disable operation temporarily to avoid updating the previous deltas.
                valid_op(source_collapsed, target_collapsed) = false;
                update_incoming_arcs_scores(model, score, dwn_op.target());
                valid_op(source_collapsed, target_collapsed) = true;
            }
                break;
            case OperatorType::REMOVE_ARC: {
                auto& dwn_op = dynamic_cast<const ArcOperator&>(op);   

                auto source_collapsed = model.collapsed_index(dwn_op.source());
                auto target_collapsed = model.collapsed_index(dwn_op.target());
                // (ADD_ARC, source -> target) = - (REMOVE_ARC, source -> target)
                delta(source_collapsed, target_collapsed) = -dwn_op.delta();

                if (valid_op(target_collapsed, source_collapsed)) {
                    // (ADD_ARC, target -> source) = (FLIP_ARC, source -> target) - (REMOVE_ARC, source -> target)
                    //                             = (FLIP_ARC, source -> target) + (ADD_ARC, source -> target)

                    delta(target_collapsed, source_collapsed) = delta(target_collapsed, source_collapsed) + 
                                                                delta(source_collapsed, target_collapsed);

                }

                // Disable operation temporarily to avoid updating the previous deltas.
                valid_op(source_collapsed, target_collapsed) = false;
                update_incoming_arcs_scores(model, score, dwn_op.target());
                valid_op(source_collapsed, target_collapsed) = true;
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto& dwn_op = dynamic_cast<const ArcOperator&>(op);

                auto source_collapsed = model.collapsed_index(dwn_op.source());
                auto target_collapsed = model.collapsed_index(dwn_op.target());

                // Cache (REMOVE_ARC, source -> target)
                auto remove_arc_cache = delta(source_collapsed, target_collapsed);

                // (FLIP_ARC, target -> source) = - (FLIP_ARC, source -> target)
                delta(source_collapsed, target_collapsed) = -dwn_op.delta();

                // (REMOVE_ARC, target -> source) = (FLIP_ARC, target -> source) - (ADD_ARC, source -> target)
                //                                = (FLIP_ARC, target -> source) + (REMOVE_ARC, source -> target)
                delta(target_collapsed, source_collapsed) = delta(source_collapsed, target_collapsed) +
                                                            remove_arc_cache;

                // Disable operations temporarily to avoid updating the previous deltas.
                valid_op(source_collapsed, target_collapsed) = false;
                valid_op(target_collapsed, source_collapsed) = false;
                update_incoming_arcs_scores(model, score, dwn_op.source());
                update_incoming_arcs_scores(model, score, dwn_op.target());
                valid_op(source_collapsed, target_collapsed) = true;
                valid_op(target_collapsed, source_collapsed) = true;
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto& dwn_op = dynamic_cast<const ChangeNodeType&>(op);
                update_incoming_arcs_scores(model, score, dwn_op.node());
            }
                break;
        }
    }

    void ArcOperatorSet::update_incoming_arcs_scores(const ConditionalBayesianNetworkBase& model,
                                                     const Score& score,
                                                     const std::string& target_node) {
        auto target_collapsed = model.collapsed_index(target_node);
        auto parents = model.parents(target_node);

        for (const auto& source_node : model.all_nodes()) {
            auto source_joint_collapsed = model.joint_collapsed_index(source_node);

            if (valid_op(source_joint_collapsed, target_collapsed)) {
                if (model.has_arc(source_node, target_node)) {
                    // Update remove arc: source_node -> target_node
                    util::swap_remove_v(parents, source_node);
                    double d = score.local_score(model, target_node, parents) - 
                               this->m_local_cache->local_score(model, target_node);
                    parents.push_back(source_node);
                    delta(source_joint_collapsed, target_collapsed) = d;

                    if (!model.is_interface(source_node)) {
                        // Update flip arc: source_node -> target_node
                        int target_joint_collapsed = model.joint_collapsed_index(target_node);
                        int source_collapsed = model.collapsed_index(source_node);

                        if (valid_op(target_joint_collapsed, source_collapsed)) {                       
                            auto parents_source = model.parents(source_node);
                            parents_source.push_back(target_node);

                            delta(target_joint_collapsed, source_collapsed) = d + score.local_score(model, source_node, parents_source)
                                                                            - this->m_local_cache->local_score(model, source_node);
                        }
                    }
                } else if (!model.is_interface(source_node) && model.has_arc(target_node, source_node)) {
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
                } else {
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
                                       const Operator& op) {
        raise_uninitialized();

        if (owns_local_cache()) {
            m_local_cache->update_local_score(model, score, op);
        }

        switch(op.type()) {
            case OperatorType::ADD_ARC: {
                auto& dwn_op = dynamic_cast<const ArcOperator&>(op);

                // (REMOVE_ARC, source -> target) = - (ADD_ARC, source -> target)
                auto source_joint_collapsed = model.joint_collapsed_index(dwn_op.source());
                auto target_collapsed = model.collapsed_index(dwn_op.target());
                delta(source_joint_collapsed, target_collapsed) = -dwn_op.delta();

                if (!model.is_interface(dwn_op.source())) {
                    auto source_collapsed = model.collapsed_index(dwn_op.source());
                    auto target_joint_collapsed = model.joint_collapsed_index(dwn_op.target());

                    if (valid_op(target_joint_collapsed, source_collapsed)) {
                        // (FLIP_ARC, source -> target) = (REMOVE_ARC, source -> target) + (ADD_ARC, target -> source)
                        delta(target_joint_collapsed, source_collapsed) = delta(source_joint_collapsed, target_collapsed) +
                                                                          delta(target_joint_collapsed, source_collapsed);
                    }
                }

                // Disable operation temporarily to avoid updating the previous deltas.
                valid_op(source_joint_collapsed, target_collapsed) = false;
                update_incoming_arcs_scores(model, score, dwn_op.target());
                valid_op(source_joint_collapsed, target_collapsed) = true;
            }
                break;
            case OperatorType::REMOVE_ARC: {
                auto& dwn_op = dynamic_cast<const ArcOperator&>(op);   

                auto source_joint_collapsed = model.joint_collapsed_index(dwn_op.source());
                auto target_collapsed = model.collapsed_index(dwn_op.target());

                // (ADD_ARC, source -> target) = - (REMOVE_ARC, source -> target)
                delta(source_joint_collapsed, target_collapsed) = -dwn_op.delta();

                if (!model.is_interface(dwn_op.source())) {
                    auto source_collapsed = model.collapsed_index(dwn_op.source());
                    auto target_joint_collapsed = model.joint_collapsed_index(dwn_op.target());
                
                    if (valid_op(target_joint_collapsed, source_collapsed)) {
                        // (ADD_ARC, target -> source) = (FLIP_ARC, source -> target) - (REMOVE_ARC, source -> target)
                        //                             = (FLIP_ARC, source -> target) + (ADD_ARC, source -> target)

                        delta(target_joint_collapsed, source_collapsed) = delta(target_joint_collapsed, source_collapsed) + 
                                                                          delta(source_joint_collapsed, target_collapsed);

                    }
                }

                // Disable operation temporarily to avoid updating the previous deltas.
                valid_op(source_joint_collapsed, target_collapsed) = false;
                update_incoming_arcs_scores(model, score, dwn_op.target());
                valid_op(source_joint_collapsed, target_collapsed) = true;
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto& dwn_op = dynamic_cast<const ArcOperator&>(op);

                auto source_collapsed = model.collapsed_index(dwn_op.source());
                auto source_joint_collapsed = model.joint_collapsed_index(dwn_op.source());
                auto target_collapsed = model.collapsed_index(dwn_op.target());
                auto target_joint_collapsed = model.joint_collapsed_index(dwn_op.target());
                
                // Cache (REMOVE_ARC, source -> target)
                auto remove_arc_cache = delta(source_joint_collapsed, target_collapsed);
                
                // (FLIP_ARC, target -> source) = - (FLIP_ARC, source -> target)
                delta(source_joint_collapsed, target_collapsed) = -dwn_op.delta();

                // (REMOVE_ARC, target -> source) = (FLIP_ARC, target -> source) - (ADD_ARC, source -> target)
                //                                = (FLIP_ARC, target -> source) + (REMOVE_ARC, source -> target)
                delta(target_joint_collapsed, source_collapsed) = delta(source_joint_collapsed, target_collapsed) +
                                                                  remove_arc_cache;
                
                // Disable operations temporarily to avoid updating the previous deltas.
                valid_op(source_joint_collapsed, target_collapsed) = false;
                valid_op(target_joint_collapsed, source_collapsed) = false;
                update_incoming_arcs_scores(model, score, dwn_op.source());
                update_incoming_arcs_scores(model, score, dwn_op.target());
                valid_op(source_joint_collapsed, target_collapsed) = true;
                valid_op(target_joint_collapsed, source_collapsed) = true;
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto& dwn_op = dynamic_cast<const ChangeNodeType&>(op);
                update_incoming_arcs_scores(model, score, dwn_op.node());
            }
                break;
        }
    }

    void ChangeNodeTypeSet::cache_scores(const BayesianNetworkBase& model, const Score& score) {
        if (model.type() != BayesianNetworkType::Semiparametric) {
            throw std::invalid_argument("ChangeNodeTypeSet can only be used with SemiparametricBN");
        }
        
        if (!util::compatible_score(model, score.type())) {
            throw std::invalid_argument("Invalid score " + score.ToString() + " for model type " + 
                                        models::BayesianNetworkType_ToString(model.type()) + ".");
        }

        initialize_local_cache(model);

        if (owns_local_cache()) {
            this->m_local_cache->cache_local_scores(model, score);
        }

        update_whitelisted(model);

        for(int i = 0, num_nodes = model.num_nodes(); i < num_nodes; ++i) {
            if(valid_op(i)) {
                update_local_delta(model, score, model.collapsed_name(i));
            }
        }
    }

    std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(const BayesianNetworkBase& model) const {
        raise_uninitialized();
        
        auto delta_ptr = delta.data();
        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        auto& spbn = dynamic_cast<const SemiparametricBNBase&>(model);
        for(auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
            int idx_max = *it;
            const auto& node = model.collapsed_name(idx_max);
            auto node_type = spbn.node_type(node);
            return std::make_shared<ChangeNodeType>(node, node_type.opposite(), delta(idx_max));
        }

        return nullptr;
    }

    std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(const BayesianNetworkBase& model,
                                                          const OperatorTabuSet& tabu_set) const {
        raise_uninitialized();
        
        auto delta_ptr = delta.data();
        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        auto& spbn = dynamic_cast<const SemiparametricBNBase&>(model);
        for(auto it = sorted_idx.begin(), end = sorted_idx.end(); it != end; ++it) {
            int idx_max = *it;
            const auto& node = model.collapsed_name(idx_max);
            auto node_type = spbn.node_type(node);
            std::shared_ptr<Operator> op = std::make_shared<ChangeNodeType>(node, node_type.opposite(), delta(idx_max));
            if (!tabu_set.contains(op))
                return op;
        }

        return nullptr;
    }

    void ChangeNodeTypeSet::update_scores(const BayesianNetworkBase& model, const Score& score, const Operator& op) {
        raise_uninitialized();

        if (owns_local_cache()) {
            m_local_cache->update_local_score(model, score, op);
        }

        switch(op.type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto& dwn_op = dynamic_cast<const ArcOperator&>(op);
                update_local_delta(model, score, dwn_op.target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto& dwn_op = dynamic_cast<const ArcOperator&>(op);
                update_local_delta(model, score, dwn_op.source());
                update_local_delta(model, score, dwn_op.target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto& dwn_op = dynamic_cast<const ChangeNodeType&>(op);
                int index = model.collapsed_index(dwn_op.node());
                delta(index) = -dwn_op.delta();
            }
                break;
        }
    }
}