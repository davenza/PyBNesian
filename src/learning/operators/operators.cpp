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

    void OperatorPool::cache_scores(BayesianNetworkBase& model) {
        local_cache->cache_local_scores(model, *m_score);

        for (auto& op_set : m_op_sets) {
            op_set->cache_scores(model, *m_score);
        }
    }

    std::shared_ptr<Operator> OperatorPool::find_max(BayesianNetworkBase& model) {

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

    std::shared_ptr<Operator> OperatorPool::find_max(BayesianNetworkBase& model, OperatorTabuSet& tabu_set) {
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

    void OperatorPool::update_scores(BayesianNetworkBase& model, Operator& op) {
        local_cache->update_local_score(model, *m_score, op);
        for (auto& op_set : m_op_sets) {
            op_set->update_scores(model, *m_score, op);
        }
    }

    void ArcOperatorSet::update_listed_arcs(BayesianNetworkBase& model) {
        if (required_arclist_update) {
            int num_nodes = model.num_nodes();
            if (delta.rows() != num_nodes) {
                delta = MatrixXd(num_nodes, num_nodes);
                valid_op = MatrixXb(num_nodes, num_nodes);
            }

            auto val_ptr = valid_op.data();

            std::fill(val_ptr, val_ptr + num_nodes*num_nodes, true);

            auto indices = model.indices();
            auto valid_ops = (num_nodes * num_nodes) - 2*m_whitelist.size() - m_blacklist.size() - num_nodes;

            for(auto whitelist_edge : m_whitelist) {
                auto source_pair = indices.find(whitelist_edge.first);
                if (source_pair == indices.end())
                    throw std::invalid_argument("Node " + whitelist_edge.first + " present in the" 
                                                    " whitelist list, but not present in the Bayesian network.");
                int source_index = source_pair->second;
                
                auto target_pair = indices.find(whitelist_edge.second);
                if (target_pair == indices.end())
                    throw std::invalid_argument("Node " + whitelist_edge.second + " present in the" 
                                                    " whitelist list, but not present in the Bayesian network.");
                int target_index = target_pair->second;

                valid_op(source_index, target_index) = false;
                valid_op(target_index, source_index) = false;
                delta(source_index, target_index) = std::numeric_limits<double>::lowest();
                delta(target_index, source_index) = std::numeric_limits<double>::lowest();
            }
            
            for(auto blacklist_edge : m_blacklist) {
                auto source_pair = indices.find(blacklist_edge.first);
                if (source_pair == indices.end())
                    throw std::invalid_argument("Node " + blacklist_edge.first + " present in the"
                                                    " blacklist list, but not present in the Bayesian network.");
                int source_index = source_pair->second;
                
                auto target_pair = indices.find(blacklist_edge.second);
                if (target_pair == indices.end())
                    throw std::invalid_argument("Node " + blacklist_edge.second + " present in the" 
                                                    "blacklist list, but not present in the Bayesian network.");
                int target_index = target_pair->second;


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

    void ArcOperatorSet::cache_scores(BayesianNetworkBase& model, Score& score) {
        if (!util::compatible_score(model, score.type())) {
            throw std::invalid_argument("Invalid score " + score.ToString() + " for model type " + model.type().ToString() + ".");
        }

        update_listed_arcs(model);

        if (this->m_local_cache == nullptr) {
            auto lc = std::make_shared<LocalScoreCache>(model);
            this->set_local_score_cache(lc);
            this->m_local_cache->cache_local_scores(model, score);
        } else if (this->owns_local_cache()) {
            this->m_local_cache->cache_local_scores(model, score);
        }

        for (auto dest = 0; dest < model.num_nodes(); ++dest) {
            std::vector<int> new_parents_dest = model.parent_indices(dest);
            
            for (auto source = 0; source < model.num_nodes(); ++source) {
                if(valid_op(source, dest)) {
                    if (model.has_arc(source, dest)) {            
                        std::iter_swap(std::find(new_parents_dest.begin(), new_parents_dest.end(), source), new_parents_dest.end() - 1);
                        double d = score.local_score(model, dest, new_parents_dest.begin(), new_parents_dest.end() - 1) - 
                                    this->m_local_cache->local_score(dest);
                        delta(source, dest) = d;
                    } else if (model.has_arc(dest, source)) {
                        auto new_parents_source = model.parent_indices(source);
                        util::swap_remove_v(new_parents_source, dest);
                        
                        new_parents_dest.push_back(source);
                        double d = score.local_score(model, source, new_parents_source.begin(), new_parents_source.end()) + 
                                   score.local_score(model, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                   - this->m_local_cache->local_score(source) - this->m_local_cache->local_score(dest);
                        new_parents_dest.pop_back();
                        delta(dest, source) = d;
                    } else {
                        new_parents_dest.push_back(source);
                        double d = score.local_score(model, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                    - this->m_local_cache->local_score(dest);
                        new_parents_dest.pop_back();
                        delta(source, dest) = d;
                    }
                }
            }
        }
    }

    std::shared_ptr<Operator> ArcOperatorSet::find_max(BayesianNetworkBase& model) {
        if (this->m_local_cache == nullptr) {
            throw pybind11::value_error("Local cache not initialized. Call cache_scores() before find_max()");
        }

        if (max_indegree > 0)
            return find_max_indegree<true>(model);
        else
            return find_max_indegree<false>(model);
    }

    std::shared_ptr<Operator> ArcOperatorSet::find_max(BayesianNetworkBase& model, OperatorTabuSet& tabu_set) {
        if (this->m_local_cache == nullptr) {
            throw pybind11::value_error("Local cache not initialized. Call cache_scores() before find_max()");
        }

        if (max_indegree > 0)
            return find_max_indegree<true>(model, tabu_set);
        else
            return find_max_indegree<false>(model, tabu_set);
    }

    void ArcOperatorSet::update_scores(BayesianNetworkBase& model, Score& score, Operator& op) {
        if (this->m_local_cache == nullptr) {
            auto lc = std::make_shared<LocalScoreCache>(model);
            this->set_local_score_cache(lc);
            this->m_local_cache->update_local_score(model, score, op);
        } else if(this->owns_local_cache()) {
            this->m_local_cache->update_local_score(model, score, op);
        }

        switch(op.type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto& dwn_op = dynamic_cast<ArcOperator&>(op);
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
                    std::iter_swap(std::find(parents.begin(), parents.end(), i), parents.end() - 1);
                    double d = score.local_score(model, dest_idx, parents.begin(), parents.end() - 1) - 
                               this->m_local_cache->local_score(dest_idx);
                    delta(i, dest_idx) = d;

                    auto new_parents_i = model.parent_indices(i);
                    new_parents_i.push_back(dest_idx);

                    delta(dest_idx, i) = d + score.local_score(model, i, new_parents_i.begin(), new_parents_i.end())
                                            - this->m_local_cache->local_score(i);
                } else if (model.has_arc(dest_idx, i)) {
                    auto new_parents_i = model.parent_indices(i);
                    util::swap_remove_v(new_parents_i, dest_idx);

                    parents.push_back(i);
                    double d = score.local_score(model, i, new_parents_i.begin(), new_parents_i.end()) + 
                               score.local_score(model, dest_idx, parents.begin(), parents.end()) 
                                - this->m_local_cache->local_score(i) - this->m_local_cache->local_score(dest_idx);
                    parents.pop_back();
                    delta(dest_idx, i) = d;
                } else {
                    parents.push_back(i);
                    double d = score.local_score(model, dest_idx, parents.begin(), parents.end()) - 
                                this->m_local_cache->local_score(dest_idx);
                    parents.pop_back();
                    delta(i, dest_idx) = d;
                }
            }
        }
    }

    void ChangeNodeTypeSet::cache_scores(BayesianNetworkBase& model, Score& score) {
        if (model.type() != BayesianNetworkType::SPBN) {
            throw std::invalid_argument("ChangeNodeTypeSet can only be used with SemiparametricBN");
        }

        if (!util::compatible_score(model, score.type())) {
            throw std::invalid_argument("Invalid score " + score.ToString() + " for model type " + model.type().ToString() + ".");
        }
        
        update_whitelisted(model);

        if (this->m_local_cache == nullptr) {
            auto lc = std::make_shared<LocalScoreCache>(model);
            this->set_local_score_cache(lc);
            this->m_local_cache->cache_local_scores(model, score);
        } else if (this->owns_local_cache()) {
            this->m_local_cache->cache_local_scores(model, score);
        }

        for(int i = 0, num_nodes = model.num_nodes(); i < num_nodes; ++i) {
            if(valid_op(i)) {
                update_local_delta(model, score, i);
            }
        }
    }

    std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(BayesianNetworkBase& model) {
        if (this->m_local_cache == nullptr) {
            throw pybind11::value_error("Local cache not initialized. Call cache_scores() before find_max()");
        }

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

    std::shared_ptr<Operator> ChangeNodeTypeSet::find_max(BayesianNetworkBase& model, OperatorTabuSet& tabu_set) {
        if (this->m_local_cache == nullptr) {
            throw pybind11::value_error("Local cache not initialized. Call cache_scores() before find_max()");
        }
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

    void ChangeNodeTypeSet::update_scores(BayesianNetworkBase& model, Score& score, Operator& op) {
        if (this->m_local_cache == nullptr) {
            auto lc = std::make_shared<LocalScoreCache>(model);
            this->set_local_score_cache(lc);
            this->m_local_cache->update_local_score(model, score, op);
        } else if(this->owns_local_cache()) {
            this->m_local_cache->update_local_score(model, score, op);
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
}