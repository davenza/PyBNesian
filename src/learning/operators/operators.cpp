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
}