#ifndef PGM_DATASET_OPERATORS_HPP
#define PGM_DATASET_OPERATORS_HPP

#include <queue>
#include <Eigen/Dense>
#include <models/BayesianNetwork.hpp>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::Matrix, Eigen::Dynamic;
using MatrixXb = Matrix<bool, Dynamic, Dynamic>;
using VectorXb = Matrix<bool, Dynamic, 1>;

using models::BayesianNetwork, models::SemiparametricBN, models::NodeType;
using util::ArcVector;

namespace learning::operators {

    enum class OperatorType {
        ADD_ARC,
        REMOVE_ARC,
        FLIP_ARC,
        CHANGE_NODE_TYPE
    };
    
    template<typename Model>
    class Operator {
    public:
        Operator(double delta, OperatorType type) : m_delta(delta), m_type(type) {}

        virtual void apply(Model& m) = 0;
        
        double delta() { return m_delta; }
        
        OperatorType type() { return m_type; }
    private:
        double m_delta;
        OperatorType m_type;
    };

    template<typename Model>
    class ArcOperator : public Operator<Model> {
    public:
        ArcOperator(typename Model::node_descriptor source, 
                    typename Model::node_descriptor target,
                    double delta,
                    OperatorType type) : Operator<Model>(delta, type), m_source(source), m_target(target) {}

        typename Model::node_descriptor source() { return m_source; }
        typename Model::node_descriptor target() { return m_target; }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_target;
    };

    template<typename Model>
    class AddArc : public ArcOperator<Model> {
    public:
        AddArc(typename Model::node_descriptor source, 
               typename Model::node_descriptor target,
               double delta) :  ArcOperator<Model>(source, target, delta, OperatorType::ADD_ARC) {}
        
        void apply(Model& m) override {
            m.add_edge(this->source(), this->target());
        }
    };

    template<typename Model>
    class RemoveArc : public ArcOperator<Model> {
    public:
        RemoveArc(typename Model::node_descriptor source, 
                  typename Model::node_descriptor target,
                  double delta) : ArcOperator<Model>(source, target, delta, OperatorType::REMOVE_ARC) {}
        
        void apply(Model& m) override {
            m.remove_edge(this->source(), this->target());
        }
    };

    template<typename Model>
    class FlipArc : public ArcOperator<Model> {
    public:
        FlipArc(typename Model::node_descriptor source, 
                typename Model::node_descriptor target,
                double delta) : ArcOperator<Model>(source, target, delta, OperatorType::FLIP_ARC) {}

        void apply(Model& m) override {
            m.remove_edge(this->source(), this->target());
            m.add_edge(this->target(), this->source());
        }
    };

    template<typename DagType>
    class ChangeNodeType : public Operator<SemiparametricBN<DagType>> {
    public:
        ChangeNodeType(typename SemiparametricBN<DagType>::node_descriptor node,
                       NodeType new_node_type,
                       double delta) : Operator<SemiparametricBN<DagType>>(delta, OperatorType::CHANGE_NODE_TYPE),
                                       m_node(node),
                                       m_new_node_type(new_node_type) {}

        void apply(SemiparametricBN<DagType>& m) override {
            m.set_node_type(m_node, m_new_node_type);
        }

        typename SemiparametricBN<DagType>::node_descriptor node() { return m_node; }
        NodeType node_type() { return m_new_node_type; }
    private:
        typename SemiparametricBN<DagType>::node_descriptor m_node;
        NodeType m_new_node_type;
    };

    class OperatorSetType
    {
    public:
        enum Value : uint8_t
        {
            ARCS,
            NODE_TYPE
        };

        struct Hash
        {
            inline std::size_t operator ()(OperatorSetType const opset_type) const
            {
                return static_cast<std::size_t>(opset_type.value);
            }
        };

        using HashType = Hash;

        OperatorSetType() = default;
        constexpr OperatorSetType(Value opset_type) : value(opset_type) { }

        operator Value() const { return value; }  
        explicit operator bool() = delete;

        constexpr bool operator==(OperatorSetType a) const { return value == a.value; }
        constexpr bool operator==(Value v) const { return value == v; }
        constexpr bool operator!=(OperatorSetType a) const { return value != a.value; }
        constexpr bool operator!=(Value v) const { return value != v; }

        std::string ToString() const { 
            switch(value) {
                case Value::ARCS:
                    return "arcs";
                case Value::NODE_TYPE:
                    return "node_type";
                default:
                    throw std::invalid_argument("Unreachable code in OperatorSetType.");
            }
        }

    private:
        Value value;
    };


    template<typename Model>
    class OperatorSet {
    public:
        virtual void cache_scores(Model& model) = 0;
        virtual std::unique_ptr<Operator<Model>> find_max(Model& model) = 0;
        virtual void update_scores(Model& model, std::unique_ptr<Operator<Model>>& op) = 0;
    };

    template<typename Model, typename Score>
    class ArcOperatorSet : public OperatorSet<Model> {
    public:
        using AddArc_t = AddArc<Model>;
        using RemoveArc_t = RemoveArc<Model>;
        using FlipArc_t = FlipArc<Model>;

        ArcOperatorSet(Model& model, const Score& score, ArcVector& whitelist, ArcVector& blacklist, 
                       const VectorXd& local_score, int max_indegree);

        void cache_scores(Model& model) override;
        std::unique_ptr<Operator<Model>> find_max(Model& model) override;
        
        template<bool limited_indigree>
        std::unique_ptr<Operator<Model>> find_max_indegree(Model& model);

        void update_scores(Model& model, std::unique_ptr<Operator<Model>>& op) override;

        void update_node_arcs_scores(Model& model, typename Model::node_descriptor dest_node);

    private:
        const Score& m_score;
        MatrixXd delta;
        MatrixXb valid_op;
        const VectorXd& m_local_score;
        std::vector<int> sorted_idx;
        int max_indegree;
    };


    template<typename Model, typename Score>
    ArcOperatorSet<Model, Score>::ArcOperatorSet(Model& model,
                                                 const Score& score,
                                                 ArcVector& blacklist,
                                                 ArcVector& whitelist, 
                                                 const VectorXd& local_score,
                                                 int max_indegree) : m_score(score),
                                                                     delta(model.num_nodes(), model.num_nodes()),
                                                                     valid_op(model.num_nodes(), model.num_nodes()), 
                                                                     m_local_score(local_score), 
                                                                     sorted_idx(),
                                                                     max_indegree(max_indegree)
    {
        auto num_nodes = model.num_nodes();
        auto val_ptr = valid_op.data();

        std::fill(val_ptr, val_ptr + num_nodes*num_nodes, true);

        auto indices = model.indices();
        auto valid_ops = (num_nodes * num_nodes) - 2*whitelist.size() - blacklist.size() - num_nodes;

        for(auto whitelist_edge : whitelist) {
            auto source_index = indices[whitelist_edge.first];
            auto dest_index = indices[whitelist_edge.second];

            valid_op(source_index, dest_index) = false;
            valid_op(dest_index, source_index) = false;
            delta(source_index, dest_index) = std::numeric_limits<double>::lowest();
            delta(dest_index, source_index) = std::numeric_limits<double>::lowest();
        }
        
        for(auto blacklist_edge : blacklist) {
            auto source_index = indices[blacklist_edge.first];
            auto dest_index = indices[blacklist_edge.second];

            valid_op(source_index, dest_index) = false;
            delta(source_index, dest_index) = std::numeric_limits<double>::lowest();
        }

        for (int i = 0; i < num_nodes; ++i) {
            valid_op(i, i) = false;
            delta(i, i) = std::numeric_limits<double>::lowest();
        }

        sorted_idx.reserve(valid_ops);

        for (int i = 0; i < num_nodes; ++i) {
            for (int j = 0; j < num_nodes; ++j) {
                if (valid_op(i, j)) {
                    sorted_idx.push_back(i + j * num_nodes);
                }
            }
        }
    }

    template<typename Model, typename Score>
    void ArcOperatorSet<Model, Score>::cache_scores(Model& model) {
        for (auto dest = 0; dest < model.num_nodes(); ++dest) {
            std::vector<int> new_parents_dest = model.get_parent_indices(dest);
            
            for (auto source = 0; source < model.num_nodes(); ++source) {
                if(valid_op(source, dest)) {
                    if (model.has_edge(source, dest)) {
                        std::iter_swap(std::find(new_parents_dest.begin(), new_parents_dest.end(), source), new_parents_dest.end() - 1);
                        double d = m_score.local_score(model, dest, new_parents_dest.begin(), new_parents_dest.end() - 1) - m_local_score(dest);
                        delta(source, dest) = d;
                    } else if (model.has_edge(dest, source)) {
                        auto new_parents_source = model.get_parent_indices(source);
                        std::iter_swap(std::find(new_parents_source.begin(), new_parents_source.end(), dest), new_parents_source.end() - 1);
                        
                        new_parents_dest.push_back(source);
                        double d = m_score.local_score(model, source, new_parents_source.begin(), new_parents_source.end() - 1) + 
                                   m_score.local_score(model, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                   - m_local_score(source) - m_local_score(dest);
                        new_parents_dest.pop_back();
                        delta(dest, source) = d;
                    } else {
                        new_parents_dest.push_back(source);
                        double d = m_score.local_score(model, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                    - m_local_score(dest);
                        new_parents_dest.pop_back();
                        delta(source, dest) = d;
                    }
                }
            }
        }
    }


    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> ArcOperatorSet<Model, Score>::find_max(Model& model) {
        if (max_indegree > 0)
            return find_max_indegree<true>(model);
        else
            return find_max_indegree<false>(model);
    }

    template<typename Model, typename Score>
    template<bool limited_indegree>
    std::unique_ptr<Operator<Model>> ArcOperatorSet<Model, Score>::find_max_indegree(Model& model) {

        auto delta_ptr = delta.data();

        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(); it != sorted_idx.end(); ++it) {
            auto idx = *it;
            auto source = idx % model.num_nodes();
            auto dest = idx / model.num_nodes();

            if(model.has_edge(source, dest)) {
                return std::make_unique<RemoveArc_t>(model.node(source), model.node(dest), delta(source, dest));
            } else if (model.has_edge(dest, source) && model.can_flip_edge(dest, source)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_unique<FlipArc_t>(model.node(dest), model.node(source), delta(dest, source));
            } else if (model.can_add_edge(source, dest)) {
                if constexpr (limited_indegree) {
                    if (model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_unique<AddArc_t>(model.node(source), model.node(dest), delta(source, dest));
            }
        }

        return nullptr;
    }

    template<typename Model, typename Score>
    void ArcOperatorSet<Model, Score>:: update_scores(Model& model, std::unique_ptr<Operator<Model>>& op) {
        switch(op->type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto dwn_op = dynamic_cast<ArcOperator<Model>*>(op.get());
                update_node_arcs_scores(model, dwn_op->target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto dwn_op = dynamic_cast<ArcOperator<Model>*>(op.get());
                update_node_arcs_scores(model, dwn_op->source());
                update_node_arcs_scores(model, dwn_op->target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto dwn_op = dynamic_cast<ChangeNodeType<Model>*>(op.get());
                update_node_arcs_scores(model, dwn_op->node());
            }
                break;
        }
    }

    template<typename Model, typename Score>
    void ArcOperatorSet<Model, Score>::update_node_arcs_scores(Model& model, typename Model::node_descriptor dest_node) {

        auto parents = model.get_parent_indices(dest_node);
        auto dest_idx = model.index(dest_node);
        
        for (int i = 0; i < model.num_nodes(); ++i) {
            if (valid_op(i, dest_idx)) {

                if (model.has_edge(i, dest_idx)) {
                    std::iter_swap(std::find(parents.begin(), parents.end(), i), parents.end() - 1);
                    double d = m_score.local_score(model, dest_idx, parents.begin(), parents.end() - 1) - m_local_score(dest_idx);
                    delta(i, dest_idx) = d;

                    auto new_parents_i = model.get_parent_indices(i);
                    new_parents_i.push_back(dest_idx);

                    delta(dest_idx, i) = d + m_score.local_score(model, i, new_parents_i.begin(), new_parents_i.end())
                                            - m_local_score(i);
                } else if (model.has_edge(dest_idx, i)) {
                    auto new_parents_i = model.get_parent_indices(i);
                    std::iter_swap(std::find(new_parents_i.begin(), new_parents_i.end(), dest_idx), new_parents_i.end() - 1);
                        
                    parents.push_back(i);
                    double d = m_score.local_score(model, i, new_parents_i.begin(), new_parents_i.end() - 1) + 
                                m_score.local_score(model, dest_idx, parents.begin(), parents.end()) 
                                - m_local_score(i) - m_local_score(dest_idx);
                    parents.pop_back();
                    delta(dest_idx, i) = d;
                } else {
                    parents.push_back(i);
                    double d = m_score.local_score(model, dest_idx, parents.begin(), parents.end()) - m_local_score(dest_idx);
                    parents.pop_back();
                    delta(i, dest_idx) = d;
                }
            }
        }
    }

    template<typename Model, typename Score>
    class ChangeNodeTypeSet : public OperatorSet<Model> {
    public:
        using ChangeNodeType_t = ChangeNodeType<Model>;

        ChangeNodeTypeSet(Model& model, 
                          const Score& score, 
                          NodeTypeVector& type_whitelist,
                          const VectorXd& local_score) : m_score(score),
                                                         delta(model.num_nodes()),
                                                         valid_op(model.num_nodes()),
                                                         m_local_score(local_score)
        {
            auto val_ptr = valid_op.data();
            std::fill(val_ptr, val_ptr + model.num_nodes(), true);

            auto indices = model.indices();

            for (auto &node : type_whitelist) {
                delta(indices[node.first]) = std::numeric_limits<double>::lowest();;
                valid_op(indices[node.first]) = false;
            }
        }

        void cache_scores(Model& model) override;
        std::unique_ptr<Operator<Model>> find_max(Model& model) override;
        void update_scores(Model& model, std::unique_ptr<Operator<Model>>& op) override;

        void update_local_delta(Model& model, typename Model::node_descriptor node) {
            update_local_delta(model.index(node));
        }

        void update_local_delta(Model& model, int node_index) {
            NodeType type = model.node_type(node_index);
            auto parents = model.get_parent_indices(node_index);
            delta(node_index) = m_score.local_score(node_index, parents.begin(), parents.end(), type.opposite()) 
                                - m_local_score(node_index);
        }

    private:
        const Score& m_score;
        VectorXd delta;
        VectorXb valid_op;
        const VectorXd& m_local_score;
    };

    template<typename Model, typename Score>
    void ChangeNodeTypeSet<Model, Score>::cache_scores(Model& model) {
        for(auto i = 0; i < model.num_nodes(); ++i) {
            if(valid_op(i)) {
                update_local_delta(i);
            }
        }
    }

    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> ChangeNodeTypeSet<Model, Score>::find_max(Model& model) {
        auto delta_ptr = delta.data();
        auto max_element = std::max_element(delta_ptr, delta_ptr + model.num_nodes());
        auto node_type = model.node_type(*max_element);
        return std::make_unique<ChangeNodeType_t>(model.node(*max_element), node_type.opposite(), delta(*max_element));
    }

    template<typename Model, typename Score>
    void ChangeNodeTypeSet<Model, Score>::update_scores(Model& model, std::unique_ptr<Operator<Model>>& op) {
        switch(op->type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto dwn_op = dynamic_cast<ArcOperator<Model>*>(op.get());
                update_local_delta(model, dwn_op->target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto dwn_op = dynamic_cast<ArcOperator<Model>*>(op.get());
                update_local_delta(model, dwn_op->source());
                update_local_delta(model, dwn_op->target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto dwn_op = dynamic_cast<ChangeNodeType<Model>*>(op.get());
                int index = model.index(dwn_op->node());
                delta(index) = -dwn_op->delta();
            }
                break;
        }
    }

    using OperatorSetTypeS = std::unordered_set<OperatorSetType, typename OperatorSetType::HashType>;

    template<typename Model, typename Score>
    class OperatorPool {
    public:
        template<util::enable_if_semiparametricbn_t<Model, int> = 0>
        OperatorPool(Model& model, const Score& score, OperatorSetTypeS op_sets, ArcVector arc_blacklist, 
                     ArcVector arc_whitelist, NodeTypeVector type_whitelist, int max_indegree) : m_score(score),
                                                                                                 local_score(model.num_nodes()),
                                                                                                 m_op_sets(op_sets.size()),
                                                                                                 max_indegree(max_indegree) 
        {
            if (op_sets.empty()) {
                throw std::invalid_argument("Cannot create an OperatorPool without any OperatorType.");
            }

            for (auto& opset : op_sets) {
                switch(opset) {
                    case OperatorSetType::ARCS: {
                        auto arcs = std::make_unique<ArcOperatorSet<Model, Score>>(model, score, arc_blacklist, arc_whitelist, 
                                                                                        local_score, max_indegree);
                        m_op_sets.push_back(std::move(arcs));
                    }
                        break;
                    case OperatorSetType::NODE_TYPE: {
                        auto change_node_type = std::make_unique<ChangeNodeTypeSet<Model, Score>>(model, score, type_whitelist, local_score);
                        m_op_sets.push_back(std::move(change_node_type));
                    }
                        break;
                }
            }
        };

        void cache_scores(Model& model);
        std::unique_ptr<Operator<Model>> find_max(Model& model);
        void update_scores(Model& model, std::unique_ptr<Operator<Model>>& op);
        
        void update_local_score(Model& model, typename Model::node_descriptor node) {
            update_local_score(model.index(node));
        }
        
        void update_local_score(Model& model, int index) {
            local_score(index) = m_score.local_score(model, index);
        }
        
        double score() {
            return local_score.sum();
        }
    private:
        const Score m_score;
        VectorXd local_score;
        std::vector<std::unique_ptr<OperatorSet<Model>>> m_op_sets;
        int max_indegree;
    };

    template<typename Model, typename Score>
    void OperatorPool<Model, Score>::cache_scores(Model& model) {        
        for (int i = 0; i < model.num_nodes(); ++i) {
            local_score(i) = m_score.local_score(model, i);
        }

        for (auto& op_set : m_op_sets) {
            op_set->cache_scores();
        }
    }

    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> OperatorPool<Model, Score>::find_max(Model& model) {

        double max_delta = std::numeric_limits<double>::lowest();
        std::unique_ptr<Operator<Model>> max_op = nullptr;

        for (auto it = m_op_sets.begin(); it != m_op_sets.end(); ++it) {
            auto new_op = (*it)->find_max();
            if (new_op && new_op->delta() > max_delta) {
                max_op = std::move(new_op);
                max_delta = max_op->delta();
            }
        }

        return max_op;
    }

    template<typename Model, typename Score>
    void OperatorPool<Model, Score>::update_scores(Model& model, std::unique_ptr<Operator<Model>>& op) {
        switch(op->type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto dwn_op = dynamic_cast<ArcOperator<Model>*>(op.get());
                update_local_score(model, dwn_op->target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto dwn_op = dynamic_cast<ArcOperator<Model>*>(op.get());
                update_local_score(model, dwn_op->source());
                update_local_score(model, dwn_op->target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto dwn_op = dynamic_cast<ChangeNodeType<Model>*>(op.get());
                update_local_score(model, dwn_op->node());
            }
                break;
        }

        for (auto& op_set : m_op_sets) {
            op_set->update_scores(op);
        }
    }

}

#endif //PGM_DATASET_OPERATORS_HPP