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
    class ArcOperator : Operator<Model> {

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
        constexpr bool operator!=(OperatorSetType a) const { return value != a.value; }

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
        virtual void cache_scores(const Model& m) = 0;
        virtual std::unique_ptr<Operator<Model>> find_max(Model& m) = 0;
        virtual void update_scores(const Model& m, Operator<Model>& op) = 0;
    };

    template<typename Model, typename Score>
    class ArcOperatorSet : public OperatorSet<Model> {
    public:
        using AddArc_t = AddArc<Model>;
        using RemoveArc_t = RemoveArc<Model>;
        using FlipArc_t = FlipArc<Model>;

        ArcOperatorSet(const Model& model, const Score& score, ArcVector& whitelist, ArcVector& blacklist, 
                       const VectorXd& local_score, int max_indegree);

        void cache_scores() override;
        std::unique_ptr<Operator<Model>> find_max() override;
        
        template<bool limited_indigree>
        std::unique_ptr<Operator<Model>> find_max_indegree();

        void update_scores(Operator<Model>& op) override;

        void update_node_arcs_scores(typename Model::node_descriptor dest_node);

    private:
        const Model& m_model;
        const Score& m_score;
        MatrixXd delta;
        MatrixXb valid_op;
        const VectorXd& m_local_score;
        std::vector<int> sorted_idx;
        int max_indegree;
    };


    template<typename Model, typename Score>
    ArcOperatorSet<Model, Score>::ArcOperatorSet(const Model& model,
                                                 const Score& score,
                                                 ArcVector& whitelist, 
                                                 ArcVector& blacklist,
                                                 const VectorXd& local_score,
                                                 int max_indegree) : m_model(model),
                                                                     m_score(score),
                                                                     delta(model.num_nodes(), model.num_nodes()),
                                                                     valid_op(model.num_nodes(), model.num_nodes()), 
                                                                     m_local_score(local_score), 
                                                                     sorted_idx(),
                                                                     max_indegree(max_indegree)
    {
        int nnodes = model.num_nodes();
        auto val_ptr = valid_op.data();

        std::fill(val_ptr, val_ptr + nnodes*nnodes, true);

        auto indices = model.indices();
        auto valid_ops = (nnodes * nnodes) - 2*whitelist.size() - blacklist.size() - nnodes;

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

        for (int i = 0; i < nnodes; ++i) {
            valid_op(i, i) = false;
            delta(i, i) = std::numeric_limits<double>::lowest();
        }

        sorted_idx.reserve(valid_ops);

        for (int i = 0; i < nnodes; ++i) {
            for (int j = 0; j < nnodes; ++j) {
                if (valid_op(i, j)) {
                    sorted_idx.push_back(i + j * nnodes);
                }
            }
        }
    }

    template<typename Model, typename Score>
    void ArcOperatorSet<Model, Score>::cache_scores() {
        for (auto dest = 0; dest < m_model.num_nodes(); ++dest) {
            std::vector<int> new_parents_dest = m_model.get_parent_indices(dest);
            
            for (auto source = 0; source < m_model.num_nodes(); ++source) {
                if(valid_op(source, dest)) {

                    if (m_model.has_edge(source, dest)) {
                        std::iter_swap(std::find(new_parents_dest.begin(), new_parents_dest.end(), source), new_parents_dest.end() - 1);
                        double d = m_score.local_score(m_model, dest, new_parents_dest.begin(), new_parents_dest.end() - 1) - m_local_score(dest);
                        delta(source, dest) = d;
                    } else if (m_model.has_edge(dest, source)) {
                        auto new_parents_source = m_model.get_parent_indices(source);
                        std::iter_swap(std::find(new_parents_source.begin(), new_parents_source.end(), dest), new_parents_source.end() - 1);
                        
                        new_parents_dest.push_back(source);
                        double d = m_score.local_score(m_model, source, new_parents_source.begin(), new_parents_source.end() - 1) + 
                                   m_score.local_score(m_model, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                   - m_local_score(source) - m_local_score(dest);
                        new_parents_dest.pop_back();
                        delta(dest, source) = d;
                    } else {
                        new_parents_dest.push_back(source);
                        double d = m_score.local_score(m_model, dest, new_parents_dest) - m_local_score(dest);
                        new_parents_dest.pop_back();
                        delta(source, dest) = d;
                    }
                }
            }
        }
    }


    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> ArcOperatorSet<Model, Score>::find_max() {
        if (max_indegree > 0)
            return find_max_indegree<true>();
        else
            return find_max_indegree<false>();
    }

    template<typename Model, typename Score>
    template<bool limited_indegree>
    std::unique_ptr<Operator<Model>> ArcOperatorSet<Model, Score>::find_max_indegree() {

        auto delta_ptr = delta.data();

        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(); it != sorted_idx.end(); ++it) {
            auto idx = *it;
            auto source = idx % m_model.num_nodes();
            auto dest = idx / m_model.num_nodes();

            if(m_model.has_edge(source, dest)) {
                return std::make_unique<RemoveArc_t>(m_model.node(source), m_model.node(dest), delta(source, dest));
            } else if (m_model.has_edge(dest, source) && m_model.can_flip_edge(dest, source)) {
                if constexpr (limited_indegree) {
                    if (m_model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_unique<FlipArc_t>(m_model.node(dest), m_model.node(source), delta(dest, source));
            } else if (m_model.can_add_edge(source, dest)) {
                if constexpr (limited_indegree) {
                    if (m_model.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_unique<AddArc_t>(m_model.node(source), m_model.node(dest), delta(source, dest));
            }
        }

        return nullptr;
    }

    template<typename Model, typename Score>
    void ArcOperatorSet<Model, Score>:: update_scores(Operator<Model>& op) {
        switch(op.type()) {
            case OperatorType::ADD_ARC:
                std::cout << "Add arc" << std::endl;
                break;
            case OperatorType::REMOVE_ARC:
                std::cout << "Remove arc" << std::endl;
                break;
            case OperatorType::FLIP_ARC:
                std::cout << "Flip arc" << std::endl;
                break;
            case OperatorType::CHANGE_NODE_TYPE:
                std::cout << "Change Node Type" << std::endl;
                break;
        }
    }

    template<typename Model, typename Score>
    void ArcOperatorSet<Model, Score>::update_node_arcs_scores(typename Model::node_descriptor dest_node) {

        auto parents = m_model.get_parent_indices(dest_node);
        auto dest_idx = m_model.index(dest_node);
        local_score(dest_idx) = m_score.local_score(m_model, dest_idx, parents);
        
        for (int i = 0; i < m_model.num_nodes(); ++i) {
            if (valid_op(i, dest_idx)) {

                if (m_model.has_edge(i, dest_idx)) {
                    std::iter_swap(std::find(parents.begin(), parents.end(), i), parents.end() - 1);
                    double d = m_score.local_score(m_model, dest_idx, parents.begin(), parents.end() - 1) - m_local_score(dest_idx);
                    delta(i, dest_idx) = d;

                    auto new_parents_i = m_model.get_parent_indices(i);
                    new_parents_i.push_back(dest_idx);

                    delta(dest_idx, i) = d + m_score.local_score(m_model, i, new_parents_i.begin(), new_parents_i.end())
                                            - m_local_score(i);
                } else if (m_model.has_edge(dest_idx, i)) {
                    auto new_parents_i = m_model.get_parent_indices(i);
                    std::iter_swap(std::find(new_parents_i.begin(), new_parents_i.end(), dest_idx), new_parents_i.end() - 1);
                        
                    parents.push_back(i);
                    double d = m_score.local_score(m_model, i, new_parents_i.begin(), new_parents_i.end() - 1) + 
                                m_score.local_score(m_model, dest_idx, parents.begin(), parents.end()) 
                                - m_local_score(i) - m_local_score(dest_idx);
                    parents.pop_back();
                    delta(dest_idx, i) = d;
                } else {
                    parents.push_back(i);
                    double d = m_score.local_score(m_model, dest_idx, parents) - m_local_score(dest_idx);
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

        ChangeNodeTypeSet(const Model& model, 
                          const Score& score, 
                          NodeTypeVector& type_whitelist,
                          const VectorXd& local_score) : m_model(model),
                                                         m_score(score),
                                                         delta(model.num_nodes()),
                                                         valid_op(model.num_nodes()),
                                                         m_local_score(local_score)
        {
            auto val_ptr = valid_op.data();
            std::fill(val_ptr, m_model.num_nodes(), true);

            auto indices = m_model.indices();

            for (auto &node : type_whitelist) {
                delta(indices[node.first]) = std::numeric_limits<double>::lowest();;
                valid_op(indices[node.first]) = false;
            }

            auto valid_ops = m_model.num_nodes() - type_whitelist.size();
        }

        void cache_scores() override;
        std::unique_ptr<Operator<Model>> find_max() override;
        void update_scores(Operator<Model>& op) override;

        void update_local_delta(typename Model::node_descriptor node) {
            update_local_delta(m_model.index(node));
        }

        void update_local_delta(int node_index) {
            NodeType type = m_model.node_type(node_index);
            auto parents = m_model.get_parent_indices(node_index);
            delta(node_index) = m_score.local_score(type.opposite(), node_index, parents) - m_local_score(node_index);
        }

    private:
        const Model& m_model;
        const Score& m_score;
        VectorXd delta;
        VectorXb valid_op;
        const VectorXd& m_local_score;
    };

    template<typename Model, typename Score>
    void ChangeNodeTypeSet<Model, Score>::cache_scores() {
        for(auto i = 0; i < m_model.num_nodes; ++i) {
            if(valid_op(i)) {
                update_local_delta(i);
            }
        }
    }

    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> ChangeNodeTypeSet<Model, Score>::find_max() {
        auto delta_ptr = delta.data();
        auto max_element = std::max_element(delta_ptr, delta_ptr + m_model.num_nodes());
        auto node_type = m_model.node_type(*max_element);
        return std::make_unique<ChangeNodeType_t>(m_model.node(*max_element), node_type.opposite(), delta(*max_element));
    }

    template<typename Model, typename Score>
    void ChangeNodeTypeSet<Model, Score>::update_scores(Operator<Model>& op) {
        switch(op.type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto dwn_op = dynamic_cast<ArcOperator<Model>>(op);
                update_local_delta(op.target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto dwn_op = dynamic_cast<ArcOperator<Model>>(op);
                update_local_delta(op.source());
                update_local_delta(op.target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto dwn_op = dynamic_cast<ChangeNodeType<Model>>(op);
                int index = m_model.index(dwn_op.node());
                delta(index) = -op.delta();
            }
                break;
        }
    }

    template<typename Model, typename Score>
    class OperatorPool {
    public:
        OperatorPool(const Model& model, const Score& score, int max_indegree,
                     std::vector<OperatorSet<Model>> op_sets) : 
                                                            m_model(model),
                                                            m_score(score),
                                                            local_score(model.num_nodes()),
                                                            m_op_sets(op_sets),
                                                            max_indegree(max_indegree) {
            if (op_sets.empty()) {
                throw std::invalid_argument("Cannot create an OperatorPool without any OperatorType.");
            }
        };

        void cache_scores();
        std::unique_ptr<Operator<Model>> find_max();
        void update_scores(Operator<Model>& op);
        
        void update_local_score(typename Model::node_descriptor node) {
            update_local_score(m_model.index(node));
        }
        
        void update_local_score(int index) {
            auto parents = m_model.get_parent_indices(index);
            local_score(index) = m_score.local_score(m_model, index, parents);
        }
        
        double score() {
            return local_score.sum();
        }
    private:
        Model& m_model;
        const Score m_score;
        VectorXd local_score;
        std::vector<std::unique_ptr<OperatorSet<Model>>> m_op_sets;
        int max_indegree;
    };

    template<typename Model, typename Score>
    void OperatorPool<Model, Score>::cache_scores() {        
        for (int i = 0; i < m_model.num_nodes(); ++i) {
            auto parents = m_model.get_parent_indices(i);
            local_score(i) = m_score.local_score(m_model, i, parents);
        }

        for (auto& op_set : m_op_sets) {
            op_set->cache_scores();
        }
    }

    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> OperatorPool<Model, Score>::find_max() {

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
    void OperatorPool<Model, Score>::update_scores(Operator<Model>& op) {
        switch(op.type()) {
            case OperatorType::ADD_ARC:
            case OperatorType::REMOVE_ARC: {
                auto dwn_op = dynamic_cast<ArcOperator<Model>>(op);
                update_local_score(op.target());
            }
                break;
            case OperatorType::FLIP_ARC: {
                auto dwn_op = dynamic_cast<ArcOperator<Model>>(op);
                update_local_score(op.source());
                update_local_score(op.target());
            }
                break;
            case OperatorType::CHANGE_NODE_TYPE: {
                auto dwn_op = dynamic_cast<ChangeNodeType<Model>>(op);
                update_local_score(op.node());
            }
                break;
        }

        for (auto& op_set : m_op_sets) {
            op_set->update_scores(op);
        }
    }

}

#endif //PGM_DATASET_OPERATORS_HPP