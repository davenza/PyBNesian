#ifndef PGM_DATASET_OPERATORS_HPP
#define PGM_DATASET_OPERATORS_HPP

#include <queue>
#include <Eigen/Dense>
#include <models/BayesianNetwork.hpp>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::Matrix, Eigen::Dynamic;
using MatrixXb = Matrix<bool, Dynamic, Dynamic>;

using models::BayesianNetwork, models::SemiparametricBN, models::NodeType;
using graph::arc_vector;

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
    class AddArc : public Operator<Model> {
    public:
        AddArc(typename Model::node_descriptor source, 
               typename Model::node_descriptor dest,
               double delta) :  Operator<Model>(delta, OperatorType::ADD_ARC), m_source(source), m_dest(dest) {}
        
        void apply(Model& m) override {
            m.add_edge(m_source, m_dest);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model>
    class RemoveArc : public Operator<Model> {
    public:
        RemoveArc(typename Model::node_descriptor source, 
                  typename Model::node_descriptor dest,
                  double delta) : Operator<Model>(delta, OperatorType::REMOVE_ARC), m_source(source), m_dest(dest) {}
        
        void apply(Model& m) override {
            m.remove_edge(m_source, m_dest);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model>
    class FlipArc : public Operator<Model> {
    public:
        FlipArc(typename Model::node_descriptor source, 
                  typename Model::node_descriptor dest,
                  double delta) : Operator<Model>(delta, OperatorType::FLIP_ARC), m_source(source), m_dest(dest) {}

        void apply(Model& m) override {
            m.remove_edge(m_source, m_dest);
            m.add_edge(m_dest, m_source);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
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
    private:
        typename SemiparametricBN<DagType>::node_descriptor m_node;
        NodeType m_new_node_type;
    };


    enum class OperatorSetType {
        ARCS,
        NODE_TYPE
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

        ArcOperatorSet(const Score& score, const Model& model, arc_vector whitelist, arc_vector blacklist, int indegree);

        void cache_scores(const Model& m) override;
        std::unique_ptr<Operator<Model>> find_max(Model& m) override;
        
        template<bool limited_indigree>
        std::unique_ptr<Operator<Model>> find_max_indegree(Model& m);

        void update_scores(const Model& m, Operator<Model>& op) override;

        void update_node_arcs_scores(const Model& model, typename Model::node_descriptor dest_node);

        double score() {
            return local_score.sum();
        }

    private:
        const Score& m_score;
        MatrixXd delta;
        MatrixXb valid_op;
        VectorXd local_score;
        std::vector<int> sorted_idx;
        int max_indegree;
    };

    template<typename Model, typename Score> ArcOperatorSet(const Score& df, 
                                                            const Model& model, 
                                                            arc_vector whitelist, 
                                                            arc_vector blacklist, 
                                                            int indegree) -> 
                                                ArcOperatorSet<Model, Score>;


    template<typename Model, typename Score>
    ArcOperatorSet<Model, Score>::ArcOperatorSet(const Score& score, 
                                                     const Model& model, 
                                                     arc_vector whitelist, 
                                                     arc_vector blacklist, 
                                                     int max_indegree) :
                                                    m_score(score),
                                                    delta(model.num_nodes(), model.num_nodes()),
                                                    valid_op(model.num_nodes(), model.num_nodes()), 
                                                    local_score(model.num_nodes()), 
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
    void ArcOperatorSet<Model, Score>::cache_scores(const Model& m) {
        
        for (int i = 0; i < m.num_nodes(); ++i) {
            auto parents = m.get_parent_indices(i);
            local_score(i) = m_score.local_score(m, i, parents);
        }

        for (auto dest = 0; dest < m.num_nodes(); ++dest) {
            std::vector<int> new_parents_dest = m.get_parent_indices(dest);
            
            for (auto source = 0; source < m.num_nodes(); ++source) {
                if(valid_op(source, dest)) {

                    if (m.has_edge(source, dest)) {
                        std::iter_swap(std::find(new_parents_dest.begin(), new_parents_dest.end(), source), new_parents_dest.end() - 1);
                        double d = m_score.local_score(m, dest, new_parents_dest.begin(), new_parents_dest.end() - 1) - local_score(dest);
                        delta(source, dest) = d;
                    } else if (m.has_edge(dest, source)) {
                        auto new_parents_source = m.get_parent_indices(source);
                        std::iter_swap(std::find(new_parents_source.begin(), new_parents_source.end(), dest), new_parents_source.end() - 1);
                        
                        new_parents_dest.push_back(source);
                        double d = m_score.local_score(m, source, new_parents_source.begin(), new_parents_source.end() - 1) + 
                                   m_score.local_score(m, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                   - local_score(source) - local_score(dest);
                        new_parents_dest.pop_back();
                        delta(dest, source) = d;
                    } else {
                        new_parents_dest.push_back(source);
                        double d = m_score.local_score(m, dest, new_parents_dest) - local_score(dest);
                        new_parents_dest.pop_back();
                        delta(source, dest) = d;
                    }
                }
            }
        }
    }


    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> ArcOperatorSet<Model, Score>::find_max(Model& m) {
        if (max_indegree > 0)
            return find_max_indegree<true>(m);
        else
            return find_max_indegree<false>(m);
    }

    template<typename Model, typename Score>
    template<bool limited_indegree>
    std::unique_ptr<Operator<Model>> ArcOperatorSet<Model, Score>::find_max_indegree(Model& m) {

        auto delta_ptr = delta.data();

        // TODO: Not checking sorted_idx empty
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&delta_ptr](auto i1, auto i2) {
            return delta_ptr[i1] >= delta_ptr[i2];
        });

        for(auto it = sorted_idx.begin(); it != sorted_idx.end(); ++it) {
            auto idx = *it;
            auto source = idx % m.num_nodes();
            auto dest = idx / m.num_nodes();

            if(m.has_edge(source, dest)) {
                return std::make_unique<RemoveArc_t>(m.node(source), m.node(dest), delta(source, dest));
            } else if (m.has_edge(dest, source) && m.can_flip_edge(dest, source)) {
                if constexpr (limited_indegree) {
                    if (m.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_unique<FlipArc_t>(m.node(dest), m.node(source), delta(dest, source));
            } else if (m.can_add_edge(source, dest)) {
                if constexpr (limited_indegree) {
                    if (m.num_parents(dest) >= max_indegree) {
                        continue;
                    }
                }
                return std::make_unique<AddArc_t>(m.node(source), m.node(dest), delta(source, dest));
            }
        }

        return nullptr;
    }

    template<typename Model, typename Score>
    void ArcOperatorSet<Model, Score>:: update_scores(const Model& m, Operator<Model>& op) {
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
    void ArcOperatorSet<Model, Score>::update_node_arcs_scores(const Model& model, typename Model::node_descriptor dest_node) {

        auto parents = model.get_parent_indices(dest_node);
        auto dest_idx = model.index(dest_node);
        local_score(dest_idx) = m_score.local_score(model, dest_idx, parents);
        
        for (int i = 0; i < model.num_nodes(); ++i) {
            if (valid_op(i, dest_idx)) {

                if (model.has_edge(i, dest_idx)) {
                    std::iter_swap(std::find(parents.begin(), parents.end(), i), parents.end() - 1);
                    double d = m_score.local_score(model, dest_idx, parents.begin(), parents.end() - 1) - local_score(dest_idx);
                    delta(i, dest_idx) = d;

                    auto new_parents_i = model.get_parent_indices(i);
                    new_parents_i.push_back(dest_idx);

                    delta(dest_idx, i) = d + m_score.local_score(model, i, new_parents_i.begin(), new_parents_i.end())
                                            - local_score(i);
                } else if (model.has_edge(dest_idx, i)) {
                    auto new_parents_i = model.get_parent_indices(i);
                    std::iter_swap(std::find(new_parents_i.begin(), new_parents_i.end(), dest_idx), new_parents_i.end() - 1);
                        
                    parents.push_back(i);
                    double d = m_score.local_score(model, i, new_parents_i.begin(), new_parents_i.end() - 1) + 
                                m_score.local_score(model, dest_idx, parents.begin(), parents.end()) 
                                - local_score(i) - local_score(dest_idx);
                    parents.pop_back();
                    delta(dest_idx, i) = d;
                } else {
                    parents.push_back(i);
                    double d = m_score.local_score(model, dest_idx, parents) - local_score(dest_idx);
                    parents.pop_back();
                    delta(i, dest_idx) = d;
                }
            }
        }
    }

    // template<typename Model>
    // class OperatorPoolBase {


    template<typename Model>
    class OperatorPool {
    public:
        OperatorPool(std::vector<std::unique_ptr<OperatorSet<Model>>> op_sets) : m_op_sets(op_sets) {
            if (op_sets.empty()) {
                throw std::invalid_argument("Cannot create an OperatorPool without any OperatorType.");
            }
        };

        void cache_scores(const Model& model);
        std::unique_ptr<Operator<Model>> find_max(const Model& model);
        void update_scores(const Model& model, Operator<Model>& op);
    private:
        std::vector<std::unique_ptr<OperatorSet<Model>>> m_op_sets;
    };

    template<typename Model>
    void OperatorPool<Model>::cache_scores(const Model& model) {
        for (auto& op_set : m_op_sets) {
            op_set->cache_scores(model);
        }
    }

    template<typename Model>
    std::unique_ptr<Operator<Model>> OperatorPool<Model>::find_max(const Model& model) {
        std::unique_ptr<Operator<Model>> max_op = m_op_sets[0]->find_max(model);

        for (auto it = ++m_op_sets.begin(); it != m_op_sets.end(); ++it) {
            auto new_op = (*it)->find_max(model);
            if (new_op->delta() > max_op->delta()) {
                max_op = std::move(new_op);
            }
        }

        return max_op;
    }

    template<typename Model>
    void OperatorPool<Model>::update_scores(const Model& model, Operator<Model>& op) {
        for (auto& op_set : m_op_sets) {
            op_set->update_scores(model, op);
        }
    }


    // template<typename Model, typename Score>
    // class ArcOperatorsType {
    // public:
    //     using AddArc_t = AddArc<Model, ArcOperatorsType<Model, Score>>;
    //     using RemoveArc_t = RemoveArc<Model, ArcOperatorsType<Model, Score>>;
    //     using FlipArc_t = FlipArc<Model, ArcOperatorsType<Model, Score>>;

    //     ArcOperatorsType(const DataFrame& df, const Model& model, arc_vector whitelist, arc_vector blacklist, int max_indegree);

    //     void cache_scores(const Model& m);
    //     void update_node_arcs_scores(Model& model, typename Model::node_descriptor dest_node);
    //     std::unique_ptr<Operator<Model, ArcOperatorsType<Model, Score>>> find_max(Model& m);

    // private:
    //     MatrixXd delta;
    //     std::vector<int> sorted_idx;
    //     MatrixXb valid_op;
    //     VectorXd local_score;
    //     const DataFrame& df;
    //     int max_indegree;
    // };

    // template<typename Model, typename Score>
    // struct default_operator {};

    // template<BayesianNetworkType bn_type, typename Score>
    // struct default_operator<BayesianNetwork<bn_type>, Score> {
    //     using default_operator_t = ArcOperatorsType<BayesianNetwork<bn_type>, Score>;
    // };
    

    // template<typename Model, typename Score>
    // class OperatorPool {
    // public:

    //     OperatorPool(std::vector<std::unique_ptr<OperatorType<Model>>>&& ops) : operator_types(std::move(ops)) {}

    //     void cache_scores(const Model& m);
    //     void update_scores(const Model& m, std::unique_ptr<Operator<Model>> new_op);
    //     std::unique_ptr<Operator<Model>> find_max(const Model& m);

    // private:
    //     std::vector<std::unique_ptr<OperatorType<Model>>> operator_types;
    // };


    // template<typename Model, typename Score>
    // class DefaultOperatorPool {
    // public:
    //     using default_operator_t = typename default_operator<Model, Score>::default_operator_t;

    //     DefaultOperatorPool(int n_nodes) : operator_type(n_nodes) {}

    //     void cache_scores(const Model& m);
    //     void update_scores(const Model& m, std::unique_ptr<Operator<Model>> new_op);
    //     std::unique_ptr<Operator<Model>> find_max(const Model& m);

    // private:
    //     default_operator_t operator_type;
    // };

}

#endif //PGM_DATASET_OPERATORS_HPP