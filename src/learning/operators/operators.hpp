#include <queue>
#include <Eigen/Dense>
#include <models/BayesianNetwork.hpp>

using Eigen::MatrixXd, Eigen::VectorXd, Eigen::Matrix, Eigen::Dynamic;
using MatrixXb = Matrix<bool, Dynamic, Dynamic>;

using models::BayesianNetwork, models::BayesianNetworkType;
using graph::arc_vector;

namespace learning::operators {

    

    template<typename Model>
    class Operator {
    public:
        Operator(double delta) : m_delta(delta) {}

        virtual bool can_apply(Model& m) = 0;
        virtual void apply_operator(Model& m) = 0;

        double delta() { return m_delta; }
    private:
        double m_delta;
    };

    template<typename Model>
    class ArcOperator : public Operator<Model> {
    public:
        ArcOperator(typename Model::node_descriptor source, typename Model::node_descriptor dest, double delta) : m_source(source),
                                                                                                                  m_dest(dest),
                                                                                                                  Operator<Model>(delta) {}

        typename Model::node_descriptor source() { 
            return m_source;
        }

        typename Model::node_descriptor target() { 
            return m_dest;
        }
    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model>
    class AddArc : public Operator<Model> {
    public:
        AddArc(typename Model::node_descriptor source, 
               typename Model::node_descriptor dest,
               double delta) : m_source(source), m_dest(dest), Operator<Model>(delta) {}
        

        bool can_apply(Model& m) override {
            if (m.num_parents(m_source) == 0 || m.num_children(m_dest) == 0 || !m.has_path(m_dest, m_source)) {
                return true;
            }

            return false;
        }

        void apply_operator(Model& m) override {
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
                  double delta) : m_source(source), m_dest(dest), Operator<Model>(delta) {}
        
        bool can_apply(Model& m) override {
            return true;
        }

        void apply_operator(Model& m) override {
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
                  double delta) : m_source(source), m_dest(dest), Operator<Model>(delta){}

        bool can_apply(Model& m) override {
            if (m.num_parents(m_dest) == 0 || m.num_children(m_source) == 0) {
                return true;
            } else {
                m.remove_edge(m_source, m_dest);
                bool has_path = m.has_path(m_source, m_dest);
                m.add_edge(m_source, m_dest);
                if (has_path) {
                    return false;
                } else {
                    return true;
                }
            }
        }
        
        void apply_operator(Model& m) override {
            m.remove_edge(m_source, m_dest);
            m.add_edge(m_dest, m_source);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model>
    struct greater_operator_delta {
        bool operator()(std::unique_ptr<Operator<Model>>& lhs, std::unique_ptr<Operator<Model>>& rhs) {
            return lhs->delta() >= rhs->delta();
        }
    };

    template<typename Model>
    using priority_op = std::priority_queue<std::unique_ptr<Operator<Model>>, 
                                               std::vector<std::unique_ptr<Operator<Model>>>, 
                                               greater_operator_delta<Model>>;





    // template<typename Model>
    // class priority_op : public std_priority_op<Model> {
        


    //     void remove_op_to_node(node_descriptor node) {

    //         std::remove_if(this->c.begin(), this->c.end(), [](auto op) {

    //         })
            
    //         for (auto it = this->c.begin(); it != this->end())
    //     }
    // }

    template<typename Model, typename Score>
    class ArcOperatorsType {
    public:
        ArcOperatorsType(const DataFrame& df, const Model& model, arc_vector whitelist, arc_vector blacklist);

        void cache_scores(const Model& m);
        void update_scores(const Model& m, std::unique_ptr<Operator<Model>> new_op);
        std::unique_ptr<Operator<Model>> find_max(Model& m);

    private:
        priority_op<Model> delta;
        MatrixXb valid_op;
        VectorXd local_score;
        const DataFrame& df;

        void add_arc_update(Model& model, typename Model::node_descriptor dest);
    };

    template<typename Model, typename Score>
    ArcOperatorsType<Model, Score>::ArcOperatorsType(const DataFrame& df, const Model& model, arc_vector whitelist, arc_vector blacklist) :
                                                    delta(),
                                                    valid_op(model.num_nodes(), model.num_nodes()), 
                                                    local_score(model.num_nodes()), 
                                                    df(df)
    {
        auto nnodes = model.num_nodes();
        auto val_ptr = valid_op.data();

        std::fill(val_ptr, val_ptr + nnodes*nnodes, true);

        auto indices = model.indices();

        for(auto whitelist_edge : whitelist) {
            auto source_index = indices[whitelist_edge.first];
            auto dest_index = indices[whitelist_edge.second];

            valid_op(source_index, dest_index) = false;
            valid_op(dest_index, source_index) = false;
        }
        
        for(auto blacklist_edge : blacklist) {
            auto source_index = indices[blacklist_edge.first];
            auto dest_index = indices[blacklist_edge.second];

            valid_op(source_index, dest_index) = false;
        }

        for (auto i = 0; i < model.num_nodes(); ++i) {
            valid_op(i, i) = false;
        }
    }

    template<typename Model, typename Score>
    void ArcOperatorsType<Model, Score>::cache_scores(const Model& m) {
        std::cout << "valid_op:" << std::endl;
        std::cout << valid_op << std::endl;
        
        for (int i = 0; i < m.num_nodes(); ++i) {
            auto parents = m.get_parent_indices(i);
            local_score(i) = Score::local_score(df, i, parents);
        }

        for (auto dest = 0; dest < m.num_nodes(); ++dest) {
            std::vector<int> new_parents_dest = m.get_parent_indices(dest);
            
            for (auto source = 0; source < m.num_nodes(); ++source) {
                if(valid_op(source, dest)) {

                    if (m.has_edge(source, dest)) {
                        std::iter_swap(std::find(new_parents_dest.begin(), new_parents_dest.end(), source), new_parents_dest.end() - 1);
                        double d = Score::local_score(df, dest, new_parents_dest.begin(), new_parents_dest.end() - 1) - local_score(dest);
                        delta.push_back(std::make_unique<RemoveArc<Model>>(source, dest, d));
                    } else if (m.has_edge(dest, source)) {
                        auto new_parents_source = m.get_parent_indices(source);
                        std::iter_swap(std::find(new_parents_source.begin(), new_parents_source.end(), dest), new_parents_source.end() - 1);
                        
                        new_parents_dest.push_back(source);
                        double d = Score::local_score(df, source, new_parents_source.begin(), new_parents_source.end() - 1) + 
                                   Score::local_score(df, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                   - local_score(source) - local_score(dest);
                        new_parents_dest.pop_back();
                        delta.push_back(std::make_unique<FlipArc<Model>>(dest, source, d));
                    } else {
                        new_parents_dest.push_back(source);
                        double d = Score::local_score(df, dest, new_parents_dest) - local_score(dest);
                        new_parents_dest.pop_back();
                        delta.push_back(std::make_unique<AddArc<Model>>(dest, source, d));
                    }
                }
            }
        }

    }

    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> ArcOperatorsType<Model, Score>::find_max(Model& m) {

        // std::vector<double*> discarded_ops;

        // for (auto i = 0, ; i < delta_heap.size() && !delta_heap.top()->can_apply(m); ++i) {
        //     discarded_ops.push_back(delta.top());
        //     delta.pop();
        // }

        // if (delta.size()) {
        //     std::unique_ptr<Operator<Model>> to_return = delta.top();

        //     for (auto discarded : discarded_ops) {
        //         delta.push(discarded);
        //     }

        //     return std::make_unique<Operator<Model>>(*to_return);
        // } else
        //     return nullptr;
    }

    template<typename Model, typename Score>
    void ArcOperatorsType<Model, Score>::update_scores(const Model& m, std::unique_ptr<Operator<Model>> new_op) {
        
    }

    template<typename Model, typename Score>
    void ArcOperatorsType<Model, Score>::add_arc_update(Model& model, typename Model::node_descriptor dest) {
        
        int idx = model.index(dest);
        
        for(auto other = 0; other < model.num_nodes(); ++other) {
            if (valid_op(other, idx)) {
                if (model.has_edge(other, idx)) {
                    
                } else {

                }
            } else if(model.has_edge(idx, other) && valid_op(idx, other)) {

            }
        }
    }

    template<typename Model, typename Score>
    struct default_operator {};

    template<BayesianNetworkType bn_type, typename Score>
    struct default_operator<BayesianNetwork<bn_type>, Score> {
        using default_operator_t = ArcOperatorsType<BayesianNetwork<bn_type>, Score>;
    };
    

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