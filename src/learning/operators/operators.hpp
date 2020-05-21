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
        virtual void apply_operator(Model& m) = 0;

        double delta() { return delta; }
    private:
        double m_delta;
    };

    template<typename Model>
    class AddEdge : public Operator<Model> {

        AddEdge(typename Model::node_descriptor source, 
               typename Model::node_descriptor dest,
               double delta) : m_source(source), m_dest(dest), Operator<Model>(delta) {}
        
        void apply_operator(Model& m) override {
            m.add_edge(m_source, m_dest);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model>
    class RemoveEdge : public Operator<Model> {

        RemoveEdge(typename Model::node_descriptor source, 
                  typename Model::node_descriptor dest,
                  double delta) : m_source(source), m_dest(dest), Operator<Model>(delta) {}
        
        void apply_operator(Model& m) override {
            m.remove_edge(m_source, m_dest);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model>
    class FlipEdge : public Operator<Model> {

        FlipEdge(typename Model::node_descriptor source, 
                  typename Model::node_descriptor dest,
                  double delta) : m_source(source), m_dest(dest), Operator<Model>(delta){}
        
        void apply_operator(Model& m) override {
            m.flip_edge(m_source, m_dest);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };



    template<typename Model, typename Score>
    class ArcOperatorsType {
    public:

        ArcOperatorsType(const DataFrame& df, const Model& model, arc_vector whitelist, arc_vector blacklist);

        void cache_scores();
        void update_scores(std::unique_ptr<Operator<Model>> new_op);
        std::unique_ptr<Operator<Model>> find_max(const Model& m);

    private:
        MatrixXd delta;
        MatrixXb valid_op;
        VectorXd local_score;
        const DataFrame& df;
        const Model& model;
    };

    template<typename Model, typename Score>
    ArcOperatorsType<Model, Score>::ArcOperatorsType(const DataFrame& df, const Model& model, arc_vector whitelist, arc_vector blacklist) :
                                                    delta(model.num_nodes(), model.num_nodes()), 
                                                    valid_op(model.num_nodes(), model.num_nodes()), 
                                                    local_score(model.num_nodes()), 
                                                    df(df), 
                                                    model(model)
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

            delta(source_index, dest_index) = std::numeric_limits<double>::lowest();
            delta(dest_index, source_index) = std::numeric_limits<double>::lowest();
        }
        
        for(auto blacklist_edge : blacklist) {
            auto source_index = indices[blacklist_edge.first];
            auto dest_index = indices[blacklist_edge.second];

            valid_op(source_index, dest_index) = false;
            delta(source_index, dest_index) = std::numeric_limits<double>::lowest();
        }

        for (auto i = 0; i < model.num_nodes(); ++i) {
            valid_op(i, i) = false;
            delta(i, i) = std::numeric_limits<double>::lowest();
        }
    }

    template<typename Model, typename Score>
    void ArcOperatorsType<Model, Score>::cache_scores() {
        std::cout << "valid_op:" << std::endl;
        std::cout << valid_op << std::endl;
        
        for (int i = 0; i < model.num_nodes(); ++i) {
            auto parents = model.get_parent_indices(i);
            local_score(i) = Score::local_score(df, i, parents);
        }

        for (auto dest = 0; dest < model.num_nodes(); ++dest) {
            std::vector<int> new_parents_dest = model.get_parent_indices(dest);
            
            for (auto source = 0; source < model.num_nodes(); ++source) {
                if(valid_op(source, dest)) {

                    if (model.has_edge(source, dest)) {
                        std::iter_swap(std::find(new_parents_dest.begin(), new_parents_dest.end(), source), new_parents_dest.end() - 1);
                        delta(source, dest) = Score::local_score(df, dest, new_parents_dest.begin(), new_parents_dest.end() - 1) - local_score(dest);
                    } else if (model.has_edge(dest, source)) {
                        auto new_parents_source = model.get_parent_indices(source);
                        std::iter_swap(std::find(new_parents_source.begin(), new_parents_source.end(), dest), new_parents_source.end() - 1);
                        
                        new_parents_dest.push_back(source);

                        delta(source, dest) = Score::local_score(df, source, new_parents_source.begin(), new_parents_source.end() - 1) + 
                                              Score::local_score(df, dest, new_parents_dest.begin(), new_parents_dest.end()) 
                                              - local_score(source) - local_score(dest);
                        new_parents_dest.pop_back();
                    } else {
                        new_parents_dest.push_back(source);
                        delta(source, dest) = Score::local_score(df, dest, new_parents_dest) - local_score(dest);
                        new_parents_dest.pop_back();
                    }
                }
            }
        }
    }

    template<typename Model, typename Score>
    std::unique_ptr<Operator<Model>> ArcOperatorsType<Model, Score>::find_max(const Model& m) {

        auto delta_ptr = delta.data();
        auto max_index = std::max_element(delta_ptr, delta_ptr + model.num_nodes()*model.num_nodes() - 1) - delta_ptr;

        auto sorted_

        auto source = max_index % model.num_nodes();
        auto dest = max_index / model.num_nodes();

        // Check if can be applied (cycles!)
        if (model.has_edge(source, dest)) {
            return std::make_unique<RemoveEdge>(model.node(source), model.node(dest), delta(source, dest));
        } else if (model.has_edge(dest, source)) {
            return std::make_unique<FlipEdge>(model.node(dest), model.node(source), delta(dest, source));
        } else {
            return std::make_unique<AddEdge>(model.node(source), model.node(dest), delta(source, dest));
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