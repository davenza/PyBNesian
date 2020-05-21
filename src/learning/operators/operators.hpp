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
    class AddArc : public Operator<Model> {

        AddArc(typename Model::node_descriptor source, 
               typename Model::node_descriptor dest,
               double delta) : m_source(source), m_dest(dest), Operator<Model>(delta) {}
        
        void apply_operator(Model& m) override {
            m.add_arc(m_source, m_dest);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model>
    class RemoveArc : public Operator<Model> {

        RemoveArc(typename Model::node_descriptor source, 
                  typename Model::node_descriptor dest,
                  double delta) : m_source(source), m_dest(dest), Operator<Model>(delta) {}
        
        void apply_operator(Model& m) override {
            m.remove_arc(m_source, m_dest);
        }

    private:
        typename Model::node_descriptor m_source;
        typename Model::node_descriptor m_dest;
    };

    template<typename Model>
    class FlipArc : public Operator<Model> {

        FlipArc(typename Model::node_descriptor source, 
                  typename Model::node_descriptor dest,
                  double delta) : m_source(source), m_dest(dest), Operator<Model>(delta){}
        
        void apply_operator(Model& m) override {
            m.flip_arc(m_source, m_dest);
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

        std::cout << "valid_ops:" << std::endl;
        std::cout << valid_op << std::endl;

        for (int i = 0; i < model.num_nodes(); ++i) {
            auto parents = model.get_parent_indices(i);
            local_score(i) = Score::local_score(df, i, parents);
        }

        std::cout << "Local scores:" << std::endl;
        std::cout << local_score << std::endl; 

        for (auto dest = 0; dest < model.num_nodes(); ++dest) {
            std::vector<int> new_parents_dest = model.get_parent_indices(dest);
            
            for (auto source = 0; source < model.num_nodes(); ++source) {
                if(valid_op(source, dest)) {

                    if (model.has_edge(source, dest)) {
                        new_parents_dest.erase(std::remove(new_parents_dest.begin(), new_parents_dest.end(), source));
                        delta(source, dest) = Score::local_score(df, dest, new_parents_dest) - local_score(dest);
                        new_parents_dest.push_back(source);
                    } else if (model.has_edge(dest, source)) {
                        auto new_parents_source = model.get_parent_indices(source);
                        new_parents_source.erase(std::remove(new_parents_source.begin(), new_parents_source.end(), dest));
                        new_parents_dest.push_back(source);

                        delta(source, dest) = Score::local_score(df, source, new_parents_source) + 
                                              Score::local_score(df, dest, new_parents_dest) 
                                              - local_score(source) - local_score(dest);

                        new_parents_dest.erase(std::remove(new_parents_dest.begin(), new_parents_dest.end(), source));
                    } else {
                        new_parents_dest.push_back(source);
                        delta(source, dest) = Score::local_score(df, dest, new_parents_dest) - local_score(dest);
                        new_parents_dest.erase(std::remove(new_parents_dest.begin(), new_parents_dest.end(), source));
                    }
                }
            }
        }

        std::cout << "delta scores:" << std::endl;
        std::cout << delta << std::endl; 
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