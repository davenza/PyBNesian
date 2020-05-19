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
    {}



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