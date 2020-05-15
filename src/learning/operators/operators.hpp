#include <Eigen/Dense>
#include <models/BayesianNetwork.hpp>


using Eigen::MatrixXd;

using models::BayesianNetwork, models::BayesianNetworkType;

namespace learning::operators {

    template<typename Model>
    class Operator {
    public:

        virtual void apply_operator(Model& m) = 0;

    private:
        double delta;
    };

    template<typename Model>
    class AddArc : Operator<Model> {

        AddArc(typename Model::node_descriptor source, 
               typename Model::node_descriptor dest) : source(source), dest(dest) {}
        
        void apply_operator(Model& m) override {
            m.add_arc(source, dest);
        }

    private:
        typename Model::node_descriptor source;
        typename Model::node_descriptor dest;
    };

    template<typename Model>
    class RemoveArc : Operator<Model> {

        RemoveArc(typename Model::node_descriptor source, 
                  typename Model::node_descriptor dest) : source(source), dest(dest) {}
        
        void apply_operator(Model& m) override {
            m.remove_arc(source, dest);
        }

    private:
        typename Model::node_descriptor source;
        typename Model::node_descriptor dest;
    };

    template<typename Model>
    class FlipArc : Operator<Model> {

        FlipArc(typename Model::node_descriptor source, 
                  typename Model::node_descriptor dest) : source(source), dest(dest) {}
        
        void apply_operator(Model& m) override {
            m.flip_arc(source, dest);
        }

    private:
        typename Model::node_descriptor source;
        typename Model::node_descriptor dest;
    };


    template<typename Model>
    class OperatorType {
    public:

        virtual void cache_scores(Model& m) = 0;

        virtual void update_scores(Model& m, std::unique_ptr<Operator<Model>> new_op) = 0;

        virtual std::unique_ptr<Operator<Model>> find_max(Model& m) = 0;
    };


    template<typename Model, typename Score>
    class ArcOperatorsType : OperatorType<Model>{

        using model = Model;

        ArcOperatorsType(int n_nodes) : scores(n_nodes, n_nodes) {}

        void cache_scores(Model& m) override {

        }

        void update_scores(Model& m, std::unique_ptr<Operator<Model>> new_op) override {
            
        }

        std::unique_ptr<Operator<Model>> find_max(Model& m) override {
            return nullptr;
        }

    private:
        MatrixXd scores;
    };


    template<typename Model, typename Score>
    class OperatorPool {

        OperatorPool(std::vector<OperatorType<Model>> ops) : operator_types(ops) {}

    private:
        std::vector<OperatorType<Model>> operator_types;
    };


    template<typename Model, typename Score>
    struct default_operator {

    };

    template<BayesianNetworkType bn_type, typename Score>
    struct default_operator<BayesianNetwork<bn_type>, Score> {
        using default_operator_t = ArcOperatorsType<BayesianNetwork<bn_type>, Score>;
    };
    
    template<typename Model, typename Score>
    class DefaultOperatorPool {

        DefaultOperatorPool(int n_nodes) : operator_type(n_nodes) {}

    private:
        typename default_operator<Model, Score>::type operator_type;
    };



}