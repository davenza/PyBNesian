#include <iostream>
#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_dtype.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/operators/operators.hpp>

using namespace dataset;

using Eigen::VectorXd, Eigen::MatrixXd;;
using models::GaussianNetwork, models::GaussianNetworkList;
using learning::scores::BIC;
using graph::arc_vector;
using learning::operators::ArcOperatorsType;


namespace learning::algorithms {

    // TODO: Include start model.
    void estimate(py::handle data, std::string str_score, 
                  std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, 
                  int max_indegree, double epsilon) {


        auto rb = dataset::to_record_batch(data);
        auto df = DataFrame(rb);

        auto blacklist_cpp = util::check_edge_list(df, blacklist);
        auto whitelist_cpp = util::check_edge_list(df, whitelist);

        auto nodes = df.column_names();

        GreedyHillClimbing<GaussianNetwork> hc;

        GaussianNetwork gbn = (whitelist_cpp.size() > 0) ? GaussianNetwork(nodes, whitelist_cpp) :
                                                           GaussianNetwork(nodes);

        gbn.print();


        if (str_score == "bic") {
            ArcOperatorsType<GaussianNetwork, BIC<GaussianNetwork>> arc_op(df, gbn, whitelist_cpp, blacklist_cpp, max_indegree);
            hc.estimate(df, arc_op, epsilon, gbn);
        }
         else {
            throw std::invalid_argument("Wrong score \"" + str_score + "\". Currently supported scores: \"bic\".");
        }
    }
    
    template<typename Model>
    template<typename Operators>
    void GreedyHillClimbing<Model>::estimate(const DataFrame& df, 
                                              Operators& op,
                                              double epsilon,
                                              const Model& start) {


        Model::requires(df);

        auto current_model = start;
        op.cache_scores(current_model);
        
        while(true) {

            auto best_op = op.find_max(current_model);
            if (!best_op || best_op->delta() <= epsilon) {
                break;
            }

            best_op->apply(current_model, op);
        }

        std::cout << "start:" << std::endl;
        start.print();
        std::cout << "final_model:" << std::endl;
        current_model.print();
    }


}