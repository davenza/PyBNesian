#include <iostream>
#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_dtype.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/operators/operators.hpp>

using namespace dataset;

using Eigen::VectorXd, Eigen::MatrixXd;;
using models::GaussianNetwork, models::GaussianNetwork_M, models::GaussianNetwork_L;
using learning::scores::BIC;
// learning::scores::CVLikelihood;
using graph::arc_vector;
using learning::operators::ArcOperatorsType;

// #include <util/benchmark_basic.hpp>

namespace learning::algorithms {

    // TODO: Include start model.
    void hc(py::handle data, std::string str_score, 
                  std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, 
                  int max_indegree, double epsilon) {

        auto rb = dataset::to_record_batch(data);
        auto df = DataFrame(rb);

        auto blacklist_cpp = util::check_edge_list(df, blacklist);
        auto whitelist_cpp = util::check_edge_list(df, whitelist);

        auto nodes = df.column_names();

        // GreedyHillClimbing<GaussianNetwork> hc;
        GreedyHillClimbing<GaussianNetwork_M> hc;

        GaussianNetwork_M gbn = (whitelist_cpp.size() > 0) ? GaussianNetwork_M(nodes, whitelist_cpp) :
                                                           GaussianNetwork_M(nodes);

        // GaussianNetwork_L gbn = (whitelist_cpp.size() > 0) ? GaussianNetwork_L(nodes, whitelist_cpp) :
        //                                                    GaussianNetwork_L(nodes);

        if (str_score == "bic") {
            BIC bic(df);
            // ArcOperatorsType<GaussianNetwork, BIC<GaussianNetwork>> arc_op(df, gbn, whitelist_cpp, blacklist_cpp, max_indegree);
            ArcOperatorsType arc_op(bic, gbn, whitelist_cpp, blacklist_cpp, max_indegree);
            // BENCHMARK_PRE_SCOPE(10)
            hc.estimate(arc_op, epsilon, gbn);
            // BENCHMARK_POST_SCOPE(10)
        } 
        // else if (str_score == "predictive-l") {
        //     CVLikelihood cv(df, 10);

        //     ArcOperatorsType arc_op(cv, gbn, whitelist_cpp, blacklist_cpp, max_indegree);
        //     hc.estimate(arc_op, epsilon, gbn);
        // }
         else {
            throw std::invalid_argument("Wrong score \"" + str_score + "\". Currently supported scores: \"bic\".");
        }
    }
    
    template<typename Model>
    template<typename Operators>
    void GreedyHillClimbing<Model>::estimate(Operators& op,
                                             double epsilon,
                                             const Model& start) {


        // Model::requires(df);

        auto current_model = start;
        op.cache_scores(current_model);
        
        while(true) {

            auto best_op = op.find_max(current_model);
            if (!best_op || best_op->delta() <= epsilon) {
                break;
            }

            best_op->apply(current_model, op);
            // std::cout << "New op" << std::endl;
        }

        // std::cout << "Final score: " << op.score() << std::endl;
    }


}