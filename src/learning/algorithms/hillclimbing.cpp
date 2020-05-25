#include <iostream>
#include <learning/algorithms/hillclimbing.hpp>
#include <util/validate_dtype.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/operators/operators.hpp>

using namespace dataset;

using Eigen::VectorXd;
using models::GaussianNetwork, models::GaussianNetworkList;
using learning::scores::BIC;
using graph::arc_vector;
using learning::operators::ArcOperatorsType;


#include <random>
#include <util/benchmark_basic.hpp>
#include <queue>
using Eigen::MatrixXd;

namespace learning::algorithms {



    void benchmark_sort_vec(int nodes, int iterations, int sampling) {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> d{5,2};
        std::uniform_int_distribution in(0, nodes-1);
        
        MatrixXd scores(nodes, nodes);
        
        BENCHMARK_PRE_SCOPE(sampling)

        for (auto i = 0; i < nodes; ++i) {
            for (auto j = 0; j < nodes; ++j) {
                scores(i,j) = d(gen);
            }
        }

        auto scores_ptr = scores.data();
        std::vector<size_t> idx(nodes*nodes);
        std::iota(idx.begin(), idx.end(), 0);

        std::sort(idx.begin(), idx.end(), [&scores_ptr](auto i1, auto i2) {
            return scores_ptr[i1] >= scores_ptr[i2];
        });


        for (auto k = 0; k < iterations; ++k) {
            for (auto i = 0; i < nodes; ++i) {
                auto s = in(gen);
                auto dest = in(gen);
                scores(s,dest) = d(gen);
            }

            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&scores_ptr](auto i1, auto i2) {
                return scores_ptr[i1] >= scores_ptr[i2];
            });
        }

        BENCHMARK_POST_SCOPE(sampling)
    }

    void benchmark_sort_set(int nodes, int iterations, int sampling) {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> d{5,2};
        std::uniform_int_distribution in(0, nodes-1);
        
        std::set<double, std::greater<double>> scores;
        
        BENCHMARK_PRE_SCOPE(sampling)

        for (auto i = 0; i < nodes; ++i) {
            for (auto j = 0; j < nodes; ++j) {
                scores.insert(d(gen));
            }
        }

        for (auto k = 0; k < iterations; ++k) {
            auto b = scores.begin();
            std::advance(b, nodes);
            scores.erase(b);
            for (auto i = 0; i < nodes; ++i) {
                auto s = in(gen);
                auto dest = in(gen);
                scores.insert(d(gen));
            }
        }

        BENCHMARK_POST_SCOPE(sampling)
    }

    void benchmark_sort_priority(int nodes, int iterations, int sampling) {
        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> d{5,2};
        std::uniform_int_distribution in(0, nodes-1);
        std::vector<double> sorted_double(nodes*nodes);
        
        std::priority_queue<double, std::vector<double>, std::greater<double>> scores;
        
        BENCHMARK_PRE_SCOPE(sampling)
        for (auto i = 0; i < nodes; ++i) {
            for (auto j = 0; j < nodes; ++j) {
                scores.push(d(gen));
            }
        }

        for (auto i = 0; i < nodes*nodes; ++i) {
            sorted_double[i] = scores.top();
            scores.pop();
        }

        for (auto i = 0; i < nodes*nodes; ++i) {
            scores.push(sorted_double[i]);
        }


        for (auto k = 0; k < iterations; ++k) {
            for (auto i = 0; i < nodes; ++i) {
                sorted_double[i] = scores.top();
                scores.pop();
            }

            for (auto i = 0; i < nodes; ++i) {
                auto s = in(gen);
                auto dest = in(gen);
                scores.push(d(gen));
            }
        }
        BENCHMARK_POST_SCOPE(sampling)
    }


    // TODO: Include start model.
    void estimate(py::handle data, std::string str_score, 
                  std::vector<py::tuple> blacklist, std::vector<py::tuple> whitelist, 
                  int max_indegree, double epsilon) {


        auto rb = dataset::to_record_batch(data);
        auto df = DataFrame(rb);

        auto blacklist_cpp = util::check_edge_list(df, blacklist);
        auto whitelist_cpp = util::check_edge_list(df, whitelist);

        auto nodes = df.column_names();
        auto nnodes = nodes.size();

        GreedyHillClimbing<GaussianNetwork> hc;

        GaussianNetwork gbn = (whitelist_cpp.size() > 0) ? GaussianNetwork(nodes, whitelist_cpp) :
                                                           GaussianNetwork(nodes);

        gbn.print();

        std::cout << "path a -> b " << gbn.has_path("a", "b") << std::endl;;
        std::cout << "path a -> c " << gbn.has_path("a", "c") << std::endl;;
        std::cout << "path a -> d " << gbn.has_path("a", "d") << std::endl;;
        
        std::cout << "path b -> a " << gbn.has_path("b", "a") << std::endl;;
        std::cout << "path b -> c " << gbn.has_path("b", "c") << std::endl;;
        std::cout << "path b -> d " << gbn.has_path("b", "d") << std::endl;;
        
        std::cout << "path c -> a " << gbn.has_path("c", "a") << std::endl;;
        std::cout << "path c -> b " << gbn.has_path("c", "b") << std::endl;;
        std::cout << "path c -> d " << gbn.has_path("c", "d") << std::endl;;
        
        std::cout << "path d -> a " << gbn.has_path("d", "a") << std::endl;;
        std::cout << "path d -> b " << gbn.has_path("d", "b") << std::endl;;
        std::cout << "path d -> c " << gbn.has_path("d", "c") << std::endl;;
        if (str_score == "bic") {

            ArcOperatorsType<GaussianNetwork, BIC<GaussianNetwork>> arc_op(df, gbn, whitelist_cpp, blacklist_cpp);

            hc.estimate(df, arc_op, blacklist_cpp, whitelist_cpp, max_indegree, epsilon, gbn);
        }
         else {
            throw std::invalid_argument("Wrong score \"" + str_score + "\". Currently supported scores: \"bic\".");
        }
    }
    
    template<typename Model>
    template<typename Operators>
    void GreedyHillClimbing<Model>::estimate(const DataFrame& df, 
                                              Operators op,
                                              arc_vector blacklist, 
                                              arc_vector whitelist, 
                                              int max_indegree, 
                                              double epsilon,
                                              const Model& start) {


        Model::requires(df);

        op.cache_scores();


        // Model current_model = start;

        // while(true) {

        //     auto best_op = op_pool.find_max(current_model);

        //     // if (best_op->delta() < epsilon) {
        //     //     break;
        //     // }

        //     // best_op.apply_operator(current_model);
        // }




    }


}