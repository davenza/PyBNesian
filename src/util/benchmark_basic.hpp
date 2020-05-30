#ifndef PGM_DATASET_BENCHMARK_BASIC_HPP
#define PGM_DATASET_BENCHMARK_BASIC_HPP


#include <chrono>

#define BENCHMARK_PRE_SCOPE(N)                                                                                                               \
            {                                                                                                                                \
                auto t1 = std::chrono::high_resolution_clock::now();                                                                         \
                for (auto _i = 0; _i < N; ++_i) {                                                                                               \

#define BENCHMARK_POST_SCOPE(N)                                                                                                              \
                }                                                                                                                            \
                auto t2 = std::chrono::high_resolution_clock::now();                                                                         \
                std::cout << "Time: " << (std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count() / N) << " ms" << std::endl;\
            }                                                                                                                                \



#define BENCHMARK_PRE(N, i)                                                                                                                 \
                auto t1 = std::chrono::high_resolution_clock::now();                                                                         \
                for (auto i = 0; i < N; ++i) {                                                                                         \      

#define BENCHMARK_POST(N)                                                                                                              \
                }                                                                                                                            \
                auto t2 = std::chrono::high_resolution_clock::now();                                                                         \
                std::cout << "Time: " << (std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count() / N) << " ms" << std::endl;\


#endif //PGM_DATASET_BENCHMARK_BASIC_HPP
