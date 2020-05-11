#include <iostream>
#include <Python.h>
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>
//#include <linalg/linalg.hpp>
#include <util/bit_util.hpp>
#include <Eigen/Dense>

#include <arrow/compute/kernels/cast.h>
#include <arrow/compute/context.h>
#include <chrono>
#include <iomanip>

namespace py = pybind11;
namespace pyarrow = arrow::py;

using arrow::Type;
using Eigen::Matrix, Eigen::Dynamic, Eigen::Map;

using dataset::DataFrame;

typedef std::shared_ptr<arrow::Array> Array_ptr;


#define BENCHMARK(code, N)                                                                                                                   \
            {                                                                                                                                \
                auto t1 = std::chrono::high_resolution_clock::now();                                                                         \
                for (auto i = 0; i < N; ++i) {                                                                                               \
                    code                                                                                                                     \
                }                                                                                                                            \
                auto t2 = std::chrono::high_resolution_clock::now();                                                                         \
                std::cout << "Time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count() / N) << " ns" << std::endl;\
            }                                                                                                                                \

#define BENCHMARK_CODE(code, N)                                                                                                                   \
            {                                                                                                                                \
                auto t1 = std::chrono::high_resolution_clock::now();                                                                         \
                for (auto i = 0; i < N; ++i) {                                                                                               \
                    code                                                                                                                     \
                }                                                                                                                            \
                auto t2 = std::chrono::high_resolution_clock::now();                                                                         \
                std::cout << "Time (" << #code << "): " <<                                                                                   \
                        (std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count() / N) << " ns" << std::endl;                 \
            }                                                                                                                                \


namespace factors::continuous {


    LinearGaussianCPD::LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence) :
    variable(variable),
    evidence(evidence)
    {
        this->beta.reserve(evidence.size() + 1);
    };

    LinearGaussianCPD::LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence,
            const std::vector<double> beta) :
    variable(variable),
    evidence(evidence),
    beta(beta)
//    TODO: Error checking: Length of vectors
    {};

    void LinearGaussianCPD::fit(py::handle dataset) {
        auto rb = dataset::to_record_batch(dataset);
        auto df = DataFrame(rb);

        auto schema = df->schema();

        auto var_field = schema->GetFieldByName(variable);

        if (var_field) {
            auto type = var_field->type();
            auto type_id = type->id();

            if (type_id != Type::DOUBLE && type_id != Type::FLOAT) {
                throw py::value_error("Wrong data type to fit the linear regression. (double) or (float) data is expected.");
            }

            for (auto &ev : evidence) {
                auto ev_field = schema->GetFieldByName(ev);
                if (!ev_field) {
                    throw py::value_error("Variable \"" + ev + "\" not found in dataset.");
                } else if (!type->Equals(ev_field->type())){
                    throw py::value_error("\"" + variable + "\" has different data type than \"" + ev + "\"");
                }
            }
        } else {
            throw py::value_error("Variable \"" + variable + "\" not found in dataset.");
        }

        _fit(df);
    }

    void LinearGaussianCPD::_fit(DataFrame df) {

        if (evidence.empty()) {
            BENCHMARK(
                    auto eigen = df.to_eigen<false>(variable);

                    std::visit([this](auto &&arg) {
                        auto mean = arg->mean();
                        auto var = (arg->array() - mean).matrix().squaredNorm();
                        beta.push_back(static_cast<double>(mean));
                        variance = static_cast<double>(var) / (arg->rows() - 1);
                    }, eigen);, 1000)

            std::cout << "beta: " << std::setprecision(24) << beta[0] << std::endl;
            std::cout << "variance: " << std::setprecision(24) << variance << std::endl;

        } else if (evidence.size() == 1) {
            BENCHMARK(
            auto comb_bitmap = df.combined_bitmap({variable, evidence[0]});
//
            auto var = df.to_eigen<false>(variable, comb_bitmap);
            auto ev = df.to_eigen<false>(evidence[0], comb_bitmap);

//            TODO: Downcast instead of visit.
            std::visit(util::overloaded_same_type_and_cols {
                [this](auto &&v, auto &&e) {
                    auto mv = v->mean();
                    auto me = e->mean();

                    auto d0 = (v->array() - mv);
                    auto d1 = (e->array() - me);
                    auto var = d1.matrix().squaredNorm() / (v->rows() - 1);
                    auto cov = (d0 * d1).sum() / (v->rows() - 1);

                    auto b = cov / var;
                    auto a = mv - b*me;
                    auto vari = (d0 - b*d1).matrix().squaredNorm() / (v->rows() - 2);

                    beta.push_back(a);
                    beta.push_back(b);
                    variance = vari;
                }
            }, var, ev);
            , 1000)

//            auto comb_bitmap = df.combined_bitmap({variable, evidence[0]});
//            auto var = df.to_eigen<false>(variable, comb_bitmap);
//            auto ev = df.to_eigen<false>(evidence[0], comb_bitmap);
//            auto s = df.to_eigen<false>(evidence[0], comb_bitmap);


//            std::visit(util::overloaded_same_type_and_cols {
//                    [this](auto &&v, auto &&e, auto &&p) {
//                        auto a = (e->array() * p->array()).sum();
//                        std::cout << "Testing with 3 parameters "  << a << std::endl;
//                    }
//            }, var, ev, fl);


//            BENCHMARK(
//            auto eigen = df.to_eigen<false>({variable, evidence[0]});
////
//            std::visit([this](auto&& arg) {
//                auto m0 = arg->col(0).mean();
//                auto m1 = arg->col(1).mean();
//                auto diff0 = (arg->col(0).array() - m0);
//                auto diff1 = (arg->col(1).array() - m1);
//                auto var = diff1.matrix().squaredNorm() / (arg->rows() - 1);
//                auto cov = (diff0 * diff1).sum() / (arg->rows() - 1);
//
//                auto b = cov / var;
//                auto a = m0 - b*m1;
//                beta.push_back(a);
//                beta.push_back(b);
//
//                variance = (diff0 - b*diff1).matrix().squaredNorm() / (arg->rows() - 2);
//            }, eigen);
//            , 1000)

            std::cout << "beta: [" << std::setprecision(24) << beta[0] << ", " << beta[1] << "]" << std::endl;
            std::cout << "variance: " << std::setprecision(24) << variance << std::endl;
        } else if (evidence.size() == 2) {
//            _fit_2parent(df.loc(variable), df.loc(evidence[0]), df.loc(evidence[1]));
        } else {
//            _fit_nparent(df.loc(variable), df.loc(evidence));
        }
    }





}