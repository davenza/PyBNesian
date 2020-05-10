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
//        BENCHMARK(df.loc({"a", "c"});, 10000);
//        BENCHMARK(df.loc({0, 2});, 10000);

//        {
//            std::cout << "String - ones true" << std::endl;
//            auto m = df.loc(2.1);
//            std::cout << m->ToString() << std::endl;
//        }


        {
            std::cout << "String - ones false" << std::endl;
            auto m = df.to_eigen<false>(0);
            std::get<0>(m)(0) = 5;
            auto col = df->column(0);
            auto dwn_col = std::static_pointer_cast<typename arrow::TypeTraits<DoubleType>::ArrayType>(col);
            std::cout << "Arrow ptr: " << dwn_col->raw_values() << std::endl;
            std::cout << "Eigen ptr: " << std::get<0>(m).data() << std::endl;
            std::cout << std::get<0>(m) << std::endl;
        }

//        std::cout << "integer - ones true" << std::endl;
//        std::cout << df.to_eigen<true>(1);
//        std::cout << "integer - ones false" << std::endl;
//        std::cout << df.to_eigen<false>(1);

        if (evidence.empty()) {
//            beta[0] = linalg::mean(df.loc(variable));
//            variance = linalg::var(df.loc(variable), beta[0]);
        } else if (evidence.size() == 1) {
//            _fit_1parent(df.loc(variable), df.loc(evidence[0]));
        } else if (evidence.size() == 2) {
//            _fit_2parent(df.loc(variable), df.loc(evidence[0]), df.loc(evidence[1]));
        } else {
//            _fit_nparent(df.loc(variable), df.loc(evidence));
        }

//        auto column_array = df->GetColumnByName(this->variable);
//
//        auto fcontext = arrow::compute::FunctionContext();
//        auto cast_options = arrow::compute::CastOptions();
//
//        std::shared_ptr<arrow::Array> casted_array;
//        std::shared_ptr<arrow::DataType> to_type = std::make_shared<arrow::FloatType>();
//
//        std::cout << "Before cast: " << std::endl;
//        auto status = arrow::compute::Cast(&fcontext, *column_array, to_type, cast_options, &casted_array);
//
//        if(!status.ok()) {
//            std::cout << "Error: " << status.message() << std::endl;
//
//        } else {
//            std::cout << "Casted array: " << casted_array->ToString() << std::endl;
//            auto t1 = std::chrono::high_resolution_clock::now();
//            auto m = linalg::mean(casted_array, to_type);
//            auto t2 = std::chrono::high_resolution_clock::now();
//
//            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
//
//            std::cout << "Mean casted: " << std::setprecision(24) << m << std::endl;
//            std::cout << "Nanoseconds: " << duration << std::endl;
//
//        }


    }


//    void LinearGaussianCPD::_fit_1parent(Array_ptr y, Array_ptr regressor) {
//        auto length = y->length();
//        auto combined_bitmap = util::bit_util::combined_bitmap(y->null_bitmap(), regressor->null_bitmap(), length);
//
//        auto [mean_y, mean_reg, cov, var_reg] = [=]() {
//            if (combined_bitmap) {
//                auto mean_y = linalg::mean(y, combined_bitmap);
//                auto mean_reg = linalg::mean(regressor, combined_bitmap);
//                auto cov = linalg::covariance(y, regressor, mean_y, mean_reg, combined_bitmap);
//                auto var_reg = linalg::var(regressor, mean_reg, combined_bitmap);
//                return std::make_tuple(mean_y, mean_reg, cov, var_reg);
//            } else {
//                auto mean_y = linalg::mean(y);
//                auto mean_reg = linalg::mean(regressor);
//                auto cov = linalg::covariance(y, regressor, mean_y, mean_reg);
//                auto var_reg = linalg::var(regressor, mean_reg);
//                return std::make_tuple(mean_y, mean_reg, cov, var_reg);
//            }
//        }();
//
////        TODO: Check var_reg != 0.
//        auto b = cov / var_reg;
//        beta.push_back(mean_y - b*mean_reg);
//        beta.push_back(b);
//
//        std::vector<Array_ptr> columns;
//        columns.reserve(1);
//        columns.push_back(std::move(regressor));
//
//        auto fitted_values = linalg::linear_regression::fitted_values(beta, columns);
//
//        auto sse = linalg::sse(y, fitted_values);
//
//        auto null_fitted = fitted_values->null_count();
//        auto null_y = y->null_count();
//
//        auto n_instances = 0;
//        if (null_fitted == 0 && null_y == 0) {
//            n_instances = length;
//        } else if (null_fitted == 0 && null_y != 0) {
//            n_instances = length - null_y;
//        } else if (null_fitted != 0 && null_y == 0) {
//            n_instances = length - null_fitted;
//        } else {
//            n_instances = util::bit_util::non_null_count(combined_bitmap, length);
//        }
//
//        variance = sse / (n_instances - 2);
//
//        std::cout << "beta: [" << std::setprecision(24) << beta[0] << ", " << beta[1] << "]" << std::endl;
//        std::cout << "variance: " << std::setprecision(24) << variance << std::endl;
//    }
//
//    void LinearGaussianCPD::_fit_2parent(Array_ptr y, Array_ptr regressor1, Array_ptr regressor2) {
//        auto length = y->length();
//        std::vector<Array_ptr> columns;
//        columns.reserve(3);
//        columns.push_back(y);
//        columns.push_back(regressor1);
//        columns.push_back(regressor2);
//
//        auto combined_bitmap = util::bit_util::combined_bitmap(columns);
//
//        auto [mean_y, mean_reg1, mean_reg2, var_reg1, var_reg2, cov_xx, cov_yx1, cov_yx2] = [=]() {
//            if (combined_bitmap) {
//                auto mean_y = linalg::mean(y, combined_bitmap);
//                auto mean_reg1 = linalg::mean(regressor1, combined_bitmap);
//                auto mean_reg2 = linalg::mean(regressor2, combined_bitmap);
//                auto var_reg1 = linalg::var(regressor1, mean_reg1, combined_bitmap);
//                auto var_reg2 = linalg::var(regressor2, mean_reg2, combined_bitmap);
//                auto cov_xx = linalg::covariance(regressor1, regressor2, mean_reg1, mean_reg2, combined_bitmap);
//                auto cov_yx1 = linalg::covariance(y, regressor1, mean_y, mean_reg1, combined_bitmap);
//                auto cov_yx2 = linalg::covariance(y, regressor2, mean_y, mean_reg2, combined_bitmap);
//                return std::make_tuple(mean_y, mean_reg1, mean_reg2, var_reg1, var_reg2, cov_xx, cov_yx1, cov_yx2);
//            } else {
//                auto mean_y = linalg::mean(y);
//                auto mean_reg1 = linalg::mean(regressor1);
//                auto mean_reg2 = linalg::mean(regressor2);
//                auto var_reg1 = linalg::var(regressor1, mean_reg1);
//                auto var_reg2 = linalg::var(regressor2, mean_reg2);
//                auto cov_xx = linalg::covariance(regressor1, regressor2, mean_reg1, mean_reg2);
//                auto cov_yx1 = linalg::covariance(y, regressor1, mean_y, mean_reg1);
//                auto cov_yx2 = linalg::covariance(y, regressor2, mean_y, mean_reg2);
//                return std::make_tuple(mean_y, mean_reg1, mean_reg2, var_reg1, var_reg2, cov_xx, cov_yx1, cov_yx2);
//            }
//        }();
//
////        TODO: Check den not 0
//        auto den = var_reg1*var_reg2 - cov_xx*cov_xx;
//        auto b1 = (var_reg2 * cov_yx1 - cov_xx * cov_yx2) / den;
//        auto b2 = (cov_yx2 - b1 * cov_xx) / var_reg2;
//
//        beta.push_back(mean_y - b1*mean_reg1 - b2*mean_reg2);
//        beta.push_back(b1);
//        beta.push_back(b2);
//
//        std::vector<Array_ptr> regressors;
//        regressors.reserve(2);
//        regressors.push_back(std::move(regressor1));
//        regressors.push_back(std::move(regressor2));
//
//        auto fitted_values = linalg::linear_regression::fitted_values(beta, regressors);
//
//        auto sse = linalg::sse(y, fitted_values);
//
//        auto null_fitted = fitted_values->null_count();
//        auto null_y = y->null_count();
//
//        auto n_instances = 0;
//        if (null_fitted == 0 && null_y == 0) {
//            n_instances = length;
//        } else if (null_fitted == 0 && null_y != 0) {
//            n_instances = length - null_y;
//        } else if (null_fitted != 0 && null_y == 0) {
//            n_instances = length - null_fitted;
//        } else {
//            n_instances = arrow::internal::CountSetBits(combined_bitmap->data(), 0, length);
//        }
//
//        variance = sse / (n_instances - 3);
//
//        std::cout << "beta: [" << std::setprecision(24) << beta[0] << ", " << beta[1] << ", " << beta[2] << "]" << std::endl;
//        std::cout << "variance: " << std::setprecision(24) << variance << std::endl;
//    }
//
//
////    TODO: Research about gradient descent to solve linear regression.
//    template<typename ArrowType>
//    void _fit_nparent_typed(Array_ptr y, DataFrame evidence) {
//        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
//        using MatrixX = Matrix<typename ArrowType::c_type, Dynamic, Dynamic>;
//        using VectorX = Matrix<typename ArrowType::c_type, Dynamic, 1>;
//
//        auto length = y->length();
//        auto buffer_bitmap = util::bit_util::combined_bitmap(y->null_bitmap(), evidence.combined_bitmap(), length);
//
//        if (buffer_bitmap) {
//            auto combined_bitmap = buffer_bitmap->data();
//            auto non_null_count = util::bit_util::non_null_count(buffer_bitmap, length);
//
//            BENCHMARK(linalg::mean(y, buffer_bitmap);, 1000);
//
//            MatrixX ev_matrix(non_null_count, evidence->num_columns()+1);
//            auto matrix_ptr = ev_matrix.data();
//            std::cout << "matrix_ptr: " << matrix_ptr << std::endl;
//            std::fill_n(matrix_ptr, non_null_count, 1);
//
//            for (auto i = 0; i < evidence->num_columns(); ++i) {
//                auto col = evidence.loc(i);
//                auto dwn_col = std::static_pointer_cast<ArrayType>(col);
//
//                auto raw_values = dwn_col->raw_values();
//                auto k = 0;
//
//                auto offset = (i+1)*non_null_count;
//                for (auto j = 0; j < length; ++j) {
//                    if(arrow::BitUtil::GetBit(combined_bitmap, j))
//                        matrix_ptr[offset + k++] = raw_values[j];
//                }
//            }
//
//            auto dwn_y = std::static_pointer_cast<ArrayType>(y);
//            auto raw_values = dwn_y->raw_values();
//
//            double mean_m = 0;
//            std::cout << "My Missing mean:" << std::endl;
//            BENCHMARK(mean_m = linalg::mean(y, buffer_bitmap);, 1000);
////
//            std::cout << "Eigen Missing mean:" << std::endl;
//            VectorX y_vec(non_null_count);
//            auto vec_ptr = y_vec.data();
//            auto k = 0;
//
//            for (auto j = 0; j < length; ++j) {
//                if(arrow::BitUtil::GetBit(combined_bitmap, j))
//                    vec_ptr[k++] = raw_values[j];
//            }
//            BENCHMARK(
//
//            mean_m = y_vec.mean();
//            , 1000);
//
//            std::cout << "My mean: " << linalg::mean(y, buffer_bitmap) << std::endl;
//            std::cout << "Eigen mean: " << mean_m << std::endl;
//        } else {
//
//            MatrixX ev_matrix(evidence->num_rows(), evidence->num_columns()+1);
//
//            auto matrix_ptr = ev_matrix.data();
//            std::fill_n(matrix_ptr, length, 1);
//            for (auto i = 0; i < evidence->num_columns(); ++i) {
//                auto col = evidence.loc(i);
//                auto dwn_col = std::static_pointer_cast<ArrayType>(col);
//                std::memcpy(matrix_ptr + (i+1)*length, dwn_col->raw_values(), sizeof(typename ArrowType::c_type)*length);
//            }
//
//            auto dwn_y = std::static_pointer_cast<ArrayType>(y);
//            const Map<const VectorX> y_vec(dwn_y->raw_values(), length);
//
////            std::cout << "ev_matrix: " << ev_matrix << std::endl;
////            std::cout << "y_vec: " << y_vec << std::endl;
//            const auto N = 1000;
//            std::cout << "The solution using BDCSVD decomposition" << std::endl;
//            BENCHMARK(ev_matrix.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y_vec);, N);
//            std::cout << "The solution using householderQr decomposition" << std::endl;
//            BENCHMARK(ev_matrix.householderQr().solve(y_vec);, N);
//            std::cout << "The solution using colPivHouseholderQr decomposition" << std::endl;
//            BENCHMARK(ev_matrix.colPivHouseholderQr().solve(y_vec);, N);
//            std::cout << "The solution using fullPivHouseholderQr decomposition" << std::endl;
//            BENCHMARK(ev_matrix.fullPivHouseholderQr().solve(y_vec);, N);
//            std::cout << "The solution using normal equations" << std::endl;
//            BENCHMARK((ev_matrix.transpose() * ev_matrix).ldlt().solve(ev_matrix.transpose() * y_vec);, N);
//
//            std::cout << "The solution using BDCSVD decomposition is:\n"
//                      << std::setprecision(24) << ev_matrix.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y_vec) << std::endl;
//            std::cout << "The solution using householderQr decomposition is:\n"
//                      << std::setprecision(24) << ev_matrix.householderQr().solve(y_vec) << std::endl;
//            std::cout << "The solution using colPivHouseholderQr decomposition is:\n"
//                      << std::setprecision(24) << ev_matrix.colPivHouseholderQr().solve(y_vec) << std::endl;
//            std::cout << "The solution using fullPivHouseholderQr decomposition is:\n"
//                      << std::setprecision(24) << ev_matrix.fullPivHouseholderQr().solve(y_vec) << std::endl;
//            std::cout << "The solution using normal equations is:\n"
//                      << std::setprecision(24) << (ev_matrix.transpose() * ev_matrix).ldlt().solve(ev_matrix.transpose() * y_vec) << std::endl;
//
//
//            double mean_m = 0;
//            double var_m = 0;
//            std::cout << "Mean SIMD" << std::endl;
//            BENCHMARK(mean_m = linalg::mean(y);, 1);
//            std::cout << "Mean Eigen" << std::endl;
//            BENCHMARK(mean_m = y_vec.mean();, 1);
//
//            std::cout << "My mean: " << linalg::mean(y) << std::endl;
//            std::cout << "Eigen mean: " << y_vec.mean() << std::endl;
//
//            std::cout << "Var SIMD" << std::endl;
//            BENCHMARK(var_m = linalg::var(y);, 1);
//            std::cout << "Var Eigen" << std::endl;
//
//            BENCHMARK(
//                    auto m = y_vec.mean();
//                    auto diff = y_vec.array() - m;
//                    var_m = diff.matrix().squaredNorm() / (length - 1);
//                    , 1);
//
//            std::cout << "My var: " << linalg::var(y) << std::endl;
//            std::cout << "Eigen var: " << var_m << std::endl;
//
//        }
//    }
//
//    void LinearGaussianCPD::_fit_nparent(Array_ptr y, DataFrame evidence) {
//        auto dt_id = y->type_id();
//
//        switch(dt_id) {
//            case Type::DOUBLE:
//                _fit_nparent_typed<arrow::DoubleType>(y, evidence);
//                break;
//            case Type::FLOAT:
//                _fit_nparent_typed<arrow::FloatType>(y, evidence);
//                break;
////            case Type::HALF_FLOAT:
////                _fit_nparent_typed<arrow::HalfFloatType>(y, evidence);
////                break;
//            default:
//                throw pybind11::value_error("Only floating point data is implemented in fitted_values().");
//        }
//    }

}