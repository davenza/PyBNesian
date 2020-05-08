#include <iostream>
#include <arrow/api.h>
#include <pybind11/pybind11.h>
#include <assert.h>
#include <dataset/dataset.hpp>
#include <Eigen/Dense>

using arrow::Array, arrow::NumericArray, arrow::DataType, arrow::Type;

typedef std::shared_ptr<arrow::Array> Array_ptr;
typedef std::shared_ptr<arrow::Buffer> Buffer_ptr;

using Eigen::Matrix, Eigen::Dynamic;

namespace linalg {

    namespace internals {

        template<typename ArrowType>
        using VectorType = Matrix<typename ArrowType::c_type, Dynamic, 1>;
        double mean(Array_ptr col) {

        }



    }


    double mean(Array_ptr col) {
        switch (col->type_id()) {
            case Type::DOUBLE:
//                return simd::mean<arrow::DoubleType, double>(col);
            case Type::FLOAT:
//                return simd::mean<arrow::FloatType, double>(col);
            default:
                throw pybind11::value_error("Only numeric data types are allowed in mean().");
        }
    }
//
//    double var(Array_ptr col) {
//        switch (col->type_id()) {
//            case Type::DOUBLE:
//                return simd::var<arrow::DoubleType, double>(col);
//            case Type::FLOAT:
//                return simd::var<arrow::FloatType, double>(col);
//            default:
//                throw pybind11::value_error("Only floating point data is implemented in var().");
//        }
//    }
//
//    double var(Array_ptr col, double mean) {
//        switch (col->type_id()) {
//            case Type::DOUBLE:
//                return simd::var<arrow::DoubleType, double>(col, mean);
//            case Type::FLOAT:
//                return simd::var<arrow::FloatType, double>(col,
//                                                           static_cast<typename arrow::FloatType::c_type>(mean));
//            default:
//                throw pybind11::value_error("Only floating point data is implemented in var().");
//        }
//    }
//
//    double covariance_unsafe(Array_ptr col1, Array_ptr col2, double mean1, double mean2) {
//        switch (col1->type_id()) {
//            case Type::DOUBLE:
//                return simd::covariance<arrow::DoubleType, double>(col1, col2, mean1, mean2);
//            case Type::FLOAT:
//                return simd::covariance<arrow::FloatType, double>(col1, col2,
//                                                                  static_cast<typename arrow::FloatType::c_type>(mean1),
//                                                                  static_cast<typename arrow::FloatType::c_type>(mean2));
//            default:
//                throw pybind11::value_error("Only floating point data is implemented in var().");
//        }
//    }
//
//
//    double covariance(Array_ptr col1, Array_ptr col2, double mean1, double mean2) {
//        if (col1->type_id() != col2->type_id()) {
//            throw pybind11::value_error("Data type for both columns should be the same in covariance()");
//        }
//
//        return covariance_unsafe(col1, col2, mean1, mean2);
//    }
//
//    double sse_unsafe(Array_ptr truth, Array_ptr predicted) {
//        switch (truth->type_id()) {
//            case Type::DOUBLE:
//                return simd::sse<arrow::DoubleType, double>(truth, predicted);
//            case Type::FLOAT:
//                return simd::sse<arrow::FloatType, double>(truth, predicted);
//            default:
//                throw pybind11::value_error("Only floating point data is implemented in sse().");
//        }
//    }
//
//    double sse(Array_ptr truth, Array_ptr predicted) {
//        if (truth->type_id() != predicted->type_id()) {
//            throw pybind11::value_error("Data type for both columns should be the same in sse()");
//        }
//
//        return sse_unsafe(truth, predicted);
//    }
//}
//
//namespace linalg::linear_regression {
//
//    template<typename ArrowType>
//    Array_ptr fitted_values_typed(std::vector<double> beta, std::vector<Array_ptr> columns) {
//
//    }
//
////    TODO: Document: This function only accepts columns non empty
//    Array_ptr fitted_values(std::vector<double> beta, std::vector<Array_ptr> columns) {
//        auto dt_id = columns[0]->type_id();
//
//        switch(dt_id) {
//            case Type::DOUBLE:
//                return fitted_values_typed<arrow::DoubleType>(beta, columns);
//            case Type::FLOAT:
//                return fitted_values_typed<arrow::FloatType>(beta, columns);
//            default:
//                throw pybind11::value_error("Only floating point data is implemented in fitted_values().");
//        }
//    }
}