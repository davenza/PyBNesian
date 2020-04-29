#include <iostream>
#include <arrow/api.h>
#include <pybind11/pybind11.h>
#include <assert.h>
#include <simd/simd.hpp>

using arrow::Array, arrow::NumericArray, arrow::DataType, arrow::Type;

namespace linalg {

    namespace internal {

        template<typename ArrowType>
        double mean(std::shared_ptr<Array> ar) {
            return simd::mean<ArrowType, double>(ar);
        }

        template<typename ArrowType>
        double var(std::shared_ptr<Array> ar) {
            return simd::var<ArrowType, double>(ar);
        }
    }


    double mean(std::shared_ptr<Array> ar, std::shared_ptr<DataType> dt) {
        switch (dt->id()) {
            case Type::DOUBLE:
                return internal::mean<arrow::DoubleType>(ar);
            case Type::FLOAT:
                return internal::mean<arrow::FloatType>(ar);
            case Type::HALF_FLOAT:
                return internal::mean<arrow::HalfFloatType>(ar);
            case Type::INT64:
                return internal::mean<arrow::Int64Type>(ar);
            case Type::UINT64:
                return internal::mean<arrow::UInt64Type>(ar);
            case Type::INT32:
                return internal::mean<arrow::Int32Type>(ar);
            case Type::UINT32:
                return internal::mean<arrow::UInt32Type>(ar);
            case Type::INT16:
                return internal::mean<arrow::Int16Type>(ar);
            case Type::UINT16:
                return internal::mean<arrow::UInt16Type>(ar);
            case Type::INT8:
                return internal::mean<arrow::Int8Type>(ar);
            case Type::UINT8:
                return internal::mean<arrow::UInt8Type>(ar);
            default:
                throw pybind11::value_error("Only numeric data types are allowed in mean().");
        }
    }

    double var(std::shared_ptr<Array> ar, std::shared_ptr<DataType> dt) {
        switch (dt->id()) {
            case Type::DOUBLE:
                return internal::var<arrow::DoubleType>(ar);
            case Type::FLOAT:
                return internal::var<arrow::FloatType>(ar);
            case Type::HALF_FLOAT:
                return internal::var<arrow::HalfFloatType>(ar);
            default:
                throw pybind11::value_error("Only floating point data is implemented in var().");
        }
    }
}