#include <iostream>
#include <arrow/api.h>
#include <pybind11/pybind11.h>
#include <assert.h>
#include <simd/simd.hpp>
#include <dataset/dataset.hpp>

using arrow::Array, arrow::NumericArray, arrow::DataType, arrow::Type;
using dataset::Column;

namespace linalg {
        
    double mean(Column col) {
        switch (col.data_type->id()) {
            case Type::DOUBLE:
                return simd::mean<arrow::DoubleType, double>(col.array);
            case Type::FLOAT:
                return simd::mean<arrow::FloatType, double>(col.array);
            case Type::HALF_FLOAT:
                return simd::mean<arrow::HalfFloatType, double>(col.array);
            case Type::INT64:
                return simd::mean<arrow::Int64Type, double>(col.array);
            case Type::UINT64:
                return simd::mean<arrow::UInt64Type, double>(col.array);
            case Type::INT32:
                return simd::mean<arrow::Int32Type, double>(col.array);
            case Type::UINT32:
                return simd::mean<arrow::UInt32Type, double>(col.array);
            case Type::INT16:
                return simd::mean<arrow::Int16Type, double>(col.array);
            case Type::UINT16:
                return simd::mean<arrow::UInt16Type, double>(col.array);
            case Type::INT8:
                return simd::mean<arrow::Int8Type, double>(col.array);
            case Type::UINT8:
                return simd::mean<arrow::UInt8Type, double>(col.array);
            default:
                throw pybind11::value_error("Only numeric data types are allowed in mean().");
        }
    }

    double var(Column col) {
        switch (col.data_type->id()) {
            case Type::DOUBLE:
                return simd::var<arrow::DoubleType, double>(col.array);
            case Type::FLOAT:
                return simd::var<arrow::FloatType, double>(col.array);
            case Type::HALF_FLOAT:
                return simd::var<arrow::HalfFloatType, double>(col.array);
            default:
                throw pybind11::value_error("Only floating point data is implemented in var().");
        }
    }

    double var(Column col, double mean) {
        switch (col.data_type->id()) {
            case Type::DOUBLE:
                return simd::var<arrow::DoubleType, double>(col.array, mean);
            case Type::FLOAT:
                return simd::var<arrow::FloatType, double>(col.array,
                                                           static_cast<typename arrow::FloatType::c_type>(mean));
            case Type::HALF_FLOAT:
                return simd::var<arrow::HalfFloatType, double>(col.array,
                                                           static_cast<typename arrow::HalfFloatType::c_type>(mean));
            default:
                throw pybind11::value_error("Only floating point data is implemented in var().");
        }
    }
}