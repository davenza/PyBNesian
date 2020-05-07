#include <iostream>
#include <arrow/api.h>
#include <pybind11/pybind11.h>
#include <assert.h>
#include <simd/simd.hpp>
#include <dataset/dataset.hpp>

using arrow::Array, arrow::NumericArray, arrow::DataType, arrow::Type;

typedef std::shared_ptr<arrow::Array> Array_ptr;
typedef std::shared_ptr<arrow::Buffer> Buffer_ptr;

namespace linalg {

    double mean(Array_ptr col) {
        switch (col->type_id()) {
            case Type::DOUBLE:
                return simd::mean<arrow::DoubleType, double>(col);
            case Type::FLOAT:
                return simd::mean<arrow::FloatType, double>(col);
            case Type::HALF_FLOAT:
                return simd::mean<arrow::HalfFloatType, double>(col);
            case Type::INT64:
                return simd::mean<arrow::Int64Type, double>(col);
            case Type::UINT64:
                return simd::mean<arrow::UInt64Type, double>(col);
            case Type::INT32:
                return simd::mean<arrow::Int32Type, double>(col);
            case Type::UINT32:
                return simd::mean<arrow::UInt32Type, double>(col);
            case Type::INT16:
                return simd::mean<arrow::Int16Type, double>(col);
            case Type::UINT16:
                return simd::mean<arrow::UInt16Type, double>(col);
            case Type::INT8:
                return simd::mean<arrow::Int8Type, double>(col);
            case Type::UINT8:
                return simd::mean<arrow::UInt8Type, double>(col);
            default:
                throw pybind11::value_error("Only numeric data types are allowed in mean().");
        }
    }

    double mean(Array_ptr col, Buffer_ptr bitmap) {
        switch (col->type_id()) {
            case Type::DOUBLE:
                return simd::mean<arrow::DoubleType, double>(col, bitmap);
            case Type::FLOAT:
                return simd::mean<arrow::FloatType, double>(col, bitmap);
            case Type::HALF_FLOAT:
                return simd::mean<arrow::HalfFloatType, double>(col, bitmap);
            case Type::INT64:
                return simd::mean<arrow::Int64Type, double>(col, bitmap);
            case Type::UINT64:
                return simd::mean<arrow::UInt64Type, double>(col, bitmap);
            case Type::INT32:
                return simd::mean<arrow::Int32Type, double>(col, bitmap);
            case Type::UINT32:
                return simd::mean<arrow::UInt32Type, double>(col, bitmap);
            case Type::INT16:
                return simd::mean<arrow::Int16Type, double>(col, bitmap);
            case Type::UINT16:
                return simd::mean<arrow::UInt16Type, double>(col, bitmap);
            case Type::INT8:
                return simd::mean<arrow::Int8Type, double>(col, bitmap);
            case Type::UINT8:
                return simd::mean<arrow::UInt8Type, double>(col, bitmap);
            default:
                throw pybind11::value_error("Only numeric data types are allowed in mean().");
        }
    }

    double var(Array_ptr col) {
        switch (col->type_id()) {
            case Type::DOUBLE:
                return simd::var<arrow::DoubleType, double>(col);
            case Type::FLOAT:
                return simd::var<arrow::FloatType, double>(col);
            case Type::HALF_FLOAT:
                return simd::var<arrow::HalfFloatType, double>(col);
            default:
                throw pybind11::value_error("Only floating point data is implemented in var().");
        }
    }

    double var(Array_ptr col, Buffer_ptr bitmap) {
        switch (col->type_id()) {
            case Type::DOUBLE:
                return simd::var<arrow::DoubleType, double>(col, bitmap);
            case Type::FLOAT:
                return simd::var<arrow::FloatType, double>(col, bitmap);
            case Type::HALF_FLOAT:
                return simd::var<arrow::HalfFloatType, double>(col, bitmap);
            default:
                throw pybind11::value_error("Only floating point data is implemented in var().");
        }
    }

    double var(Array_ptr col, double mean) {
        switch (col->type_id()) {
            case Type::DOUBLE:
                return simd::var<arrow::DoubleType, double>(col, mean);
            case Type::FLOAT:
                return simd::var<arrow::FloatType, double>(col,
                                                           static_cast<typename arrow::FloatType::c_type>(mean));
            case Type::HALF_FLOAT:
                return simd::var<arrow::HalfFloatType, double>(col,
                                                           static_cast<typename arrow::HalfFloatType::c_type>(mean));
            default:
                throw pybind11::value_error("Only floating point data is implemented in var().");
        }
    }

    double var(Array_ptr col, double mean, Buffer_ptr bitmap) {
        switch (col->type_id()) {
            case Type::DOUBLE:
                return simd::var<arrow::DoubleType, double>(col, mean, bitmap);
            case Type::FLOAT:
                return simd::var<arrow::FloatType, double>(col,
                                                           static_cast<typename arrow::FloatType::c_type>(mean), bitmap);
            case Type::HALF_FLOAT:
                return simd::var<arrow::HalfFloatType, double>(col,
                                                               static_cast<typename arrow::HalfFloatType::c_type>(mean), bitmap);
            default:
                throw pybind11::value_error("Only floating point data is implemented in var().");
        }
    }

    double covariance_unsafe(Array_ptr col1, Array_ptr col2, double mean1, double mean2) {
        switch (col1->type_id()) {
            case Type::DOUBLE:
                return simd::covariance<arrow::DoubleType, double>(col1, col2, mean1, mean2);
            case Type::FLOAT:
                return simd::covariance<arrow::FloatType, double>(col1, col2,
                                                                  static_cast<typename arrow::FloatType::c_type>(mean1),
                                                                  static_cast<typename arrow::FloatType::c_type>(mean2));
            case Type::HALF_FLOAT:
                return simd::covariance<arrow::HalfFloatType, double>(col1, col2,
                                                                      static_cast<typename arrow::HalfFloatType::c_type>(mean1),
                                                                      static_cast<typename arrow::HalfFloatType::c_type>(mean2));
            default:
                throw pybind11::value_error("Only floating point data is implemented in var().");
        }
    }

    double covariance_unsafe(Array_ptr col1, Array_ptr col2, double mean1, double mean2, Buffer_ptr bitmap) {
        switch (col1->type_id()) {
            case Type::DOUBLE:
                return simd::covariance<arrow::DoubleType, double>(col1, col2, mean1, mean2, bitmap);
            case Type::FLOAT:
                return simd::covariance<arrow::FloatType, double>(col1, col2,
                                                           static_cast<typename arrow::FloatType::c_type>(mean1),
                                                           static_cast<typename arrow::FloatType::c_type>(mean2), bitmap);
            case Type::HALF_FLOAT:
                return simd::covariance<arrow::HalfFloatType, double>(col1, col2,
                                                                static_cast<typename arrow::HalfFloatType::c_type>(mean1),
                                                                static_cast<typename arrow::HalfFloatType::c_type>(mean2), bitmap);
            default:
                throw pybind11::value_error("Only floating point data is implemented in var().");
        }
    }

    double covariance(Array_ptr col1, Array_ptr col2, double mean1, double mean2) {
        if (col1->type_id() != col2->type_id()) {
            throw pybind11::value_error("Data type for both columns should be the same in covariance()");
        }

        return covariance_unsafe(col1, col2, mean1, mean2);
    }

    double covariance(Array_ptr col1, Array_ptr col2, double mean1, double mean2, Buffer_ptr bitmap) {
        if (col1->type_id() != col2->type_id()) {
            throw pybind11::value_error("Data type for both columns should be the same in covariance()");
        }

        return covariance_unsafe(col1, col2, mean1, mean2, bitmap);
    }

    double sse_unsafe(Array_ptr truth, Array_ptr predicted) {
        switch (truth->type_id()) {
            case Type::DOUBLE:
                return simd::sse<arrow::DoubleType, double>(truth, predicted);
            case Type::FLOAT:
                return simd::sse<arrow::FloatType, double>(truth, predicted);
            case Type::HALF_FLOAT:
                return simd::sse<arrow::HalfFloatType, double>(truth, predicted);
            default:
                throw pybind11::value_error("Only floating point data is implemented in sse().");
        }
    }

    double sse_unsafe(Array_ptr truth, Array_ptr predicted, Buffer_ptr bitmap) {
        switch (truth->type_id()) {
            case Type::DOUBLE:
                return simd::sse<arrow::DoubleType, double>(truth, predicted, bitmap);
            case Type::FLOAT:
                return simd::sse<arrow::FloatType, double>(truth, predicted, bitmap);
            case Type::HALF_FLOAT:
                return simd::sse<arrow::HalfFloatType, double>(truth, predicted, bitmap);
            default:
                throw pybind11::value_error("Only floating point data is implemented in sse().");
        }
    }

    double sse(Array_ptr truth, Array_ptr predicted) {
        if (truth->type_id() != predicted->type_id()) {
            throw pybind11::value_error("Data type for both columns should be the same in sse()");
        }

        return sse_unsafe(truth, predicted);
    }

    double sse(Array_ptr truth, Array_ptr predicted, Buffer_ptr bitmap) {
        if (truth->type_id() != predicted->type_id()) {
            throw pybind11::value_error("Data type for both columns should be the same in sse()");
        }

        return sse_unsafe(truth, predicted, bitmap);
    }
}

namespace linalg::linear_regression {

    template<typename ArrowType>
    Array_ptr fitted_values_typed(std::vector<double> beta, std::vector<Array_ptr> columns) {
        using CType = typename ArrowType::c_type;

        auto length = columns[0]->length();
        auto result = arrow::AllocateBuffer(length * sizeof(CType));

        if (!result.ok()) {
            throw std::bad_alloc();
        }

        auto buffer_data = std::move(result).ValueOrDie();
        auto raw_buffer_data = reinterpret_cast<CType*>(buffer_data->mutable_data());
        auto intercept = static_cast<CType>(beta[0]);
        std::fill(raw_buffer_data, raw_buffer_data + length, intercept);


        auto output_datatype = arrow::TypeTraits<ArrowType>::type_singleton();

        auto combined_bitmap = util::bit_util::combined_bitmap(columns);

        std::shared_ptr<NumericArray<ArrowType>> output_array;
        if (combined_bitmap) {
            auto null_count = util::bit_util::null_count(combined_bitmap, length);
            output_array = std::make_shared<arrow::NumericArray<ArrowType>>(output_datatype, length,
                                                                        std::move(buffer_data),
                                                                        std::move(combined_bitmap),
                                                                        null_count);
        } else {
            output_array = std::make_shared<arrow::NumericArray<ArrowType>>(output_datatype, length,
                                                                            std::move(buffer_data),
                                                                            nullptr, 0);
        }

        for (uint64_t i = 0; i < columns.size(); ++i) {
            simd::fmadd<ArrowType>(columns[i], beta[i+1], output_array);
        }

        return std::static_pointer_cast<Array>(output_array);
    }

//    TODO: Document: This function only accepts columns non empty
    Array_ptr fitted_values(std::vector<double> beta, std::vector<Array_ptr> columns) {
        auto dt_id = columns[0]->type_id();

        switch(dt_id) {
            case Type::DOUBLE:
                return fitted_values_typed<arrow::DoubleType>(beta, columns);
            case Type::FLOAT:
                return fitted_values_typed<arrow::FloatType>(beta, columns);
            case Type::HALF_FLOAT:
                return fitted_values_typed<arrow::HalfFloatType>(beta, columns);
            default:
                throw pybind11::value_error("Only floating point data is implemented in fitted_values().");
        }
    }
}