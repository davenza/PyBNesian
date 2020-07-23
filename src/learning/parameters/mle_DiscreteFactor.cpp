#include <learning/parameters/mle_DiscreteFactor.hpp>

namespace learning::parameters {

    void sum_indices_null(VectorXi& accum_indices, Array_ptr& indices, int stride, Buffer_ptr& combined_bitmap) {
        switch(indices->type_id()) {
            case Type::INT8:
                sum_indices_null<arrow::Int8Type>(accum_indices, indices, stride, combined_bitmap);
                break;
            case Type::INT16:
                sum_indices_null<arrow::Int16Type>(accum_indices, indices, stride, combined_bitmap);
                break;
            case Type::INT32:
                sum_indices_null<arrow::Int32Type>(accum_indices, indices, stride, combined_bitmap);
                break;
            case Type::INT64:
                sum_indices_null<arrow::Int64Type>(accum_indices, indices, stride, combined_bitmap);
                break;
            default:
                throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
        }
    }

    void sum_indices(VectorXi& accum_indices, Array_ptr& indices, int stride) {
        switch(indices->type_id()) {
            case Type::INT8:
                sum_indices<arrow::Int8Type>(accum_indices, indices, stride);
                break;
            case Type::INT16:
                sum_indices<arrow::Int16Type>(accum_indices, indices, stride);
                break;
            case Type::INT32:
                sum_indices<arrow::Int32Type>(accum_indices, indices, stride);
                break;
            case Type::INT64:
                sum_indices<arrow::Int64Type>(accum_indices, indices, stride);
                break;
            default:
                throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
        }
    }
}