#ifndef PGM_DATASET_LINEAR_REGRESSION_HPP
#define PGM_DATASET_LINEAR_REGRESSION_HPP

#include <arrow/api.h>
#include <pybind11/pybind11.h>
#include <simd/bit_util.hpp>

namespace simd::fmadd_internals {

    template<typename ArrowType,
             typename = std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD>>
    void fmadd_same_alignment(std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType> input_array,
                              typename ArrowType::c_type coeff,
                              std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType> output_array,
                              SimdProperties p) {
        using CType = typename ArrowType::c_type;
        auto raw_input = input_array->raw_values();
        auto raw_output = reinterpret_cast<CType*>(output_array->values()->mutable_data());

        for (uint64_t i = 0; i < p.leading_elements; ++i) {
            raw_output[i] += coeff*raw_input[i];
        }

        auto simd_coeff = SimdTraits<ArrowType>::simd_set1(coeff);
        for (uint64_t w = 0, offset = p.leading_elements;
            w < p.aligned_words;
            ++w, offset += SimdTraits<ArrowType>::LANES) {

            auto simd_input = SimdTraits<ArrowType>::simd_load(raw_input + offset);
            auto simd_output = SimdTraits<ArrowType>::simd_load(raw_output + offset);
            simd_output = SimdTraits<ArrowType>::simd_fmadd(simd_input, simd_coeff, simd_output);

            SimdTraits<ArrowType>::simd_stream_store(raw_output + offset, simd_output);
        }

        uint64_t length = input_array->length();
        for (uint64_t i = length - p.trailing_elements; i < length; ++i) {
            raw_output[i] += coeff*raw_input[i];
        }
    }

    template<typename ArrowType,
            typename = std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD>>
    void fmadd_different_alignment(std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType> input_array,
                                   typename ArrowType::c_type coeff,
                                   std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType> output_array,
                                   SimdProperties p) {
        using CType = typename ArrowType::c_type;
        auto raw_input = input_array->raw_values();
        auto raw_output = reinterpret_cast<CType*>(output_array->values()->mutable_data());

        for (uint64_t i = 0; i < p.leading_elements; ++i) {
            raw_output[i] += coeff*raw_input[i];
        }

        auto simd_coeff = SimdTraits<ArrowType>::simd_set1(coeff);
        for (uint64_t w = 0, offset = p.leading_elements;
             w < p.aligned_words;
             ++w, offset += SimdTraits<ArrowType>::LANES) {

            auto simd_input = SimdTraits<ArrowType>::simd_loadu(raw_input + offset);
            auto simd_output = SimdTraits<ArrowType>::simd_load(raw_output + offset);
            simd_output = SimdTraits<ArrowType>::simd_fmadd(simd_input, simd_coeff, simd_output);

            SimdTraits<ArrowType>::simd_stream_store(raw_output + offset, simd_output);
        }

        uint64_t length = input_array->length();
        for (uint64_t i = length - p.trailing_elements; i < length; ++i) {
            raw_output[i] += coeff*raw_input[i];
        }
    }
}


namespace simd {
    typedef std::shared_ptr<arrow::Array> Array_ptr;

    template<typename ArrowType,
            std::enable_if_t<!SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    void fmadd(Array_ptr input_array, typename ArrowType::c_type coeff, Array_ptr output_array) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto dwn_input = std::static_pointer_cast<ArrayType>(input_array);
        auto dwn_output = std::static_pointer_cast<ArrayType>(output_array);

        auto raw_input = dwn_input->raw_values();
        auto raw_output = dwn_output->values()->mutable_data();

        uint64_t length = input_array->length();

        for (uint64_t i = 0; i < length; ++i) {
            raw_output[i] += coeff*raw_input[i];
        }
    }

    template<typename ArrowType,
            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    void fmadd(Array_ptr input_array, typename ArrowType::c_type coeff, Array_ptr output_array) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using CType = typename ArrowType::c_type;

        auto dwn_input = std::static_pointer_cast<ArrayType>(input_array);
        auto dwn_output = std::static_pointer_cast<ArrayType>(output_array);

        auto raw_input = dwn_input->raw_values();
        auto raw_output = reinterpret_cast<CType*>(dwn_output->values()->mutable_data());

        uint64_t length = dwn_input->length();

        auto p_input = simd_properties<ArrowType>(raw_input, length);
        auto p_output = simd_properties<ArrowType>(raw_output, length);

        if (p_input.leading_elements == p_output.leading_elements) {
            simd::fmadd_internals::fmadd_same_alignment<ArrowType>(dwn_input, coeff, dwn_output, p_output);
        } else {
            simd::fmadd_internals::fmadd_different_alignment<ArrowType>(dwn_input, coeff, dwn_output, p_output);
        }
    }
}


#endif //PGM_DATASET_LINEAR_REGRESSION_HPP
