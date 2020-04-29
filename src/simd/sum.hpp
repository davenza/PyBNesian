#ifndef PGM_DATASET_SUM_HPP
#define PGM_DATASET_SUM_HPP

#include <arrow/api.h>
#include <arrow/util/bit_util.h>
#include <immintrin.h>
#include <iostream>
#include <util/align_util.hpp>
#include <simd/simd_properties.hpp>

using bit_util::BitmapWordAlignExtra;
using arrow::BitUtil::GetBit;

using arrow::Array;

namespace simd::sum_internals {

    template<typename ArrowType,
            std::enable_if_t<!SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sum_contiguous(std::shared_ptr <Array> array) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto dwn_array = std::static_pointer_cast<ArrayType>(array);
        auto raw_values = dwn_array->raw_values();

        auto accum = 0;
        for (int i = 0; i < dwn_array->length(); ++i) {
            accum += raw_values[i];
        }
        return accum;
    }

    template<typename ArrowType,
            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sum_contiguous(std::shared_ptr <Array> array) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using CType = typename ArrowType::c_type;

        auto dwn_array = std::static_pointer_cast<ArrayType>(array);
        auto raw_values = dwn_array->raw_values();

        auto p = simd_properties<ArrowType>(raw_values, dwn_array->length());

        if (p.aligned_words == 0) {
            CType accum = 0;
            for (int i = 0; i < dwn_array->length(); ++i) {
                accum += raw_values[i];
            }
            return accum;
        }

        CType accum = 0;
        for (int i = 0; i < p.leading_elements; ++i) {
            accum += raw_values[i];
        }

        auto simd_accum = SimdTraits<ArrowType>::simd_set1(0);
        for (int w = 0, offset = p.leading_elements; w < p.aligned_words; ++w, offset += SimdTraits<ArrowType>::LANES) {
            auto l = SimdTraits<ArrowType>::simd_load(raw_values + offset);
            simd_accum = SimdTraits<ArrowType>::simd_add(simd_accum, l);
        }

        accum += SimdTraits<ArrowType>::horizontal_sum(simd_accum);

        for (int i = dwn_array->length() - p.trailing_elements; i < dwn_array->length(); ++i) {
            accum += raw_values[i];
        }

        return accum;
    }


    template<typename ArrowType,
            std::enable_if_t<!SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sum_non_contiguous(std::shared_ptr <Array> array) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using CType = typename ArrowType::c_type;
        using arrow::internal::BitmapWordAlign;

        auto dwn_array = std::static_pointer_cast<ArrayType>(array);
        auto raw_values = dwn_array->raw_values();
        auto null_bitmap = dwn_array->null_bitmap()->data();

        auto length = dwn_array->length();

        const auto p = BitmapWordAlign<8>(null_bitmap, 0, length);

        CType accum = 0;

        for (int64_t i = 0; i < p.leading_bits; ++i) {
            if (GetBit(null_bitmap, i)) {
                accum += raw_values[i];
            }
        }

        if (p.aligned_words > 0) {
            const uint64_t *u64_bitmap = reinterpret_cast<const uint64_t *>(null_bitmap);

            auto offset_values = 0;
            for (auto i = 0; i < p.aligned_words; ++i) {
                if (u64_bitmap[i] == 0xFFFFFFFFFFFFFFFF) {
                    for (auto j = 0; j < 64; ++j) {
                        accum += raw_values[offset_values + j];
                    }
                } else {
                    for (auto j = 0; j < 64; ++j) {
                        if (GetBit(null_bitmap, offset_values + j)) {
                            accum += raw_values[offset_values + j];
                        }
                    }
                }
                offset_values += 64;
            }
        }

        for (int64_t i = p.trailing_bit_offset; i < length; ++i) {
            if (GetBit(null_bitmap, i)) {
                accum += raw_values[i];
            }
        }

        return accum;
    }

    template<typename ArrowType,
            typename = std::enable_if_t <SimdTraits<ArrowType>::CAN_SIMD>>
    inline typename SimdTraits<ArrowType>::AVX_TYPE
    sum_next64values_unsafe(const typename ArrowType::c_type *array, typename SimdTraits<ArrowType>::AVX_TYPE simd_accum) {
        for (auto j = 0; j < 64; j += SimdTraits<ArrowType>::LANES) {
            auto l = SimdTraits<ArrowType>::simd_load(array + j);
            simd_accum = SimdTraits<ArrowType>::simd_add(simd_accum, l);
        }
        return simd_accum;
    }

    template<typename ArrowType,
            typename = std::enable_if_t <SimdTraits<ArrowType>::CAN_SIMD>>
    inline void
    sum_next64values(const typename ArrowType::c_type *array,
                     const uint8_t* null_bitmap,
                     typename ArrowType::c_type& scalar_accum) {
        for (auto j = 0; j < 64; ++j) {
            if (GetBit(null_bitmap, j)) {
                scalar_accum += array[j];
            }
        }
    }

    template<typename ArrowType,
            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sum_non_contiguous(std::shared_ptr <Array> array) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using CType = typename ArrowType::c_type;


        auto dwn_array = std::static_pointer_cast<ArrayType>(array);
        auto raw_values = dwn_array->raw_values();
        auto length = dwn_array->length();

        auto p = simd_properties<ArrowType>(raw_values, length);

        auto null_bitmap = dwn_array->null_bitmap()->data();

        if (p.aligned_words == 0) {
            CType accum = 0;
            for (int i = 0; i < p.leading_elements; ++i) {
                if (GetBit(null_bitmap, i)) {
                    accum += raw_values[i];
                }
            }
            return accum;
        }

        CType accum = 0;

        for (int64_t i = 0; i < p.leading_elements; ++i) {
            if (GetBit(null_bitmap, i)) {
                accum += raw_values[i];
            }
        }

        if (p.aligned_words > 0) {
            auto simd_accum = SimdTraits<ArrowType>::simd_set1(0);

            auto u64_bitmap = reinterpret_cast<const uint64_t*>(null_bitmap + p.leading_elements);

            auto u64_end = reinterpret_cast<const uint64_t*>(null_bitmap + p.leading_elements +
                                                             (p.aligned_words*SimdTraits<ArrowType>::LANES)/8
                                                            );


            auto aligned_values = raw_values + p.leading_elements;

            auto values_offset = 0;
            for (auto bitmap_addr = u64_bitmap; bitmap_addr < u64_end; ++bitmap_addr) {

                if (*bitmap_addr == 0xFFFFFFFFFFFFFFFF) {
                    simd_accum = sum_next64values_unsafe<ArrowType>(aligned_values + values_offset, simd_accum);
                } else {
                    sum_next64values<ArrowType>(aligned_values + values_offset,
                                     reinterpret_cast<const uint8_t*>(bitmap_addr), accum);
                }
                values_offset += 64;
            }

            accum += SimdTraits<ArrowType>::horizontal_sum(simd_accum);
        }

        for (int64_t i = length - p.trailing_elements; i < length; ++i) {
            if (GetBit(null_bitmap, i)) {
                accum += raw_values[i];
            }
        }

        return accum;
    }
}

namespace simd {
    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    mean(std::shared_ptr<Array> array) {
        auto null_count = array->null_count();
        if (array->null_count() == 0) {
            return static_cast<Output>(simd::sum_internals::sum_contiguous<ArrowType>(array)) / array->length();
        } else {
            return static_cast<Output>(simd::sum_internals::sum_non_contiguous<ArrowType>(array))
                                            / (array->length() - null_count);
        }
    }


    template<typename ArrowType>
    typename ArrowType::c_type
    sum(std::shared_ptr<Array> array) {
        if (array->null_count() == 0) {
            return simd::sum_internals::sum_contiguous<ArrowType>(array);
        } else {
            return simd::sum_internals::sum_non_contiguous<ArrowType>(array);
        }
    }
}


#endif //PGM_DATASET_SUM_HPP
