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

    template<typename ArrowType>
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

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //        TODO: This is a SIMD implementation of sse_non_contiguous. It does not improve significantly.
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    template<typename ArrowType,
//            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
//    typename ArrowType::c_type
//    inline sum_scalarNelements(const typename ArrowType::c_type* raw_values,
//                               const uint8_t* null_bitmap,
//                               uint64_t offset,
//                               uint64_t length) {
//        typename ArrowType::c_type accum = 0;
//        for(uint64_t i = 0; i < length; ++i) {
//            if (GetBit(null_bitmap, offset + i)) {
//                accum += raw_values[offset+i];
//            }
//        }
//
//        return accum;
//    }
//
//    template<typename ArrowType,
//            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
//    typename SimdTraits<ArrowType>::AVX_TYPE
//    inline sum_next64values_unsafe(const typename ArrowType::c_type* raw_values,
//                                   typename SimdTraits<ArrowType>::AVX_TYPE simd_accum) {
//        for(auto i = 0; i < 64; i += SimdTraits<ArrowType>::LANES) {
//            auto l = SimdTraits<ArrowType>::simd_load(raw_values + i);
//            simd_accum = SimdTraits<ArrowType>::simd_add(simd_accum, l);
//        }
//
//        return simd_accum;
//    }
//
//    template<typename ArrowType,
//            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
//    typename ArrowType::c_type
//    inline sum_next64values(const typename ArrowType::c_type* raw_values,
//                               const uint8_t* null_bitmap) {
//        typename ArrowType::c_type accum = 0;
//        for(uint64_t i = 0; i < 64; ++i) {
//            if (GetBit(null_bitmap, i)) {
////                std::cout << "Summing " << i << ": " << raw_values[i] << std::endl;
//                accum += raw_values[i];
//            }
//        }
//
//        return accum;
//    }
//
//    template<bool zero_leading_elements>
//    inline uint64_t next_chunk_bitmap(const uint64_t* u64_bitmap,
//                                       const uint8_t* null_bitmap,
//                                       uint64_t leading_elements,
//                                       uint64_t chunk) {
//        if (zero_leading_elements) {
//            return u64_bitmap[chunk];
//        } else {
//            return (u64_bitmap[chunk] >> leading_elements) |
//                   (static_cast<uint64_t>(null_bitmap[chunk * 8]) << (64 - leading_elements));
//        }
//    }
//
//
//    template<typename ArrowType,
//             std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
//    typename ArrowType::c_type
//    sum_non_contiguous(std::shared_ptr <Array> array) {
//        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
//        using CType = typename ArrowType::c_type;
//        using arrow::internal::BitmapWordAlign;
//
//        auto dwn_array = std::static_pointer_cast<ArrayType>(array);
//        auto raw_values = dwn_array->raw_values();
//
//        auto length = dwn_array->length();
//
//        auto p = simd_properties<ArrowType>(raw_values, length);
//        auto null_bitmap = dwn_array->null_bitmap()->data();
//
//        if (p.aligned_words == 0) {
//            return sum_scalarNelements<ArrowType>(raw_values, null_bitmap, 0, length);
//        }
//
//        CType accum = 0;
//
//        auto next_bitmap_fun = next_chunk_bitmap<true>;
//        if (p.leading_elements > 0) {
//            accum += sum_scalarNelements<ArrowType>(raw_values, null_bitmap, 0, p.leading_elements);
//            next_bitmap_fun = next_chunk_bitmap<false>;
//        }
//
//
//        if (p.aligned_words > 0) {
//            auto aligned_addr = raw_values + p.leading_elements;
//            auto aligned_elements = p.aligned_words * SimdTraits<ArrowType>::LANES;
//
//            if (aligned_elements < 64) {
//                accum += sum_scalarNelements<ArrowType>(aligned_addr, null_bitmap, 0, aligned_elements);
//            } else {
//                const uint64_t* u64_bitmap = reinterpret_cast<const uint64_t*>(null_bitmap);
//
//                const uint64_t words_per_chunk = 64 / SimdTraits<ArrowType>::LANES;
//                const uint64_t simd_chunks = p.aligned_words / words_per_chunk;
//
//                auto simd_accum = SimdTraits<ArrowType>::simd_set1(0);
//
//                uint64_t current_bitmap_chunk = 0;
//                for(uint64_t w = 0; w < simd_chunks; ++w) {
//                    current_bitmap_chunk = next_bitmap_fun(u64_bitmap, null_bitmap, p.leading_elements, w);
//
//                    if (current_bitmap_chunk == 0xFFFFFFFFFFFFFFFF) {
//                        simd_accum = sum_next64values_unsafe<ArrowType>(aligned_addr + 64*w, simd_accum);
//                    } else {
//                        accum += sum_next64values<ArrowType>(aligned_addr + 64*w,
//                                                             reinterpret_cast<uint8_t*>(&current_bitmap_chunk));
//                    }
//
//                }
//
//                const auto remaining_words = p.aligned_words % words_per_chunk;
//                const auto remaining_elements = remaining_words * SimdTraits<ArrowType>::LANES;
//                accum += sum_scalarNelements<ArrowType>(aligned_addr, null_bitmap, 64*simd_chunks, remaining_elements);
//                accum += SimdTraits<ArrowType>::horizontal_sum(simd_accum);
//            }
//        }
//
//        if (p.trailing_elements > 0) {
//            accum += sum_scalarNelements<ArrowType>(raw_values, null_bitmap, length-p.trailing_elements, length);
//        }
//
//        return accum;
//    }
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
