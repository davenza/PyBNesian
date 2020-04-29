
#ifndef PGM_DATASET_VARIANCE_HPP
#define PGM_DATASET_VARIANCE_HPP

#include <simd/simd_properties.hpp>
#include <simd/sum.hpp>

namespace simd::var_internals {

    template<typename ArrowType,
            std::enable_if_t<!SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sse_contiguous(std::shared_ptr <Array> array, typename ArrowType::c_type mean) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto dwn_array = std::static_pointer_cast<ArrayType>(array);
        auto raw_values = dwn_array->raw_values();

        auto accum = 0;
        for (int i = 0; i < dwn_array->length(); ++i) {
            auto error = raw_values[i] - mean;
            accum += error*error;
        }
        return accum;
    }

    template<typename ArrowType,
            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sse_contiguous(std::shared_ptr <Array> array, typename ArrowType::c_type mean) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using CType = typename ArrowType::c_type;

        auto dwn_array = std::static_pointer_cast<ArrayType>(array);
        auto raw_values = dwn_array->raw_values();

        auto p = simd_properties<ArrowType>(raw_values, dwn_array->length());

        if (p.aligned_words == 0) {
            CType accum = 0;
            for (int i = 0; i < dwn_array->length(); ++i) {
                auto error = raw_values[i] - mean;
                accum += error*error;
            }
            return accum;
        }

        CType accum = 0;
        for (int i = 0; i < p.leading_elements; ++i) {
            auto error = raw_values[i] - mean;
            accum += error*error;
        }

        auto simd_accum = SimdTraits<ArrowType>::simd_set1(0);
        auto simd_mean = SimdTraits<ArrowType>::simd_set1(mean);
        for (int w = 0, offset = p.leading_elements; w < p.aligned_words; ++w, offset += SimdTraits<ArrowType>::LANES) {
            auto l = SimdTraits<ArrowType>::simd_load(raw_values + offset);
            auto error = SimdTraits<ArrowType>::simd_sub(l, simd_mean);
            auto sq_error = SimdTraits<ArrowType>::simd_mul(error, error);
            simd_accum = SimdTraits<ArrowType>::simd_add(simd_accum, sq_error);
        }

        accum += SimdTraits<ArrowType>::horizontal_sum(simd_accum);

        for (int i = dwn_array->length() - p.trailing_elements; i < dwn_array->length(); ++i) {
            auto error = raw_values[i] - mean;
            accum += error*error;
        }

        return accum;
    }

    template<typename ArrowType,
            std::enable_if_t<!SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sse_non_contiguous(std::shared_ptr <Array> array, typename ArrowType::c_type mean) {
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
                auto error = raw_values[i] - mean;
                accum += error*error;
            }
        }

        if (p.aligned_words > 0) {
            const uint64_t *u64_bitmap = reinterpret_cast<const uint64_t *>(null_bitmap);

            auto offset_values = 0;
            for (auto i = 0; i < p.aligned_words; ++i) {
                if (u64_bitmap[i] == 0xFFFFFFFFFFFFFFFF) {
                    for (auto j = 0; j < 64; ++j) {
                        auto error = raw_values[offset_values + j] - mean;
                        accum += error*error;
                    }
                } else {
                    for (auto j = 0; j < 64; ++j) {
                        if (GetBit(null_bitmap, offset_values + j)) {
                            auto error = raw_values[offset_values + j] - mean;
                            accum += error*error;
                        }
                    }
                }
                offset_values += 64;
            }
        }

        for (int64_t i = p.trailing_bit_offset; i < length; ++i) {
            if (GetBit(null_bitmap, i)) {
                auto error = raw_values[i] - mean;
                accum += error*error;
            }
        }

        return accum;
    }


    template<typename ArrowType,
            typename = std::enable_if_t <SimdTraits<ArrowType>::CAN_SIMD>>
    inline typename SimdTraits<ArrowType>::AVX_TYPE
    sse_next64values_unsafe(const typename ArrowType::c_type *array,
                            typename SimdTraits<ArrowType>::AVX_TYPE simd_mean,
                            typename SimdTraits<ArrowType>::AVX_TYPE simd_accum) {
        for (auto j = 0; j < 64; j += SimdTraits<ArrowType>::LANES) {
            auto l = SimdTraits<ArrowType>::simd_load(array + j);
            auto error = SimdTraits<ArrowType>::simd_sub(l, simd_mean);
            auto sq_error = SimdTraits<ArrowType>::simd_mul(error, error);
            simd_accum = SimdTraits<ArrowType>::simd_add(simd_accum, sq_error);
        }
        return simd_accum;
    }

    template<typename ArrowType,
            typename = std::enable_if_t <SimdTraits<ArrowType>::CAN_SIMD>>
    inline void
    sse_next64values(const typename ArrowType::c_type *array,
                            typename ArrowType::c_type mean,
                            const uint8_t* null_bitmap,
                            typename ArrowType::c_type& accum) {

        for (auto i = 0; i < 64; i += SimdTraits<ArrowType>::LANES) {
            if (GetBit(null_bitmap, i)) {
                auto error = array[i] - mean;
                accum += error*error;
            }
        }
    }



    template<typename ArrowType,
            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sse_non_contiguous(std::shared_ptr <Array> array, typename ArrowType::c_type mean) {
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
                    auto error = raw_values[i] - mean;
                    accum += error*error;
                }
            }
            return accum;
        }

        CType accum = 0;

        for (int64_t i = 0; i < p.leading_elements; ++i) {
            if (GetBit(null_bitmap, i)) {
                auto error = raw_values[i] - mean;
                accum += error*error;
            }
        }

        if (p.aligned_words > 0) {
            auto simd_accum = SimdTraits<ArrowType>::simd_set1(0);
            auto simd_mean = SimdTraits<ArrowType>::simd_set1(mean);

            auto u64_bitmap = reinterpret_cast<const uint64_t*>(null_bitmap + p.leading_elements);

            auto u64_end = reinterpret_cast<const uint64_t*>(null_bitmap + p.leading_elements +
                                                             (p.aligned_words*SimdTraits<ArrowType>::LANES)/8
            );

            auto aligned_values = raw_values + p.leading_elements;

            auto values_offset = 0;
            for (auto bitmap_addr = u64_bitmap; bitmap_addr < u64_end; ++bitmap_addr) {
                if (*bitmap_addr == 0xFFFFFFFFFFFFFFFF) {
                    simd_accum = sse_next64values_unsafe<ArrowType>(aligned_values + values_offset, simd_mean, simd_accum);
                } else {
                    sse_next64values<ArrowType>(aligned_values + values_offset, mean,
                                     reinterpret_cast<const uint8_t*>(bitmap_addr), accum);
                }
                values_offset += 64;
            }

            accum += SimdTraits<ArrowType>::horizontal_sum(simd_accum);
        }

        for (int64_t i = length - p.trailing_elements; i < length; ++i) {
            if (GetBit(null_bitmap, i)) {
                auto error = raw_values[i] - mean;
                accum += error*error;
            }
        }

        return accum;
    }
}

namespace simd {

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    var(std::shared_ptr<Array> array) {
        typename ArrowType::c_type mean = simd::mean<ArrowType>(array);

        auto null_count = array->null_count();
        std::cout << "null_count" << null_count << std::endl;
        if (null_count == 0) {
            return static_cast<Output>(simd::var_internals::sse_contiguous<ArrowType>(array, mean))
                                / (array->length() - 1);
        } else {
            return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean))
                        / (array->length() - null_count - 1);
        }
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    var(std::shared_ptr<Array> array, typename ArrowType::c_type mean) {
        auto null_count = array->null_count();
        if (null_count == 0) {
            return static_cast<Output>(simd::var_internals::sse_contiguous<ArrowType>(array, mean))
                                    / (array->length() - 1);
        } else {
            return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean))
                                    / (array->length() - null_count - 1);
        }
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    sse(std::shared_ptr<Array> array) {
        typename ArrowType::c_type mean = simd::mean<ArrowType>(array);

        auto null_count = array->null_count();
        if (null_count == 0) {
            return static_cast<Output>(simd::var_internals::sse_contiguous<ArrowType>(array, mean));
        } else {
            return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean));
        }
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    sse(std::shared_ptr<Array> array, typename ArrowType::c_type mean) {
        auto null_count = array->null_count();
        if (null_count == 0) {
            return static_cast<Output>(simd::var_internals::sse_contiguous<ArrowType>(array, mean));
        } else {
            return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean));
        }
    }
}

#endif //PGM_DATASET_VARIANCE_HPP
