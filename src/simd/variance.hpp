
#ifndef PGM_DATASET_VARIANCE_HPP
#define PGM_DATASET_VARIANCE_HPP

#include <simd/simd_properties.hpp>
#include <simd/sum.hpp>
#include <simd/bit_util.hpp>


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
        for (uint64_t i = 0; i < p.leading_elements; ++i) {
            auto error = raw_values[i] - mean;
            accum += error*error;
        }

        auto simd_accum = SimdTraits<ArrowType>::simd_set1(0);
        auto simd_mean = SimdTraits<ArrowType>::simd_set1(mean);
        for (uint64_t w = 0, offset = p.leading_elements; w < p.aligned_words; ++w, offset += SimdTraits<ArrowType>::LANES) {
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

//        TODO: Can we optimize better using SIMD?
    template<typename ArrowType>
    typename ArrowType::c_type
    sse_non_contiguous(std::shared_ptr <Array> array, typename ArrowType::c_type mean) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using CType = typename ArrowType::c_type;

        auto dwn_array = std::static_pointer_cast<ArrayType>(array);
        auto raw_values = dwn_array->raw_values();
        auto null_bitmap = dwn_array->null_bitmap()->data();

        auto length = dwn_array->length();

        const auto p = bit_util::bitmap_words<64>(length);

        CType accum = 0;

        if (p.words > 0) {
            const uint64_t *u64_bitmap = reinterpret_cast<const uint64_t *>(null_bitmap);

            auto offset_values = 0;
            for (uint64_t i = 0; i < p.words; ++i) {
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
            std::enable_if_t<!SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sse_covariance_contiguous(std::shared_ptr <Array> array1, std::shared_ptr <Array> array2,
                            const typename ArrowType::c_type mean1, const typename ArrowType::c_type mean2) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto dwn_array1 = std::static_pointer_cast<ArrayType>(array1);
        auto raw_values1 = dwn_array1->raw_values();

        auto dwn_array2 = std::static_pointer_cast<ArrayType>(array2);
        auto raw_values2 = dwn_array2->raw_values();

        auto accum = 0;
        auto length = dwn_array1->length();
        for (int i = 0; i < length; ++i) {
            accum += (raw_values1[i] - mean1) * (raw_values2[i] - mean2);
        }

        return accum;
    }


    using arrow::NumericArray;

    template<typename ArrowType,
            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    same_alignment_sse_covariance_contiguous(std::shared_ptr<NumericArray<ArrowType>> array1,
                                         std::shared_ptr<NumericArray<ArrowType>> array2,
                                         const typename ArrowType::c_type mean1,
                                         const typename ArrowType::c_type mean2,
                                         SimdProperties p)
    {
        using CType = typename ArrowType::c_type;

        auto raw_values1 = array1->raw_values();
        auto raw_values2 = array2->raw_values();

        if (p.aligned_words == 0) {
            CType accum = 0;
            auto length = array1->length();
            for (int i = 0; i < length; ++i) {
                accum += (raw_values1[i] - mean1) * (raw_values2[i] - mean2);
            }
            return accum / (length - 1);
        }

        CType accum = 0;
        for (uint64_t i = 0; i < p.leading_elements; ++i) {
            accum += (raw_values1[i] - mean1) * (raw_values2[i] - mean2);
        }

        auto simd_accum = SimdTraits<ArrowType>::simd_set1(0);
        auto simd_mean1 = SimdTraits<ArrowType>::simd_set1(mean1);
        auto simd_mean2 = SimdTraits<ArrowType>::simd_set1(mean2);

        for (uint64_t w = 0, offset = p.leading_elements; w < p.aligned_words; ++w, offset += SimdTraits<ArrowType>::LANES) {
            auto l1 = SimdTraits<ArrowType>::simd_load(raw_values1 + offset);
            auto l2 = SimdTraits<ArrowType>::simd_load(raw_values2 + offset);

            auto d1 = SimdTraits<ArrowType>::simd_sub(l1, simd_mean1);
            auto d2 = SimdTraits<ArrowType>::simd_sub(l2, simd_mean2);

            simd_accum = SimdTraits<ArrowType>::simd_add(simd_accum, SimdTraits<ArrowType>::simd_mul(d1, d2));
        }

        accum += SimdTraits<ArrowType>::horizontal_sum(simd_accum);

        uint64_t length = array1->length();
        for (uint64_t i = length - p.trailing_elements; i < length; ++i) {
            accum += (raw_values1[i] - mean1) * (raw_values2[i] - mean2);
        }

        return accum;
    }

    template<typename ArrowType,
            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    different_alignment_sse_covariance_contiguous(std::shared_ptr<NumericArray<ArrowType>> array1,
                                         std::shared_ptr<NumericArray<ArrowType>> array2,
                                         const typename ArrowType::c_type mean1,
                                         const typename ArrowType::c_type mean2,
                                         SimdProperties p1)
    {
        using CType = typename ArrowType::c_type;

        auto raw_values1 = array1->raw_values();
        auto raw_values2 = array2->raw_values();

        if (p1.aligned_words == 0) {
            CType accum = 0;
            auto length = array1->length();
            for (int i = 0; i < length; ++i) {
                accum += (raw_values1[i] - mean1) * (raw_values2[i] - mean2);
            }
            return accum / (length - 1);
        }

        CType accum = 0;
        for (uint64_t i = 0; i < p1.leading_elements; ++i) {
            accum += (raw_values1[i] - mean1) * (raw_values2[i] - mean2);
        }

        auto simd_accum = SimdTraits<ArrowType>::simd_set1(0);
        auto simd_mean1 = SimdTraits<ArrowType>::simd_set1(mean1);
        auto simd_mean2 = SimdTraits<ArrowType>::simd_set1(mean2);

        for (uint64_t w = 0, offset = p1.leading_elements; w < p1.aligned_words; ++w, offset += SimdTraits<ArrowType>::LANES) {
            auto l1 = SimdTraits<ArrowType>::simd_load(raw_values1 + offset);
            auto l2 = SimdTraits<ArrowType>::simd_loadu(raw_values2 + offset);

            auto d1 = SimdTraits<ArrowType>::simd_sub(l1, simd_mean1);
            auto d2 = SimdTraits<ArrowType>::simd_sub(l2, simd_mean2);

            simd_accum = SimdTraits<ArrowType>::simd_add(simd_accum, SimdTraits<ArrowType>::simd_mul(d1, d2));
        }

        accum += SimdTraits<ArrowType>::horizontal_sum(simd_accum);
        auto length = array1->length();
        for (int i = length - p1.trailing_elements; i < length; ++i) {
            accum += (raw_values1[i] - mean1) * (raw_values2[i] - mean2);
        }

        return accum;
    }

    template<typename ArrowType,
            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sse_covariance_contiguous(std::shared_ptr <Array> array1,
                          std::shared_ptr <Array> array2,
                          typename ArrowType::c_type mean1,
                          typename ArrowType::c_type mean2)
  {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        auto dwn_array1 = std::static_pointer_cast<ArrayType>(array1);
        auto raw_values1 = dwn_array1->raw_values();

        auto dwn_array2 = std::static_pointer_cast<ArrayType>(array2);
        auto raw_values2 = dwn_array2->raw_values();

        auto p1 = simd_properties<ArrowType>(raw_values1, dwn_array1->length());
        auto p2 = simd_properties<ArrowType>(raw_values2, dwn_array2->length());

        if (p1.leading_elements == p2.leading_elements) {
            return same_alignment_sse_covariance_contiguous(dwn_array1, dwn_array2, mean1, mean2, p1);
        } else {
            return different_alignment_sse_covariance_contiguous(dwn_array1, dwn_array2, mean1, mean2, p1);
        }
    }

    template<typename ArrowType>
    typename ArrowType::c_type
    sse_covariance_non_contiguous(std::shared_ptr <Array> array1,
                                  std::shared_ptr <Array> array2,
                                  typename ArrowType::c_type mean1,
                                  typename ArrowType::c_type mean2) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using CType = typename ArrowType::c_type;

        auto dwn_array1 = std::static_pointer_cast<ArrayType>(array1);
        auto raw_values1 = dwn_array1->raw_values();

        auto dwn_array2 = std::static_pointer_cast<ArrayType>(array2);
        auto raw_values2 = dwn_array2->raw_values();

        auto null_bitmap1 = dwn_array1->null_bitmap()->data();
        auto null_bitmap2 = dwn_array2->null_bitmap()->data();

        auto length = dwn_array1->length();

        const auto p = bit_util::bitmap_words<64>(length);

        CType accum = 0;

        if (p.words > 0) {
            const uint64_t *u64_bitmap1 = reinterpret_cast<const uint64_t *>(null_bitmap1);
            const uint64_t *u64_bitmap2 = reinterpret_cast<const uint64_t *>(null_bitmap2);

            auto offset_values = 0;
            for (uint64_t i = 0; i < p.words; ++i) {
                if ((u64_bitmap1[i] & u64_bitmap2[i])  == 0xFFFFFFFFFFFFFFFF) {
                    for (auto j = 0; j < 64; ++j) {
                        accum += (raw_values1[offset_values + j] - mean1) * (raw_values2[offset_values + j] - mean2);
                    }
                } else {
                    for (auto j = 0; j < 64; ++j) {
                        if (GetBit(null_bitmap1, offset_values + j) && GetBit(null_bitmap2, offset_values + j)) {
                            accum += (raw_values1[offset_values + j] - mean1) * (raw_values2[offset_values + j] - mean2);
                        }
                    }
                }
                offset_values += 64;
            }
        }

        for (int64_t i = p.trailing_bit_offset; i < length; ++i) {
            if (GetBit(null_bitmap1, i) && GetBit(null_bitmap2, i)) {
                accum += (raw_values1[i] - mean1) * (raw_values2[i] - mean2);
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

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    covariance(std::shared_ptr<Array> array1, std::shared_ptr<Array> array2,
                typename ArrowType::c_type mean1, typename ArrowType::c_type mean2) {
        auto null_count1 = array1->null_count();
        auto null_count2 = array2->null_count();
        auto length = array1->length();

        if (null_count1 == 0 && null_count2 == 0) {
            return
            static_cast<Output>(simd::var_internals::sse_covariance_contiguous<ArrowType>(array1, array2, mean1, mean2))
                                                                                    / (length - 1);
        } else {
            uint64_t combined_count = bit_util::count_combined_set_bits(array1->null_bitmap()->data(),
                                                                        array2->null_bitmap()->data(), length);
            return
            static_cast<Output>(simd::var_internals::sse_covariance_non_contiguous<ArrowType>(array1, array2, mean1, mean2))
                                                                                / (combined_count - 1);
        }
    }
}

#endif //PGM_DATASET_VARIANCE_HPP
