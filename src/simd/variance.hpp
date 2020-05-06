
#ifndef PGM_DATASET_VARIANCE_HPP
#define PGM_DATASET_VARIANCE_HPP

#include <simd/simd_properties.hpp>
#include <simd/sum.hpp>
#include <util/bit_util.hpp>

typedef std::shared_ptr<Array> Array_ptr;
typedef std::shared_ptr<arrow::Buffer> Buffer_ptr;

namespace simd::var_internals {


    template<typename ArrowType,
            std::enable_if_t<!SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sse_contiguous(Array_ptr array, typename ArrowType::c_type mean) {
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
    sse_contiguous(Array_ptr array, typename ArrowType::c_type mean) {
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
    sse_non_contiguous(Array_ptr array, typename ArrowType::c_type mean, Buffer_ptr bitmap) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using CType = typename ArrowType::c_type;

        auto dwn_array = std::static_pointer_cast<ArrayType>(array);
        auto raw_values = dwn_array->raw_values();
        auto null_bitmap = bitmap->data();

        auto length = dwn_array->length();

        const auto bp = util::bit_util::bitmap_words<64>(length);

        CType accum = 0;

        if (bp.words > 0) {
            const uint64_t *u64_bitmap = reinterpret_cast<const uint64_t *>(null_bitmap);

            auto offset_values = 0;
            for (uint64_t i = 0; i < bp.words; ++i) {
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

        for (int64_t i = bp.trailing_bit_offset; i < length; ++i) {
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
    sse_contiguous(Array_ptr array1, Array_ptr array2) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto dwn_array1 = std::static_pointer_cast<ArrayType>(array1);
        auto dwn_array2 = std::static_pointer_cast<ArrayType>(array2);

        auto raw_values1 = dwn_array1->raw_values();
        auto raw_values2 = dwn_array2->raw_values();

        auto accum = 0;
        for (int i = 0; i < dwn_array1->length(); ++i) {
            auto error = raw_values1[i] - raw_values2[i];
            accum += error*error;
        }
        return accum;
    }

    template<typename ArrowType, bool aligned_array1, bool aligned_array2,
            typename = std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD>>
    typename ArrowType::c_type
    alignment_sse_contiguous(std::shared_ptr<arrow::NumericArray<ArrowType>> array1,
                                  std::shared_ptr<arrow::NumericArray<ArrowType>> array2,
                                  SimdProperties p) {
        using CType = typename ArrowType::c_type;
        auto raw_values1 = array1->raw_values();
        auto raw_values2 = array2->raw_values();

        CType accum = 0;
        for (uint64_t i = 0; i < p.leading_elements; ++i) {
            auto error = raw_values1[i] - raw_values2[i];
            accum += error*error;
        }

        auto simd_accum = SimdTraits<ArrowType>::simd_set1(0);
        for(uint64_t w = 0, offset = p.leading_elements; w < p.aligned_words; ++w, offset += SimdTraits<ArrowType>::LANES) {
            auto l1 = [=]() -> auto {
                if constexpr(aligned_array1) return SimdTraits<ArrowType>::simd_load(raw_values1 + offset);
                else return SimdTraits<ArrowType>::simd_loadu(raw_values1 + offset);
            }();

            auto l2 = [=]() -> auto {
                if constexpr(aligned_array2) return SimdTraits<ArrowType>::simd_load(raw_values2 + offset);
                else return SimdTraits<ArrowType>::simd_loadu(raw_values2 + offset);
            }();

            auto d = SimdTraits<ArrowType>::simd_sub(l1, l2);
            simd_accum = SimdTraits<ArrowType>::simd_add(simd_accum, SimdTraits<ArrowType>::simd_mul(d,d));
        }

        accum += SimdTraits<ArrowType>::horizontal_sum(simd_accum);

        uint64_t length = array1->length();
        for (uint64_t i = length - p.trailing_elements; i < length; ++i) {
            auto error = raw_values1[i] - raw_values2[i];
            accum += error*error;
        }

        return accum;
    }

    template<typename ArrowType,
            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sse_contiguous(Array_ptr array1, Array_ptr array2) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto dwn_array1 = std::static_pointer_cast<ArrayType>(array1);
        auto dwn_array2 = std::static_pointer_cast<ArrayType>(array2);

        auto raw_values1 = dwn_array1->raw_values();
        auto raw_values2 = dwn_array2->raw_values();

        auto p1 = simd_properties<ArrowType>(raw_values1, dwn_array1->length());
        auto p2 = simd_properties<ArrowType>(raw_values2, dwn_array2->length());

        if (p1.leading_elements == p2.leading_elements) {
            return alignment_sse_contiguous<ArrowType, true, true>(dwn_array1, dwn_array2, p1);
        } else {
            return alignment_sse_contiguous<ArrowType, true, false>(dwn_array1, dwn_array2, p1);
        }
    }

    template<typename ArrowType>
    typename ArrowType::c_type
    sse_non_contiguous(Array_ptr array1, Array_ptr array2, Buffer_ptr bitmap) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using CType = typename ArrowType::c_type;

        auto dwn_array1 = std::static_pointer_cast<ArrayType>(array1);
        auto raw_values1 = dwn_array1->raw_values();

        auto dwn_array2 = std::static_pointer_cast<ArrayType>(array2);
        auto raw_values2 = dwn_array2->raw_values();

        auto null_bitmap = bitmap->data();

        auto length = dwn_array1->length();

        const auto bp = util::bit_util::bitmap_words<64>(length);

        CType accum = 0;

        if (bp.words > 0) {
            const uint64_t *u64_bitmap = reinterpret_cast<const uint64_t *>(null_bitmap);

            auto offset_values = 0;
            for (uint64_t i = 0; i < bp.words; ++i) {
                if (u64_bitmap[i] == 0xFFFFFFFFFFFFFFFF) {
                    for (auto j = 0; j < 64; ++j) {
                        auto error = raw_values1[offset_values + j] - raw_values2[offset_values + j];
                        accum += error*error;
                    }
                } else {
                    for (auto j = 0; j < 64; ++j) {
                        if (GetBit(null_bitmap, offset_values + j)) {
                            auto error = raw_values1[offset_values + j] - raw_values2[offset_values + j];
                            accum += error*error;
                        }
                    }
                }
                offset_values += 64;
            }
        }

        for (int64_t i = bp.trailing_bit_offset; i < length; ++i) {
            if (GetBit(null_bitmap, i)) {
                auto error = raw_values1[i] - raw_values2[i];
                accum += error*error;
            }
        }

        return accum;
    }

    template<typename ArrowType,
            std::enable_if_t<!SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    sse_covariance_contiguous(Array_ptr array1, Array_ptr array2,
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

    template<typename ArrowType, bool aligned_array1, bool aligned_array2,
            std::enable_if_t<SimdTraits<ArrowType>::CAN_SIMD, int> = 0>
    typename ArrowType::c_type
    alignment_sse_covariance_contiguous(std::shared_ptr<arrow::NumericArray<ArrowType>> array1,
                                             std::shared_ptr<arrow::NumericArray<ArrowType>> array2,
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
            auto l1 = [=]() -> auto {
                if constexpr(aligned_array1) return SimdTraits<ArrowType>::simd_load(raw_values1 + offset);
                else return SimdTraits<ArrowType>::simd_loadu(raw_values1 + offset);
            }();

            auto l2 = [=]() -> auto {
                if constexpr(aligned_array2) return SimdTraits<ArrowType>::simd_load(raw_values2 + offset);
                else return SimdTraits<ArrowType>::simd_loadu(raw_values2 + offset);
            }();

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
    sse_covariance_contiguous(Array_ptr array1,
                          Array_ptr array2,
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
            return alignment_sse_covariance_contiguous<ArrowType, true, true>(dwn_array1, dwn_array2, mean1, mean2, p1);
        } else {
            return alignment_sse_covariance_contiguous<ArrowType, true, false>(dwn_array1, dwn_array2, mean1, mean2, p1);
        }
    }

    template<typename ArrowType>
    typename ArrowType::c_type
    sse_covariance_non_contiguous(Array_ptr array1,
                                  Array_ptr array2,
                                  typename ArrowType::c_type mean1,
                                  typename ArrowType::c_type mean2,
                                  Buffer_ptr bitmap) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using CType = typename ArrowType::c_type;

        auto dwn_array1 = std::static_pointer_cast<ArrayType>(array1);
        auto raw_values1 = dwn_array1->raw_values();

        auto dwn_array2 = std::static_pointer_cast<ArrayType>(array2);
        auto raw_values2 = dwn_array2->raw_values();

        auto null_bitmap = bitmap->data();

        auto length = dwn_array1->length();

        const auto bp = util::bit_util::bitmap_words<64>(length);

        CType accum = 0;

        if (bp.words > 0) {
            const uint64_t *u64_bitmap = reinterpret_cast<const uint64_t *>(null_bitmap);

            auto offset_values = 0;
            for (uint64_t i = 0; i < bp.words; ++i) {
                if (u64_bitmap[i] == 0xFFFFFFFFFFFFFFFF) {
                    for (auto j = 0; j < 64; ++j) {
                        accum += (raw_values1[offset_values + j] - mean1) * (raw_values2[offset_values + j] - mean2);
                    }
                } else {
                    for (auto j = 0; j < 64; ++j) {
                        if (GetBit(null_bitmap, offset_values + j)) {
                            accum += (raw_values1[offset_values + j] - mean1) * (raw_values2[offset_values + j] - mean2);
                        }
                    }
                }
                offset_values += 64;
            }
        }

        for (int64_t i = bp.trailing_bit_offset; i < length; ++i) {
            if (GetBit(null_bitmap, i)) {
                accum += (raw_values1[i] - mean1) * (raw_values2[i] - mean2);
            }
        }

        return accum;
    }

}

namespace simd {

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    var(Array_ptr array) {
        typename ArrowType::c_type mean = simd::mean<ArrowType>(array);

        auto null_count = array->null_count();
        if (null_count == 0) {
            return static_cast<Output>(simd::var_internals::sse_contiguous<ArrowType>(array, mean))
                                / (array->length() - 1);
        } else {
            return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean, array->null_bitmap()))
                                                                                / (array->length() - null_count - 1);
        }
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    var(Array_ptr array, Buffer_ptr bitmap) {
        typename ArrowType::c_type mean = simd::mean<ArrowType>(array, bitmap);
        auto non_null = arrow::internal::CountSetBits(bitmap->data(), 0, array->length());

        return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean, bitmap))
                                                                                    / (non_null - 1);
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    var(Array_ptr array, typename ArrowType::c_type mean) {
        auto null_count = array->null_count();
        if (null_count == 0) {
            return static_cast<Output>(simd::var_internals::sse_contiguous<ArrowType>(array, mean))
                                                                                        / (array->length() - 1);
        } else {
            return
            static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean, array->null_bitmap()))
                                                                                 / (array->length() - null_count - 1);
        }
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    var(Array_ptr array, typename ArrowType::c_type mean, Buffer_ptr bitmap) {
        auto non_null = arrow::internal::CountSetBits(bitmap->data(), 0, array->length());
        return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean, bitmap))
                                                                                    / (non_null - 1);
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    sse(Array_ptr array) {
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
    sse(Array_ptr array, Buffer_ptr bitmap) {
        typename ArrowType::c_type mean = simd::mean<ArrowType>(array, bitmap);
        return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean, bitmap));
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    sse(Array_ptr array, typename ArrowType::c_type mean) {
        auto null_count = array->null_count();
        if (null_count == 0) {
            return static_cast<Output>(simd::var_internals::sse_contiguous<ArrowType>(array, mean));
        } else {
            return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean));
        }
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    sse(Array_ptr array, typename ArrowType::c_type mean, Buffer_ptr bitmap) {
        return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array, mean, bitmap));
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    sse(Array_ptr array1, Array_ptr array2) {
        auto null_count1 = array1->null_count();
        auto null_count2 = array2->null_count();

        if (null_count1 == 0 && null_count2 == 0) {
            return static_cast<Output>(simd::var_internals::sse_contiguous<ArrowType>(array1, array2));
        } else {
            auto combined_bitmap =
                    util::bit_util::combined_bitmap(array1->null_bitmap(), array2->null_bitmap(), array1->length());
            return
            static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array1, array2, combined_bitmap));
        }
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    sse(Array_ptr array1, Array_ptr array2, Buffer_ptr bitmap) {
        return static_cast<Output>(simd::var_internals::sse_non_contiguous<ArrowType>(array1, array2, bitmap));
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    covariance(Array_ptr array1, Array_ptr array2,
                typename ArrowType::c_type mean1, typename ArrowType::c_type mean2) {
        auto null_count1 = array1->null_count();
        auto null_count2 = array2->null_count();
        auto length = array1->length();

        if (null_count1 == 0 && null_count2 == 0) {
            return
            static_cast<Output>(simd::var_internals::sse_covariance_contiguous<ArrowType>(array1, array2, mean1, mean2))
                                                                                    / (length - 1);
        } else {
            auto combined_bitmap = util::bit_util::combined_bitmap(array1->null_bitmap(), array2->null_bitmap(), length);

            uint64_t combined_count = util::bit_util::non_null_count(combined_bitmap, length);
            return
            static_cast<Output>(
                    simd::var_internals::sse_covariance_non_contiguous<ArrowType>(array1, array2,
                                                                                mean1, mean2, combined_bitmap))
                                                                                / (combined_count - 1);
        }
    }

    template<typename ArrowType, typename Output = typename ArrowType::c_type>
    Output
    covariance(Array_ptr array1, Array_ptr array2,
               typename ArrowType::c_type mean1, typename ArrowType::c_type mean2, Buffer_ptr bitmap) {

        uint64_t non_null = util::bit_util::non_null_count(bitmap, array1->length());
        return static_cast<Output>(
                simd::var_internals::sse_covariance_non_contiguous<ArrowType>(array1, array2, mean1, mean2, bitmap))
                                                                                / (non_null - 1);
    }
}

#endif //PGM_DATASET_VARIANCE_HPP
