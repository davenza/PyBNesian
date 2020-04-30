#ifndef PGM_DATASET_SIMD_PROPERTIES_HPP
#define PGM_DATASET_SIMD_PROPERTIES_HPP


namespace simd {

    template<typename ArrowType>
    struct SimdTraits {
        static constexpr bool CAN_SIMD = false;
    };

    template<>
    struct SimdTraits<arrow::DoubleType> {
        static constexpr bool CAN_SIMD = true;
        using AVX_TYPE = __m256d;
        static constexpr int64_t LANES = 4;
        static constexpr uintptr_t ALIGNMENT = 32;
        static constexpr auto simd_load = _mm256_load_pd;
        static constexpr auto simd_set1 = _mm256_set1_pd;
        static constexpr auto simd_add = _mm256_add_pd;
        static constexpr auto simd_sub = _mm256_sub_pd;
        static constexpr auto simd_mul = _mm256_mul_pd;

        inline static arrow::DoubleType::c_type horizontal_sum(__m256d simd) {
            auto high = _mm256_extractf128_pd(simd, 1);
            auto simd_sum = _mm_add_pd(high, _mm256_castpd256_pd128(simd));
            simd_sum = _mm_hadd_pd(simd_sum, simd_sum);

            return _mm_cvtsd_f64(simd_sum);
        }
    };

    template<>
    struct SimdTraits<arrow::FloatType> {
        static constexpr bool CAN_SIMD = true;
        using AVX_TYPE = __m256;
        static constexpr int64_t LANES = 8;
        static constexpr uintptr_t ALIGNMENT = 32;
        static constexpr auto simd_load = _mm256_load_ps;
        static constexpr auto simd_set1 = _mm256_set1_ps;
        static constexpr auto simd_add = _mm256_add_ps;
        static constexpr auto simd_sub = _mm256_sub_ps;
        static constexpr auto simd_mul = _mm256_mul_ps;

//    Implementation from here: https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
        inline static arrow::FloatType::c_type horizontal_sum(__m256 simd) {
            auto high_quad = _mm256_extractf128_ps(simd, 1);
            auto low_quad = _mm256_castps256_ps128(simd);
            auto addition_quad = _mm_add_ps(high_quad, low_quad);
            auto low_dual = addition_quad;
            auto high_dual = _mm_movehl_ps(addition_quad, addition_quad);
            auto sum_dual = _mm_add_ps(low_dual, high_dual);
            auto low_single = sum_dual;
            auto high_single = _mm_shuffle_ps(sum_dual, sum_dual, 1);
            auto final_sum = _mm_add_ss(low_single, high_single);
            return _mm_cvtss_f32(final_sum);
        }
    };

    template<>
    struct SimdTraits<arrow::Int64Type> {
        static constexpr bool CAN_SIMD = true;
        using AVX_TYPE = __m256i;
        static constexpr int64_t LANES = 4;
        static constexpr uintptr_t ALIGNMENT = 32;
        static constexpr auto simd_set1 = _mm256_set1_epi64x;
        static constexpr auto simd_add = _mm256_add_epi64;

        inline static __m256i simd_load(const arrow::Int64Type::c_type *mem) {
            return _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(mem));
        }

        inline static arrow::Int64Type::c_type horizontal_sum(__m256i simd) {
            auto high = _mm256_extractf128_si256(simd, 1);
            auto simd_sum = _mm_add_epi64(high, _mm256_castsi256_si128(simd));
            simd_sum = _mm_add_epi64(simd_sum, _mm_srli_si128(simd_sum, 8));
            return _mm_cvtsi128_si64(simd_sum);
        }
    };

    template<>
    struct SimdTraits<arrow::UInt64Type> {
        static constexpr bool CAN_SIMD = true;
        using AVX_TYPE = __m256i;
        static constexpr int64_t LANES = 4;
        static constexpr uintptr_t ALIGNMENT = 32;
        static constexpr auto simd_set1 = _mm256_set1_epi64x;
        static constexpr auto simd_add = _mm256_add_epi64;

        inline static __m256i simd_load(const arrow::UInt64Type::c_type *mem) {
            return _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(mem));
        }

        inline static arrow::UInt64Type::c_type horizontal_sum(__m256i simd) {
            auto high = _mm256_extractf128_si256(simd, 1);
            auto simd_sum = _mm_add_epi64(high, _mm256_castsi256_si128(simd));
            simd_sum = _mm_add_epi64(simd_sum, _mm_srli_si128(simd_sum, 8));
            return _mm_cvtsi128_si64(simd_sum);
        }
    };

    template<>
    struct SimdTraits<arrow::Int32Type> {
        static constexpr bool CAN_SIMD = true;
        using AVX_TYPE = __m256i;
        static constexpr int64_t LANES = 8;
        static constexpr uintptr_t ALIGNMENT = 32;
        static constexpr auto simd_set1 = _mm256_set1_epi32;
        static constexpr auto simd_add = _mm256_add_epi32;

        inline static __m256i simd_load(const arrow::Int32Type::c_type *mem) {
            return _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(mem));
        }

        inline static arrow::Int32Type::c_type horizontal_sum(__m256i simd) {
            auto high_quad = _mm256_extractf128_si256(simd, 1);
            auto sum_quad = _mm_add_epi32(high_quad, _mm256_castsi256_si128(simd));
            auto sum_dual = _mm_add_epi32(sum_quad, _mm_srli_si128(sum_quad, 8));
            auto final_sum = _mm_add_epi32(sum_dual, _mm_shuffle_epi32(sum_dual, 1));
            return _mm_cvtsi128_si32(final_sum);
        }
    };

    template<>
    struct SimdTraits<arrow::UInt32Type> {
        static constexpr bool CAN_SIMD = true;
        using AVX_TYPE = __m256i;
        static constexpr int64_t LANES = 8;
        static constexpr uintptr_t ALIGNMENT = 32;
        static constexpr auto simd_set1 = _mm256_set1_epi32;
        static constexpr auto simd_add = _mm256_add_epi32;

        inline static __m256i simd_load(const arrow::UInt32Type::c_type *mem) {
            return _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(mem));
        }

        inline static arrow::UInt32Type::c_type horizontal_sum(__m256i simd) {
            auto high_quad = _mm256_extractf128_si256(simd, 1);
            auto sum_quad = _mm_add_epi32(high_quad, _mm256_castsi256_si128(simd));
            auto sum_dual = _mm_add_epi32(sum_quad, _mm_srli_si128(sum_quad, 8));
            auto final_sum = _mm_add_epi32(sum_dual, _mm_shuffle_epi32(sum_dual, 1));
            return _mm_cvtsi128_si32(final_sum);
        }
    };

    template<>
    struct SimdTraits<arrow::Int16Type> {
        static constexpr bool CAN_SIMD = true;
        using AVX_TYPE = __m256i;
        static constexpr int64_t LANES = 16;
        static constexpr uintptr_t ALIGNMENT = 32;
        static constexpr auto simd_set1 = _mm256_set1_epi16;
        static constexpr auto simd_add = _mm256_add_epi16;

        inline static __m256i simd_load(const arrow::Int16Type::c_type *mem) {
            return _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(mem));
        }

        inline static arrow::Int16Type::c_type horizontal_sum(__m256i simd) {
            auto high_oct = _mm256_extractf128_si256(simd, 1);
            auto sum_oct = _mm_add_epi16(high_oct, _mm256_castsi256_si128(simd));
            auto sum_quad = _mm_add_epi16(sum_oct, _mm_srli_si128(sum_oct, 8));
            auto sum_dual = _mm_add_epi16(sum_quad, _mm_srli_si128(sum_quad, 4));
            auto final_sum = _mm_add_epi16(sum_dual, _mm_srli_si128(sum_dual, 2));
            return _mm_extract_epi16(final_sum, 0);
        }
    };

    template<>
    struct SimdTraits<arrow::UInt16Type> {
        static constexpr bool CAN_SIMD = true;
        using AVX_TYPE = __m256i;
        static constexpr int64_t LANES = 16;
        static constexpr uintptr_t ALIGNMENT = 32;
        static constexpr auto simd_set1 = _mm256_set1_epi16;
        static constexpr auto simd_add = _mm256_add_epi16;

        inline static __m256i simd_load(const arrow::UInt16Type::c_type *mem) {
            return _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(mem));
        }

        inline static arrow::UInt16Type::c_type horizontal_sum(__m256i simd) {
            auto high_oct = _mm256_extractf128_si256(simd, 1);
            auto sum_oct = _mm_add_epi16(high_oct, _mm256_castsi256_si128(simd));
            auto sum_quad = _mm_add_epi16(sum_oct, _mm_srli_si128(sum_oct, 8));
            auto sum_dual = _mm_add_epi16(sum_quad, _mm_srli_si128(sum_quad, 4));
            auto final_sum = _mm_add_epi16(sum_dual, _mm_srli_si128(sum_dual, 2));
            return _mm_extract_epi16(final_sum, 0);
        }
    };

    template<>
    struct SimdTraits<arrow::Int8Type> {
        static constexpr bool CAN_SIMD = true;
        using AVX_TYPE = __m256i;
        static constexpr int64_t LANES = 32;
        static constexpr uintptr_t ALIGNMENT = 32;
        static constexpr auto simd_set1 = _mm256_set1_epi8;
        static constexpr auto simd_add = _mm256_add_epi8;

        inline static __m256i simd_load(const arrow::Int8Type::c_type *mem) {
            return _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(mem));
        }

        inline static arrow::Int8Type::c_type horizontal_sum(__m256i simd) {
            auto high_hex = _mm256_extractf128_si256(simd, 1);
            auto sum_hex = _mm_add_epi8(high_hex, _mm256_castsi256_si128(simd));
            auto sum_oct = _mm_add_epi8(sum_hex, _mm_srli_si128(sum_hex, 8));
            auto sum_quad = _mm_add_epi8(sum_oct, _mm_srli_si128(sum_oct, 4));
            auto sum_dual = _mm_add_epi8(sum_quad, _mm_srli_si128(sum_quad, 2));
            auto final_sum = _mm_add_epi8(sum_dual, _mm_srli_si128(sum_dual, 1));
            return _mm_extract_epi8(final_sum, 0);
        }
    };

    template<>
    struct SimdTraits<arrow::UInt8Type> {
        static constexpr bool CAN_SIMD = true;
        using AVX_TYPE = __m256i;
        static constexpr int64_t LANES = 32;
        static constexpr uintptr_t ALIGNMENT = 32;
        static constexpr auto simd_set1 = _mm256_set1_epi8;
        static constexpr auto simd_add = _mm256_add_epi8;

        inline static __m256i simd_load(const arrow::UInt8Type::c_type *mem) {
            return _mm256_stream_load_si256(reinterpret_cast<const __m256i *>(mem));
        }

        inline static arrow::UInt8Type::c_type horizontal_sum(__m256i simd) {
            auto high_hex = _mm256_extractf128_si256(simd, 1);
            auto sum_hex = _mm_add_epi8(high_hex, _mm256_castsi256_si128(simd));
            auto sum_oct = _mm_add_epi8(sum_hex, _mm_srli_si128(sum_hex, 8));
            auto sum_quad = _mm_add_epi8(sum_oct, _mm_srli_si128(sum_oct, 4));
            auto sum_dual = _mm_add_epi8(sum_quad, _mm_srli_si128(sum_quad, 2));
            auto final_sum = _mm_add_epi8(sum_dual, _mm_srli_si128(sum_dual, 1));
            return _mm_extract_epi8(final_sum, 0);
        }
    };


    struct SimdProperties {
        int64_t leading_elements;
        int64_t aligned_words;
        int64_t trailing_elements;
    };

    template<typename ArrowType>
    inline SimdProperties simd_properties(const typename ArrowType::c_type *data, int64_t length) {
        constexpr auto ALIGNMENT = SimdTraits<ArrowType>::ALIGNMENT;
        constexpr auto LANES = SimdTraits<ArrowType>::LANES;
        uintptr_t iptr = reinterpret_cast<uintptr_t>(data);

        int64_t leading_elements = std::min<int64_t>(length, (iptr % ALIGNMENT) / sizeof(typename ArrowType::c_type));
        int64_t words = (length - leading_elements) / LANES;
        int64_t trailing = (length - leading_elements) % LANES;

        return SimdProperties{
                .leading_elements = leading_elements,
                .aligned_words = words,
                .trailing_elements = trailing,
        };
    }

}

#endif //PGM_DATASET_SIMD_PROPERTIES_HPP
