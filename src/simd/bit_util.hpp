#ifndef PGM_DATASET_BIT_UTIL_HPP
#define PGM_DATASET_BIT_UTIL_HPP

#include <cstdint>
#include <arrow/api.h>

typedef std::shared_ptr<arrow::Array> Array_ptr;

namespace simd::bit_util {

    struct BitMapWords {
        uint64_t words;
        uint64_t trailing_bits;
        uint64_t trailing_bit_offset;
    };

    template<uint64_t WORD_BIT_SIZE>
    BitMapWords bitmap_words(uint64_t length) {
        auto words = length / WORD_BIT_SIZE;
        auto trailing = length % WORD_BIT_SIZE;

        return BitMapWords {
                .words = words,
                .trailing_bits = trailing,
                .trailing_bit_offset = length - trailing
        };
    }

    uint64_t null_count(std::vector<Array_ptr> columns);
    uint64_t null_count(std::shared_ptr<arrow::Buffer> bitmap, uint64_t length);
    std::shared_ptr<arrow::Buffer> combined_bitmap(std::vector<Array_ptr> columns);
    std::shared_ptr<arrow::Buffer> combined_bitmap_with_null(std::vector<Array_ptr> columns);

    uint64_t count_combined_set_bits(const uint8_t* bitmap1, const uint8_t* bitmap2, uint64_t length);
}

#endif //PGM_DATASET_BIT_UTIL_HPP
