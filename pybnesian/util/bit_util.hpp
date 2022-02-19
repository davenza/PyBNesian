#ifndef PYBNESIAN_UTIL_BIT_UTIL_HPP
#define PYBNESIAN_UTIL_BIT_UTIL_HPP

#include <cstdint>
#include <arrow/api.h>

using Array_ptr = std::shared_ptr<arrow::Array>;
using Buffer_ptr = std::shared_ptr<arrow::Buffer>;

namespace util::bit_util {

struct BitMapWords {
    uint64_t words;
    uint64_t trailing_bits;
    uint64_t trailing_bit_offset;
};

template <uint64_t WORD_BIT_SIZE>
BitMapWords bitmap_words(uint64_t length) {
    auto words = length / WORD_BIT_SIZE;
    auto trailing = length % WORD_BIT_SIZE;

    return BitMapWords{.words = words, .trailing_bits = trailing, .trailing_bit_offset = length - trailing};
}

uint64_t null_count(std::vector<Array_ptr> columns);
uint64_t null_count(Buffer_ptr bitmap, uint64_t length);
uint64_t non_null_count(Buffer_ptr bitmap, uint64_t length);

Buffer_ptr combined_bitmap(std::vector<Array_ptr> columns);
Buffer_ptr combined_bitmap(Buffer_ptr bitmap1, Buffer_ptr bitmap2, uint64_t length);
Buffer_ptr combined_bitmap_with_null(std::vector<Array_ptr> columns);

// Extracted from arrow/util/bit_util.h
int next_power2(int value);
int previous_power2(int value);

#if ARROW_VERSION_MAJOR >= 7
using arrow::bit_util::GetBit;
#else
using arrow::BitUtil::GetBit;
#endif

}  // namespace util::bit_util

#endif  // PYBNESIAN_UTIL_BIT_UTIL_HPP
