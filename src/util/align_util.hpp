
#ifndef PGM_DATASET_ALIGN_UTIL_HPP
#define PGM_DATASET_ALIGN_UTIL_HPP

#include <arrow/util/bit_util.h>
#include <arrow/util/align_util.h>

// Adapted from arrow's cpp/util/align_util
namespace bit_util {


    template <uint64_t ALIGN_IN_BYTES, uint64_t WORD_BIT_LENGTH = ALIGN_IN_BYTES*8>
    inline arrow::internal::BitmapWordAlignParams BitmapWordAlignExtra(const uint8_t* data, int64_t bit_offset,
                                                 int64_t length) {
        static_assert(arrow::BitUtil::IsPowerOf2(ALIGN_IN_BYTES),
                      "ALIGN_IN_BYTES should be a positive power of two");
        constexpr uint64_t ALIGN_IN_BITS = ALIGN_IN_BYTES * 8;
        static_assert(WORD_BIT_LENGTH % ALIGN_IN_BITS == 0,
                      "WORD_LENGTH should be a multiple of ALIGN_IN_BYTES");


        arrow::internal::BitmapWordAlignParams p;

        // Compute a "bit address" that we can align up to ALIGN_IN_BITS.
        // We don't care about losing the upper bits since we are only interested in the
        // difference between both addresses.
        const uint64_t bit_addr =
                reinterpret_cast<size_t>(data) * 8 + static_cast<uint64_t>(bit_offset);
        const uint64_t aligned_bit_addr = arrow::BitUtil::RoundUpToPowerOf2(bit_addr, ALIGN_IN_BITS);

        std::cout << "bit_addr: " << bit_addr << std::endl;
        std::cout << "Aligned_bit_addr: " << aligned_bit_addr << std::endl;

        p.leading_bits = std::min<int64_t>(length, aligned_bit_addr - bit_addr);
        p.aligned_words = (length - p.leading_bits) / WORD_BIT_LENGTH;
        p.aligned_bits = p.aligned_words * WORD_BIT_LENGTH;
        p.trailing_bits = length - p.leading_bits - p.aligned_bits;
        p.trailing_bit_offset = bit_offset + p.leading_bits + p.aligned_bits;

        p.aligned_start = data + (bit_offset + p.leading_bits) / 8;
        return p;
    }

}

#endif //PGM_DATASET_ALIGN_UTIL_HPP
