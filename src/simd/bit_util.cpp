
#include <simd/bit_util.hpp>
#include <simd/simd_properties.hpp>
#include <arrow/api.h>

namespace simd::bit_util {

    using arrow::BitUtil::GetBit;

    typedef std::shared_ptr<arrow::Array> Array_ptr;

    uint64_t null_count(std::vector<Array_ptr> columns) {
        int64_t null_count = 0;
        for (Array_ptr column : columns) {
            null_count += column->null_count();
        }
        return null_count;
    }

    std::shared_ptr<arrow::Buffer> combined_bitmap_with_null(std::vector<Array_ptr> columns) {
        int first_col_idx = 0;

        auto length = columns[0]->length();

        for(uint64_t i = 0; i < columns.size(); ++i) {
            auto col = columns[i];
            if (col->null_count()) {
                first_col_idx = i;
                break;
            }
        }

        auto res = arrow::Buffer::Copy(columns[first_col_idx]->null_bitmap(), arrow::default_cpu_memory_manager());
        auto bitmap = res.ValueOrDie();

        for(uint64_t i = first_col_idx + 1; i < columns.size(); ++i) {
            auto col = columns[i];

            if (col->null_count()) {
                auto other_bitmap = col->null_bitmap();

                arrow::internal::BitmapAnd(bitmap->data(), 0,
                                           other_bitmap->data(), 0,
                                           length,
                                           0, bitmap->mutable_data());
            }
        }

        return bitmap;
    }

    std::shared_ptr<arrow::Buffer> combined_bitmap(std::vector<Array_ptr> columns) {
        if (null_count(columns) > 0) {
            return combined_bitmap_with_null(columns);
        } else {
            return nullptr;
        }
    }

    uint64_t null_count(std::shared_ptr<arrow::Buffer> bitmap, uint64_t length) {
        return length - arrow::internal::CountSetBits(bitmap->data(), 0, length);
    }

    uint64_t count_combined_set_bits(const uint8_t* bitmap1, const uint8_t* bitmap2, uint64_t length) {

        auto p = bitmap_words<64>(length);

        uint64_t count = 0;

        auto u64_bitmap1 = reinterpret_cast<const uint64_t*>(bitmap1);
        auto u64_bitmap2 = reinterpret_cast<const uint64_t*>(bitmap2);

        if (p.words > 0) {
            for(uint64_t w = 0; w < p.words; ++w) {
                uint64_t comb_bitmap = u64_bitmap1[w] & u64_bitmap2[w];
                count += _mm_popcnt_u64(comb_bitmap);
            }
        }

        for(uint64_t i = p.trailing_bit_offset; i < length; ++i) {
            if (GetBit(bitmap1, i) && GetBit(bitmap2, i)) {
                ++count;
            }
        }

        return count;
    }
}