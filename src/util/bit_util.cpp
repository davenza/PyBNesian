
#include <util/bit_util.hpp>
#include <arrow/api.h>

namespace util::bit_util {

    using arrow::BitUtil::GetBit;

    typedef std::shared_ptr<arrow::Array> Array_ptr;
    typedef std::shared_ptr<arrow::Buffer> Buffer_ptr;

    uint64_t null_count(std::vector<Array_ptr> columns) {
        int64_t null_count = 0;
        for (Array_ptr column : columns) {
            null_count += column->null_count();
        }
        return null_count;
    }

    Buffer_ptr combined_bitmap_with_null(std::vector<Array_ptr> columns) {
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

    Buffer_ptr combined_bitmap(std::vector<Array_ptr> columns) {
        if (null_count(columns) > 0) {
            return combined_bitmap_with_null(columns);
        } else {
            return nullptr;
        }
    }

    Buffer_ptr combined_bitmap(Buffer_ptr bitmap1, Buffer_ptr bitmap2, uint64_t length) {
        if(bitmap1) {
            if(bitmap2) {
                auto res = arrow::Buffer::Copy(bitmap1, arrow::default_cpu_memory_manager()).ValueOrDie();
                arrow::internal::BitmapAnd(bitmap1->data(), 0,
                                           bitmap2->data(), 0,
                                           length,
                                           0, res->mutable_data());
                return res;
            } else {
                return bitmap1;
            }
        } else {
            if (bitmap2) {
                return bitmap2;
            } else {
                return nullptr;
            }
        }
    }

    uint64_t null_count(Buffer_ptr bitmap, uint64_t length) {
        return length - arrow::internal::CountSetBits(bitmap->data(), 0, length);
    }

    uint64_t non_null_count(Buffer_ptr bitmap, uint64_t length) {
        return arrow::internal::CountSetBits(bitmap->data(), 0, length);
    }
}