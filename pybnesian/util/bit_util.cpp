#include <arrow/util/bitmap_ops.h>
#include <arrow/api.h>

namespace util::bit_util {

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
    size_t first_col_idx = 0;

    auto length = columns[0]->length();

    for (size_t i = 0, num_columns = columns.size(); i < num_columns; ++i) {
        auto col = columns[i];
        if (col->null_count()) {
            first_col_idx = i;
            break;
        }
    }

    auto res = arrow::Buffer::Copy(columns[first_col_idx]->null_bitmap(), arrow::default_cpu_memory_manager());
    auto bitmap = res.ValueOrDie();

    for (uint64_t i = first_col_idx + 1, num_columns = columns.size(); i < num_columns; ++i) {
        auto col = columns[i];

        if (col->null_count()) {
            auto other_bitmap = col->null_bitmap();

            arrow::internal::BitmapAnd(bitmap->data(), 0, other_bitmap->data(), 0, length, 0, bitmap->mutable_data());
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
    if (bitmap1) {
        if (bitmap2) {
            auto res = arrow::Buffer::Copy(bitmap1, arrow::default_cpu_memory_manager()).ValueOrDie();
            arrow::internal::BitmapAnd(bitmap1->data(), 0, bitmap2->data(), 0, length, 0, res->mutable_data());
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

// Extracted from arrow/util/bit_util.h
int next_power2(int value) {
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;
    return value;
}

int previous_power2(int value) {
    value |= (value >> 1);
    value |= (value >> 2);
    value |= (value >> 4);
    value |= (value >> 8);
    value |= (value >> 16);
    return value - (value >> 1);
}

}  // namespace util::bit_util