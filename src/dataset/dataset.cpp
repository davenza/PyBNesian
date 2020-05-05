#include <iostream>
#include <arrow/api.h>
#include <arrow/util/align_util.h>
#include <arrow/util/bit_util.h>
#include <arrow/python/pyarrow.h>
#include <dataset/dataset.hpp>
#include <Eigen/Dense>

using Eigen::MatrixXd;

namespace pyarrow = arrow::py;
namespace py = pybind11;
using arrow::Array, arrow::RecordBatch, arrow::Result, arrow::Buffer, arrow::NumericBuilder, arrow::DataType,
    arrow::Type, arrow::NumericBuilder;

namespace dataset {

    bool is_pandas_dataframe(py::handle pyobject) {
        PyObject* pyobj_ptr = pyobject.ptr();

        PyObject* module = PyImport_ImportModule("pandas");
        PyObject* moduleDict = PyModule_GetDict(module);
        PyObject* protocolClass = PyDict_GetItemString(moduleDict, "DataFrame");

        return PyObject_IsInstance(pyobj_ptr, protocolClass);
    }

    py::object pandas_to_pyarrow_record_batch(py::handle pyobject) {
        auto d = py::module::import("pyarrow").attr("RecordBatch").attr("from_pandas")(pyobject);
        return d;
    }

    std::shared_ptr<RecordBatch> to_record_batch(py::handle data) {

        PyObject* py_ptr = data.ptr();

        if (pyarrow::is_batch(py_ptr)) {
            auto result = pyarrow::unwrap_batch(py_ptr);
            if (result.ok()) {
                return result.ValueOrDie();
            } else {
                throw std::runtime_error("pyarrow's RecordBatch could not be converted.");
            }
        }
        else if (is_pandas_dataframe(data)) {
            auto a = pandas_to_pyarrow_record_batch(data);
            auto result = pyarrow::unwrap_batch(a.ptr());

            if (result.ok()) {
                return result.ValueOrDie();
            } else {
                throw std::runtime_error("pyarrow's RecordBatch could not be converted.");
            }
        }
        else {
            throw std::invalid_argument("\'data\' parameter should be a pyarrow's RecordBatch or a pandas DataFrame. ");
        }

        return nullptr;
    }

    template<typename ArrowType>
    std::shared_ptr<Array> copy_array(std::shared_ptr<arrow::NumericArray<ArrowType>> ar,
                    std::shared_ptr<Buffer> bitmap,
                    uint64_t n_rows) {

        using arrow::BitUtil::GetBit;
        using arrow::internal::BitmapWordAlign;

        NumericBuilder<ArrowType> builder;

        builder.Resize(n_rows);

        auto bitmap_data = bitmap->data();
        const auto p = BitmapWordAlign<8>(bitmap_data, 0, n_rows);

        for(int64_t i = 0; i < p.leading_bits; ++i) {
            if(GetBit(bitmap_data, i)) {
                builder.UnsafeAppend(ar->Value(i));
            }
        }

        if (p.aligned_words > 0) {
            const uint64_t* u64_bitmap = reinterpret_cast<const uint64_t*>(p.aligned_start);

            const uint64_t* end = u64_bitmap + p.aligned_words;

            int64_t offset = p.leading_bits;
            for(const uint64_t* i = u64_bitmap; i < end; ++i, offset += 64) {

                if (*i == 0xFFFFFFFFFFFFFFFF) {
                    builder.AppendValues(ar->raw_values() + offset, 64);
                } else {
                    auto u8_buf = bitmap_data + (i-u64_bitmap)*8;
                    for(int64_t j = 0; j < 64; ++j) {
                        if(GetBit(u8_buf, j)) {
                            builder.UnsafeAppend(ar->Value(offset+j));
                        }
                    }
                }
            }
        }

        for (int64_t i = p.trailing_bit_offset; i < ar->length(); ++i) {
            if (GetBit(bitmap_data, i)) {
                builder.UnsafeAppend(ar->Value(i));
            }
        }

        std::shared_ptr<Array> out;
        builder.Finish(&out);
        return out;
    }


#define CASE_DOWNCAST_COPY(TypeID, Data_Type)                                                                \
case Type::TypeID:                                                                                           \
        return copy_array<Data_Type>(std::static_pointer_cast<arrow::NumericArray<Data_Type>>(ar),           \
                                                bitmap,                                                      \
                                                n_rows);                                                     \

    std::shared_ptr<Array> copy_array_with_bitmap(std::shared_ptr<Array> ar,
                                std::shared_ptr<Buffer> bitmap,
                                std::shared_ptr<DataType> dt,
                                uint64_t n_rows) {
        switch (dt->id()) {
            CASE_DOWNCAST_COPY(DOUBLE, arrow::DoubleType);
            CASE_DOWNCAST_COPY(FLOAT, arrow::FloatType);
            CASE_DOWNCAST_COPY(HALF_FLOAT, arrow::HalfFloatType);
            CASE_DOWNCAST_COPY(UINT8, arrow::UInt8Type);
            CASE_DOWNCAST_COPY(INT8, arrow::Int8Type);
            CASE_DOWNCAST_COPY(UINT16, arrow::UInt16Type);
            CASE_DOWNCAST_COPY(INT16, arrow::Int16Type);
            CASE_DOWNCAST_COPY(UINT32, arrow::UInt32Type);
            CASE_DOWNCAST_COPY(INT32, arrow::Int32Type);
            CASE_DOWNCAST_COPY(UINT64, arrow::UInt64Type);
            CASE_DOWNCAST_COPY(INT64, arrow::Int64Type);
            CASE_DOWNCAST_COPY(DATE32, arrow::Date32Type);
            CASE_DOWNCAST_COPY(DATE64, arrow::Date64Type);
            default:
                return nullptr;
        }
    }
#undef CASE_DOWNCAST_COPY


    DataFrame::DataFrame(std::shared_ptr<RecordBatch> rb) {
        m_batch = rb;
    }

    int64_t DataFrame::null_count() const {
        auto rb = this->m_batch;
        int64_t null_count = 0;
        for (std::shared_ptr<Array> column : rb->columns()) {
            null_count += column->null_count();
        }
        return null_count;
    }

    std::shared_ptr<Buffer> DataFrame::combined_bitmap_with_null() const {
        auto rb = this->m_batch;
        int first_col_idx = 0;

        for(int i = 0; i < rb->num_columns(); ++i) {
            auto col = rb->column(i);

            if (col->null_count()) {
                first_col_idx = i;
                break;
            }
        }

        auto res = Buffer::Copy(rb->column(first_col_idx)->null_bitmap(), arrow::default_cpu_memory_manager());
        auto bitmap = res.ValueOrDie();

        for(int i = first_col_idx + 1; i < rb->num_columns(); ++i) {
            auto col = rb->column(i);

            if (col->null_count()) {
                auto other_bitmap = col->null_bitmap();

                arrow::internal::BitmapAnd(bitmap->data(), 0,
                                            other_bitmap->data(), 0,
                                            rb->num_rows(),
                                            0, bitmap->mutable_data());
            }
        }

        return bitmap;
    }

    std::shared_ptr<Buffer> DataFrame::combined_bitmap() const {
        if (null_count() > 0) {
            return combined_bitmap_with_null();
        } else {
            return nullptr;
        }
    }

    int64_t DataFrame::null_instances_count() const {
        auto num_rows = this->m_batch->num_rows();
        auto comb_bitmap = combined_bitmap();

        if (comb_bitmap) {
            return num_rows - arrow::internal::CountSetBits(comb_bitmap->data(), 0, num_rows);
        } else {
            return 0;
        }
    }

    DataFrame DataFrame::drop_null_instances() {
        auto rb = this->m_batch;
        auto comb_bitmap = combined_bitmap();

        int64_t n_rows = arrow::internal::CountSetBits(comb_bitmap->data(), 0, rb->num_rows());

        if (comb_bitmap) {
            auto schema = rb->schema();

            std::vector<std::shared_ptr<Array>> columns;
            columns.reserve(rb->num_columns());

            for (int i = 0; i < schema->num_fields(); ++i) {
                auto field = schema->field(i);
                auto type = field->type();

                auto ar = copy_array_with_bitmap(rb->column(i), comb_bitmap, type, n_rows);
                columns.push_back(ar);
            }

            return DataFrame(RecordBatch::Make(schema, n_rows, columns));

        } else {
            return *this;
        }
    }

    Array_ptr DataFrame::loc(int i) const {
        return m_batch->column(i);
    }

    Array_ptr DataFrame::loc(const std::string& name) const {
        return m_batch->GetColumnByName(name);
    }

    MatrixXd DataFrame::to_eigen() const {
        auto rb = this->m_batch;
        auto comb_bitmap = combined_bitmap();
        int64_t n_rows = arrow::internal::CountSetBits(comb_bitmap->data(), 0, rb->num_rows());
        MatrixXd m(n_rows, rb->num_columns());

        return m;
    }

    std::shared_ptr<arrow::RecordBatch> DataFrame::operator->() { return m_batch; }

}


