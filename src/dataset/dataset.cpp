#include <arrow/api.h>
#include <arrow/util/align_util.h>
#include <arrow/util/bitmap_ops.h>
#include <arrow/python/pyarrow.h>
#include <dataset/dataset.hpp>
#include <Eigen/Dense>
#include <util/parameter_traits.hpp>

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

    bool is_pandas_series(py::handle pyobject) {
        PyObject* pyobj_ptr = pyobject.ptr();

        PyObject* module = PyImport_ImportModule("pandas");
        PyObject* moduleDict = PyModule_GetDict(module);
        PyObject* protocolClass = PyDict_GetItemString(moduleDict, "Series");

        return PyObject_IsInstance(pyobj_ptr, protocolClass);
    }

    py::object pandas_to_pyarrow_record_batch(py::handle pyobject) {
        auto d = py::module::import("pyarrow").attr("RecordBatch").attr("from_pandas")(pyobject);
        return d;
    }

    py::object pandas_to_pyarrow_array(py::handle pyobject) {
        auto d = py::module::import("pyarrow").attr("Array").attr("from_pandas")(pyobject);
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
        auto status = builder.Resize(n_rows);
        if (!status.ok()) {
            throw std::runtime_error("New array could not be created. Error status: " + status.ToString());
        }

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
                    status = builder.AppendValues(ar->raw_values() + offset, 64);
                    if (!status.ok()) {
                        throw std::runtime_error("New array could not be created. Error status: " + status.ToString());
                    }
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
        status = builder.Finish(&out);
        if (!status.ok()) {
            throw std::runtime_error("New array could not be created. Error status: " + status.ToString());
        }
        
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
            default:
                return nullptr;
        }
    }
#undef CASE_DOWNCAST_COPY

    DataFrame::DataFrame(std::shared_ptr<RecordBatch> rb) : m_batch(rb) { }

    std::vector<std::string> DataFrame::column_names() const {
        auto schema = m_batch->schema();
        std::vector<std::string> names;
        names.reserve(schema->num_fields());

        for(int i = 0, num_fields = schema->num_fields(); i < num_fields; ++i) {
            names.push_back(schema->field(i)->name());
        }

        return names;
    }

    std::shared_ptr<arrow::RecordBatch> DataFrame::operator->() const { return m_batch; }

    int64_t null_count(Array_iterator begin, Array_iterator end) {
        int64_t r = 0;
        for (auto it = begin; it != end; it++) {
            r += (*it)->null_count();
        }
        return r;
    }

    Buffer_ptr combined_bitmap(Array_iterator begin, Array_iterator end) {
        if (null_count(begin, end) > 0) {

            Array_iterator first_null_col = end;

            for(auto it = begin; it < end; ++it) {
                if ((*it)->null_count() != 0) {
                    first_null_col = it;
                    break;
                }
            }

            auto res = Buffer::Copy((*first_null_col)->null_bitmap(), arrow::default_cpu_memory_manager());
            auto bitmap = std::move(res).ValueOrDie();

            for(auto it = first_null_col + 1; it < end; ++it) {
                auto col = *it;
                if (col->null_count()) {
                    auto other_bitmap = col->null_bitmap();

                    arrow::internal::BitmapAnd(bitmap->data(), 0,
                                               other_bitmap->data(), 0,
                                                (*first_null_col)->length(),
                                               0, bitmap->mutable_data());
                }
            }
            return bitmap;
        } else {
            return nullptr;
        }
    }

    int64_t valid_rows(Array_iterator begin, Array_iterator end) {
        if (std::distance(begin, end) == 0) {
            return 0;
        }

        auto bitmap = combined_bitmap(begin, end);
        if (bitmap)
            return util::bit_util::non_null_count(bitmap, (*begin)->length());
        else
            return (*begin)->length();
    }

    DataFrame DataFrame::loc(int i) const {
        arrow::SchemaBuilder b;
        auto status = b.AddField(m_batch->schema()->field(i));
        if (!status.ok()) {
            throw std::runtime_error("Field could not be added to the Schema. Error status: " + status.ToString());
        }

        auto r = b.Finish();
        if (!r.ok()) {
            throw std::domain_error("Schema could not be created for column index " + std::to_string(i));
        }
        
        Array_vector c = { m_batch->column(i) };
        return RecordBatch::Make(std::move(r).ValueOrDie(), m_batch->num_rows(), c);
    }

    arrow::Type::type DataFrame::same_type(Array_iterator begin, Array_iterator end) const {
        if (std::distance(begin, end) == 0) {
            throw std::invalid_argument("Cannot check the data type of no columns");
        }

        arrow::Type::type dt = (*begin)->type_id();

        for (auto it = begin+1; it != end; ++it) {
            if((*it)->type_id() != dt) {
                throw std::invalid_argument("Column 0 [" + (*begin)->type()->ToString() + "] and "
                                            "column " + std::to_string(std::distance(begin, it)) + "[" + (*it)->type()->ToString() + "] " 
                                            "have different data types");
            }
        }

        return dt;
    }

    std::vector<int> DataFrame::continuous_columns() const {
        std::vector<int> res;

        arrow::Type::type dt = arrow::Type::NA;
        for (int i = 0; i < m_batch->num_columns() && dt == Type::NA; ++i) {
            auto column = m_batch->column(i);
            switch (column->type_id()) {
                case Type::DOUBLE:
                case Type::FLOAT:
                    dt = column->type_id();
                    res.push_back(i);
                    break;
                default:
                    break;
            }
        }

        if (dt == Type::NA) {
            return res;
        }

        for (int i = res[0]+1; i < m_batch->num_columns(); ++i) {
            auto column = m_batch->column(i);

            switch(column->type_id()) {
                case Type::DOUBLE: {
                    if (dt == Type::FLOAT)
                        throw std::invalid_argument("Column " + std::to_string(res[0]) + 
                                            " [" + m_batch->column(res[0])->type()->ToString() + "] and "
                                            "column " + std::to_string(i) + "[" + column->type()->ToString() + "] " 
                                            "have different continuous data types");
                    res.push_back(i);
                    break;
                }
                case Type::FLOAT: {
                    if (dt == Type::DOUBLE)
                        throw std::invalid_argument("Column " + std::to_string(res[0]) + 
                                            " [" + m_batch->column(res[0])->type()->ToString() + "] and "
                                            "column " + std::to_string(i) + "[" + column->type()->ToString() + "] " 
                                            "have different continuous data types");
                    res.push_back(i);
                    break;
                }
                default:
                    break;
            }
        }

        return res;
    }
}
