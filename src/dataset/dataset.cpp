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


    Array_ptr copy_array(const Array_ptr& array) {
        switch (array->type_id()) {
            case Type::DOUBLE:
                return copy_array_numeric<arrow::DoubleType>(array);
            case Type::FLOAT:
                return copy_array_numeric<arrow::FloatType>(array);
            case Type::INT64:
                return copy_array_numeric<arrow::Int64Type>(array);
            case Type::UINT64:
                return copy_array_numeric<arrow::UInt64Type>(array);
            case Type::INT32:
                return copy_array_numeric<arrow::Int32Type>(array);
            case Type::UINT32:
                return copy_array_numeric<arrow::UInt32Type>(array);
            case Type::INT16:
                return copy_array_numeric<arrow::Int16Type>(array);
            case Type::UINT16:
                return copy_array_numeric<arrow::UInt16Type>(array);
            case Type::INT8:
                return copy_array_numeric<arrow::Int8Type>(array);
            case Type::UINT8:
                return copy_array_numeric<arrow::UInt8Type>(array);
            case Type::STRING:
                return copy_array_string(array);
            case Type::DICTIONARY:
                return copy_array_dictionary(array);
            default:
                throw std::invalid_argument("Not supported datatype copy.");
        }
    }

    Array_ptr copy_array_string(const Array_ptr& array) {
        arrow::StringBuilder builder;

        auto dwn_array = std::static_pointer_cast<arrow::StringArray>(array);
        auto res_offsets = arrow::Buffer::Copy(dwn_array->value_offsets(), arrow::default_cpu_memory_manager());
        RAISE_STATUS_ERROR(res_offsets.status())
        auto new_value_offsets = std::move(res_offsets).ValueOrDie();

        auto res_values = arrow::Buffer::Copy(dwn_array->value_data(), arrow::default_cpu_memory_manager());
        RAISE_STATUS_ERROR(res_values.status())
        auto new_values = std::move(res_values).ValueOrDie();

        auto null_count = array->null_count();
        return std::make_shared<arrow::StringArray>(dwn_array->length(), 
                        new_value_offsets, new_values, dwn_array->null_bitmap(), null_count);
    }

    Array_ptr copy_array_dictionary(const Array_ptr& array) {
        auto dwn_array = std::static_pointer_cast<arrow::DictionaryArray>(array);
        auto new_dictionary = copy_array(dwn_array->dictionary());
        auto new_indices = copy_array(dwn_array->indices());
        return std::make_shared<arrow::DictionaryArray>(array->type(), new_indices, new_dictionary);
    }

    // DataFrame::DataFrame(std::shared_ptr<RecordBatch> rb) : m_batch(rb) { }

    // std::vector<std::string> DataFrame::column_names() const {
    //     auto schema = m_batch->schema();
    //     std::vector<std::string> names;
    //     names.reserve(schema->num_fields());

    //     for(int i = 0, num_fields = schema->num_fields(); i < num_fields; ++i) {
    //         names.push_back(schema->field(i)->name());
    //     }

    //     return names;
    // }

    // std::shared_ptr<arrow::RecordBatch> DataFrame::operator->() const { return m_batch; }

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
    
    std::string index_to_string(int i) {
        return std::to_string(i);
    }

    std::string index_to_string(std::string name) {
        return name;
    }

    std::string index_to_string(DynamicVariable<int> i) {
        return "(" + std::to_string(i.variable) + ", " std::to_string(i.temporal_slice) + ")";
    }
    std::string index_to_string(DynamicVariable<std::string> name) {
        return "(" + i.variable + ", " std::to_string(i.temporal_slice) + ")";
    }

    // void DataFrame::has_columns(int i) const {
    //     if (i < 0 || i >= m_batch->num_columns()) {
    //         throw std::domain_error("Index " + std::to_string(i) + 
    //                                 " do no exist for DataFrame with " + std::to_string(m_batch->num_columns()) + " columns.");
    //     }
    // }

    // DataFrame DataFrame::loc(int i) const {
    //     arrow::SchemaBuilder b;
    //     RAISE_STATUS_ERROR(b.AddField(field(i)));

    //     auto r = b.Finish();
    //     if (!r.ok()) {
    //         throw std::domain_error("Schema could not be created for column index " + std::to_string(i));
    //     }
        
    //     Array_vector c = { m_batch->column(i) };
    //     return RecordBatch::Make(std::move(r).ValueOrDie(), m_batch->num_rows(), c);
    // }

    // arrow::Type::type DataFrame::same_type(Array_iterator begin, Array_iterator end) const {
    //     if (std::distance(begin, end) == 0) {
    //         throw std::invalid_argument("Cannot check the data type of no columns");
    //     }

    //     arrow::Type::type dt = (*begin)->type_id();

    //     for (auto it = begin+1; it != end; ++it) {
    //         if((*it)->type_id() != dt) {
    //             throw std::invalid_argument("Column 0 [" + (*begin)->type()->ToString() + "] and "
    //                                         "column " + std::to_string(std::distance(begin, it)) + 
    //                                         " [" + (*it)->type()->ToString() + "] have different data types");
    //         }
    //     }

    //     return dt;
    // }

    // std::vector<int> DataFrame::continuous_columns() const {
    //     std::vector<int> res;

    //     arrow::Type::type dt = arrow::Type::NA;
    //     for (int i = 0; i < m_batch->num_columns() && dt == Type::NA; ++i) {
    //         auto column = m_batch->column(i);
    //         switch (column->type_id()) {
    //             case Type::DOUBLE:
    //             case Type::FLOAT:
    //                 dt = column->type_id();
    //                 res.push_back(i);
    //                 break;
    //             default:
    //                 break;
    //         }
    //     }

    //     if (dt == Type::NA) {
    //         return res;
    //     }

    //     for (int i = res[0]+1; i < m_batch->num_columns(); ++i) {
    //         auto column = m_batch->column(i);

    //         switch(column->type_id()) {
    //             case Type::DOUBLE: {
    //                 if (dt == Type::FLOAT)
    //                     throw std::invalid_argument("Column " + std::to_string(res[0]) + 
    //                                         " [" + m_batch->column(res[0])->type()->ToString() + "] and "
    //                                         "column " + std::to_string(i) + "[" + column->type()->ToString() + "] " 
    //                                         "have different continuous data types");
    //                 res.push_back(i);
    //                 break;
    //             }
    //             case Type::FLOAT: {
    //                 if (dt == Type::DOUBLE)
    //                     throw std::invalid_argument("Column " + std::to_string(res[0]) + 
    //                                         " [" + m_batch->column(res[0])->type()->ToString() + "] and "
    //                                         "column " + std::to_string(i) + "[" + column->type()->ToString() + "] " 
    //                                         "have different continuous data types");
    //                 res.push_back(i);
    //                 break;
    //             }
    //             default:
    //                 break;
    //         }
    //     }

    //     return res;
    // }
}
