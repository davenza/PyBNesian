#include <arrow/api.h>
#include <arrow/util/align_util.h>
#include <arrow/util/bitmap_ops.h>
#include <arrow/python/pyarrow.h>
#include <dataset/dataset.hpp>
#include <Eigen/Dense>
#include <util/parameter_traits.hpp>
#include <util/basic_eigen_ops.hpp>

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
        auto d = py::module::import("pyarrow").attr("RecordBatch").attr("from_pandas")(pyobject, py::none(), false);
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

    std::string array_type_ToString(arrow::Type::type t) {
        switch (t) {
            case Type::NA:
                return arrow::null()->name();
            case Type::BOOL:
                return arrow::boolean()->name();
            case Type::UINT8:
                return arrow::uint8()->name();
            case Type::INT8:
                return arrow::int8()->name();
            case Type::UINT16:
                return arrow::uint16()->name();
            case Type::INT16:
                return arrow::int16()->name();
            case Type::UINT32:
                return arrow::uint32()->name();
            case Type::INT32:
                return arrow::int32()->name();
            case Type::UINT64:
                return arrow::uint64()->name();
            case Type::INT64:
                return arrow::int64()->name();
            case Type::HALF_FLOAT:
                return arrow::float16()->name();
            case Type::FLOAT:
                return arrow::float32()->name();
            case Type::DOUBLE:
                return arrow::float64()->name();
            case Type::STRING:
                return arrow::utf8()->name();
            case Type::BINARY:
                return arrow::binary()->name();
            case Type::FIXED_SIZE_BINARY:
                return "FIXED_SIZE_BINARY";
            case Type::DATE32:
                return arrow::date32()->name();
            case Type::DATE64:
                return arrow::date64()->name();
            case Type::TIMESTAMP:
                return "TIMESTAMP";
            case Type::TIME32:
                return "TIME32";
            case Type::TIME64:
                return "TIME64";
            case Type::INTERVAL_MONTHS:
                return arrow::month_interval()->name();
            case Type::INTERVAL_DAY_TIME:
                return arrow::day_time_interval()->name();
            case Type::DECIMAL128:
                return "DECIMAL128";
            case Type::DECIMAL256:
                return "DECIMAL256";
            case Type::LIST:
                return "LIST";
            case Type::STRUCT:
                return "STRUCT";
            case Type::SPARSE_UNION:
                return "SPARSE_UNION";
            case Type::DENSE_UNION:
                return "DENSE_UNION";
            case Type::DICTIONARY:
                return "DICTIONARY";
            case Type::MAP:
                return "MAP";
            case Type::EXTENSION:
                return "EXTENSION";
            case Type::FIXED_SIZE_LIST:
                return "FIXED_SIZED_LIST";
            case Type::DURATION:
                return "DURATION";
            case Type::LARGE_STRING:
                return arrow::large_utf8()->name();
            case Type::LARGE_BINARY:
                return arrow::large_binary()->name();
            case Type::LARGE_LIST:
                return "LARGE_LIST";
            case Type::MAX_ID:
                return "MAX_ID";
        }

        return {};
    }

    std::vector<std::string> DataFrame::column_names() const {
        auto schema = m_batch->schema();
        std::vector<std::string> names;
        names.reserve(schema->num_fields());

        for(int i = 0, num_fields = schema->num_fields(); i < num_fields; ++i) {
            names.push_back(schema->field(i)->name());
        }

        return names;
    }

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

            for(auto it = ++first_null_col; it < end; ++it) {
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

    std::string index_to_string(const std::string& name) {
        return name;
    }

    arrow::Type::type same_type(Array_iterator begin, Array_iterator end) {
        if (std::distance(begin, end) == 0) {
            throw std::invalid_argument("Cannot check the data type of no columns");
        }

        arrow::Type::type dt = (*begin)->type_id();

        for (auto it = begin+1; it != end; ++it) {
            if((*it)->type_id() != dt) {
                throw std::invalid_argument("Column 0 [" + (*begin)->type()->ToString() + "] and "
                                            "column " + std::to_string(std::distance(begin, it)) + 
                                            " [" + (*it)->type()->ToString() + "] have different data types");
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

    double mean(Array_ptr& a) {
        switch (a->type_id()) {
            case Type::DOUBLE:
                return dataset::mean<arrow::DoubleType>(a); 
            case Type::FLOAT:
                return static_cast<double>(dataset::mean<arrow::FloatType>(a));
            default:
                throw std::invalid_argument("mean() only implemented for \"double\" and \"float\" data types.");
        }
    }

    double mean(const Buffer_ptr& bitmap, Array_ptr& a) {
        switch (a->type_id()) {
            case Type::DOUBLE:
                return dataset::mean<arrow::DoubleType>(bitmap, a); 
            case Type::FLOAT:
                return static_cast<double>(dataset::mean<arrow::FloatType>(bitmap, a));
            default:
                throw std::invalid_argument("mean() only implemented for \"double\" and \"float\" data types.");
        }
    }

    double mean(Array_ptr&& a) {
        return mean(a);
    }

    double mean(const Buffer_ptr&& bitmap, Array_ptr&& a) {
        return mean(bitmap, a);
    }

    VectorXd means(Array_iterator begin, Array_iterator end) {
        VectorXd res(std::distance(begin, end));

        int i = 0;
        for (auto it = begin; it != end; ++it, ++i) {
           res(i) = mean(*it);
        }

        return res;
    }

    VectorXd means(const Buffer_ptr& bitmap, Array_iterator begin, Array_iterator end) {
        VectorXd res(std::distance(begin, end));

        int i = 0;
        for (auto it = begin; it != end; ++it, ++i) {
           res(i) = mean(bitmap, *it);
        }

        return res;
    }

    template<typename ArrowType, typename EigenArray>
    Array_ptr normalize_column(const Array_ptr& array, EigenArray& eig) {
        util::normalize_cols(eig);

        arrow::NumericBuilder<ArrowType> builder;
        RAISE_STATUS_ERROR(builder.Reserve(array->length()));
        if (array->null_count() == 0) {
            RAISE_STATUS_ERROR(builder.AppendValues(eig.data(), eig.rows()));
        } else {
            auto bitmap_data = array->null_bitmap_data();
            for (int i = 0, j = 0; i < array->length(); ++i) {
                if (arrow::BitUtil::GetBit(bitmap_data, i)) {
                    builder.UnsafeAppend(eig(j++));
                } else {
                    builder.UnsafeAppendNull();
                }
            }
        }

        std::shared_ptr<arrow::Array> out;
        RAISE_STATUS_ERROR(builder.Finish(&out));

        return out;
    }

    DataFrame DataFrame::normalize() const {
        auto continuous_cols = continuous_columns();

        std::vector<Array_ptr> columns;

        for (auto i = 0; i < this->num_columns(); ++i) {
            auto column = col(i);

            switch(column->type_id()) {
                case Type::DOUBLE: {
                    auto eigen_vec = to_eigen<false, arrow::DoubleType>(i);
                    columns.push_back(normalize_column<arrow::DoubleType>(column, *eigen_vec));
                    break;
                }
                case Type::FLOAT: {
                    auto eigen_vec = to_eigen<false, arrow::FloatType>(i);
                    columns.push_back(normalize_column<arrow::FloatType>(column, *eigen_vec));
                    break;
                }
                default:
                    columns.push_back(column);
            }
        }

        return DataFrame(arrow::RecordBatch::Make(m_batch->schema(), this->num_rows(), columns));
    }

}
