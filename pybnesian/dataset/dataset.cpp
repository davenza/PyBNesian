#include <arrow/api.h>
#include <arrow/util/align_util.h>
#include <arrow/util/bitmap_ops.h>
#include <dataset/dataset.hpp>
#include <Eigen/Dense>
#include <util/parameter_traits.hpp>
#include <util/basic_eigen_ops.hpp>

using Eigen::MatrixXd;

namespace py = pybind11;
using arrow::Array, arrow::RecordBatch, arrow::Result, arrow::Buffer, arrow::NumericBuilder, arrow::DataType,
    arrow::Type, arrow::NumericBuilder;

namespace dataset {

bool is_pyarrow_instance(py::handle pyobject, const char* class_name) {
    PyObject* pyobj_ptr = pyobject.ptr();

    PyObject* module = PyImport_ImportModule("pyarrow");
    PyObject* moduleDict = PyModule_GetDict(module);
    PyObject* protocolClass = PyDict_GetItemString(moduleDict, class_name);

    bool is_instance = PyObject_IsInstance(pyobj_ptr, protocolClass);
    Py_DECREF(module);
    return is_instance;
}

bool is_pandas_dataframe(py::handle pyobject) {
    PyObject* pyobj_ptr = pyobject.ptr();

    PyObject* module = PyImport_ImportModule("pandas");
    PyObject* moduleDict = PyModule_GetDict(module);
    PyObject* protocolClass = PyDict_GetItemString(moduleDict, "DataFrame");

    bool is_instance = PyObject_IsInstance(pyobj_ptr, protocolClass);
    Py_DECREF(module);
    return is_instance;
}

bool is_pandas_series(py::handle pyobject) {
    PyObject* pyobj_ptr = pyobject.ptr();

    PyObject* module = PyImport_ImportModule("pandas");
    PyObject* moduleDict = PyModule_GetDict(module);
    PyObject* protocolClass = PyDict_GetItemString(moduleDict, "Series");

    bool is_instance = PyObject_IsInstance(pyobj_ptr, protocolClass);
    Py_DECREF(module);
    return is_instance;
}

py::object pandas_to_pyarrow_record_batch(py::handle pyobject) {
    auto d = py::module::import("pyarrow").attr("RecordBatch").attr("from_pandas")(pyobject, py::none(), false);
    return d;
}

py::object pandas_to_pyarrow_array(py::handle pyobject) {
    auto d = py::module::import("pyarrow").attr("Array").attr("from_pandas")(pyobject);
    return d;
}

struct ArrowSchema* extract_pycapsule_schema(py::handle pyobject) {
    PyObject* py_ptr = pyobject.ptr();
    // call the method and get the tuple
    PyObject* arrow_c_method = PyObject_GetAttrString(py_ptr, "__arrow_c_schema__");

    if (arrow_c_method == NULL) {
        throw pybind11::attribute_error("Method __arrow_c_schema__ not found.");
    }

    #ifdef Python_MAJOR_VERSION == 3 && Python_MINOR_VERSION >= 9
    PyObject* schema_capsule_obj = PyObject_CallNoArgs(arrow_c_method);
    #else
    PyObject* args = PyTuple_New(0);
    PyObject* schema_capsule_obj = PyObject_Call(arrow_c_method, args, NULL);
    Py_DECREF(args);
    #endif
    Py_DECREF(arrow_c_method);

    // extract the capsule
    struct ArrowSchema* c_schema = (struct ArrowSchema*)PyCapsule_GetPointer(schema_capsule_obj, "arrow_schema");

    return c_schema;
}

struct ArrowCAPIObjects extract_pycapsule_array(py::handle pyobject) {
    PyObject* py_ptr = pyobject.ptr();
    // call the method and get the tuple

    PyObject* arrow_c_method = PyObject_GetAttrString(py_ptr, "__arrow_c_array__");

    if (arrow_c_method == NULL) {
        throw pybind11::attribute_error("Method __arrow_c_array__ not found.");
    }

    #ifdef Python_MAJOR_VERSION == 3 && Python_MINOR_VERSION >= 9
    PyObject* array_capsule_tuple = PyObject_CallNoArgs(arrow_c_method);
    #else
    PyObject* args = PyTuple_New(0);
    PyObject* array_capsule_tuple = PyObject_Call(arrow_c_method, args, NULL);
    Py_DECREF(args);
    #endif

    Py_DECREF(arrow_c_method);

    PyObject* schema_capsule_obj = PyTuple_GetItem(array_capsule_tuple, 0);
    PyObject* array_capsule_obj = PyTuple_GetItem(array_capsule_tuple, 1);

    // extract the capsule
    struct ArrowSchema* c_schema = (struct ArrowSchema*)PyCapsule_GetPointer(schema_capsule_obj, "arrow_schema");
    struct ArrowArray* c_array = (struct ArrowArray*)PyCapsule_GetPointer(array_capsule_obj, "arrow_array");

    return ArrowCAPIObjects{c_schema, c_array};
}

void ReleaseArrowSchemaPyCapsule(PyObject* capsule) {
    struct ArrowSchema* schema = (struct ArrowSchema*)PyCapsule_GetPointer(capsule, "arrow_schema");
    if (schema->release != NULL) {
        schema->release(schema);
    }
    free(schema);
}

void ReleaseArrowArrayPyCapsule(PyObject* capsule) {
    struct ArrowArray* array = (struct ArrowArray*)PyCapsule_GetPointer(capsule, "arrow_array");
    if (array->release != NULL) {
        array->release(array);
    }
    free(array);
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
    return std::make_shared<arrow::StringArray>(
        dwn_array->length(), new_value_offsets, new_values, dwn_array->null_bitmap(), null_count);
}

Array_ptr copy_array_dictionary(const Array_ptr& array) {
    auto dwn_array = std::static_pointer_cast<arrow::DictionaryArray>(array);
    auto new_dictionary = copy_array(dwn_array->dictionary());
    auto new_indices = copy_array(dwn_array->indices());
    return std::make_shared<arrow::DictionaryArray>(array->type(), new_indices, new_dictionary);
}

std::vector<std::string> DataFrame::column_names() const {
    auto schema = m_batch->schema();
    std::vector<std::string> names;
    names.reserve(schema->num_fields());

    for (int i = 0, num_fields = schema->num_fields(); i < num_fields; ++i) {
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

        for (auto it = begin; it < end; ++it) {
            if ((*it)->null_count() != 0) {
                first_null_col = it;
                break;
            }
        }

        auto res = Buffer::Copy((*first_null_col)->null_bitmap(), arrow::default_cpu_memory_manager());
        auto bitmap = std::move(res).ValueOrDie();

        for (auto it = ++first_null_col; it < end; ++it) {
            auto col = *it;
            if (col->null_count()) {
                auto other_bitmap = col->null_bitmap();

                arrow::internal::BitmapAnd(
                    bitmap->data(), 0, other_bitmap->data(), 0, (*first_null_col)->length(), 0, bitmap->mutable_data());
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

std::string index_to_string(int i) { return std::to_string(i); }

std::string index_to_string(const std::string& name) { return name; }

std::shared_ptr<arrow::DataType> same_type(Array_iterator begin, Array_iterator end) {
    if (std::distance(begin, end) == 0) {
        throw std::invalid_argument("Cannot check the data type of no columns");
    }

    std::shared_ptr<arrow::DataType> dt = (*begin)->type();

    for (auto it = begin + 1; it != end; ++it) {
        if ((*it)->type_id() != dt->id()) {
            throw std::invalid_argument("Column 0 [" + dt->ToString() +
                                        "] and "
                                        "column " +
                                        std::to_string(std::distance(begin, it)) + " [" + (*it)->type()->ToString() +
                                        "] have different data types");
        }
    }

    return dt;
}

std::vector<int> DataFrame::discrete_columns() const {
    std::vector<int> res;

    for (int i = 0; i < m_batch->num_columns(); ++i) {
        auto column = m_batch->column(i);
        if (column->type_id() == Type::DICTIONARY) {
            res.push_back(i);
        }
    }

    return res;
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

    for (int i = res[0] + 1; i < m_batch->num_columns(); ++i) {
        auto column = m_batch->column(i);

        switch (column->type_id()) {
            case Type::DOUBLE: {
                if (dt == Type::FLOAT)
                    throw std::invalid_argument("Column " + std::to_string(res[0]) + " [" +
                                                m_batch->column(res[0])->type()->ToString() +
                                                "] and "
                                                "column " +
                                                std::to_string(i) + "[" + column->type()->ToString() +
                                                "] "
                                                "have different continuous data types");
                res.push_back(i);
                break;
            }
            case Type::FLOAT: {
                if (dt == Type::DOUBLE)
                    throw std::invalid_argument("Column " + std::to_string(res[0]) + " [" +
                                                m_batch->column(res[0])->type()->ToString() +
                                                "] and "
                                                "column " +
                                                std::to_string(i) + "[" + column->type()->ToString() +
                                                "] "
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

double mean(Array_ptr&& a) { return mean(a); }

double mean(const Buffer_ptr&& bitmap, Array_ptr&& a) { return mean(bitmap, a); }

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

template <typename ArrowType, typename EigenArray>
Array_ptr normalize_column(const Array_ptr& array, EigenArray& eig) {
    util::normalize_cols(eig);

    arrow::NumericBuilder<ArrowType> builder;
    RAISE_STATUS_ERROR(builder.Reserve(array->length()));
    if (array->null_count() == 0) {
        RAISE_STATUS_ERROR(builder.AppendValues(eig.data(), eig.rows()));
    } else {
        auto bitmap_data = array->null_bitmap_data();
        for (int i = 0, j = 0; i < array->length(); ++i) {
            if (util::bit_util::GetBit(bitmap_data, i)) {
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

        switch (column->type_id()) {
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

}  // namespace dataset
