#include <memory>
#include <iostream>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <arrow/python/pyarrow.h>
#include <arrow/api.h>

namespace py = pybind11;
namespace arpy = arrow::py;
//namespace ar = arrow;

using namespace std;
//using namespace arrow;

int add(int i, int j) {
    return i + j;
}

void print_information(shared_ptr<arrow::Table> t) {
    for (auto chunked_array : t->columns()) {
        cout << "New Column" << "\n";
        cout << "=====================" << "\n";

        cout << "chunked_array length " << chunked_array->length() << "\n";
        cout << "chunked_array null_count " << chunked_array->null_count() << "\n";
        cout << "chunked_array num_chunks " << chunked_array->num_chunks() << "\n";

        auto array = chunked_array->chunk(0);

        cout << "Length " << array->length() << "\n";
        cout << "IsNull[0] " << array->IsNull(0) << "\n";
        cout << "IsValid[0] " << array->IsValid(0) << "\n";
        cout << "offset " << array->offset() << "\n";
        cout << "null_count " << array->null_count() << "\n";
        auto bitmap = array->null_bitmap();
        if (bitmap)
            cout << "null_bitmap " << bitmap->ToHexString() << "\n";
        else
            cout << "null_bitmap nullptr" << "\n";
        cout << "type " << array->type()->ToString() << "\n";
        cout << "num_fields " << array->num_fields() << "\n";
        cout << "ToString " << array->ToString() << "\n";

        auto type = array->type();


        switch (type->id()) {
            case arrow::Type::DOUBLE:
            {
                auto casted_array = static_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(array);
                cout << "Casted \n";
                cout << "Pointer " << casted_array->raw_values() << "\n";
                cout << "0 value" << casted_array->raw_values()[0] << "\n";
                cout << "1 value" << casted_array->raw_values()[1] << "\n";
                cout << "2 value" << casted_array->Value(2) << "\n";
                break;
            }
            case arrow::Type::INT64:
            {
                auto casted_array = static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(array);
                cout << "Casted \n";
                cout << "Pointer " << casted_array->raw_values() << "\n";
                cout << "0 value" << casted_array->raw_values()[0] << "\n";
                cout << "1 value" << casted_array->raw_values()[1] << "\n";
                cout << "2 value" << casted_array->Value(2) << "\n";
                break;
            }
            default:
                cout << "Not Casted \n";
                continue;
        }
//        cout << "0 value" << casted_array->raw_values()[0] << "\n";
//        cout << "1 value" << casted_array->raw_values()[1] << "\n";
//        cout << "2 value" << casted_array->Value(2) << "\n";

    }
}

bool is_table_py(py::handle t) {
    PyObject *table = t.ptr();
    arpy::import_pyarrow();
    return arpy::is_table(table);
}

int num_rows(py::handle t) {
    PyObject *table = t.ptr();
    arpy::import_pyarrow();

    std::shared_ptr<arrow::Table> unwraped;
    arpy::unwrap_table(table, &unwraped);

    print_information(unwraped);

    return unwraped->num_rows();
}


PYBIND11_MODULE(data, m) {
    m.doc() = "pybind11 data plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    m.def("is_table_py", &is_table_py, "A function which checks table");
    m.def("num_rows", &num_rows, "A function which returns the number of rows in the table");
}
