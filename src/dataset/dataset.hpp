#ifndef PGM_DATASET_DATASET_HPP
#define PGM_DATASET_DATASET_HPP

#include <pybind11/pybind11.h>
#include <Eigen/Dense>
//#include <arrow/python/pyarrow.h>
#include <arrow/api.h>
#include <util/util.hpp>

namespace py = pybind11;

using Eigen::MatrixXd;

typedef std::shared_ptr<arrow::Array> Array_ptr;

namespace dataset {
    typedef py::handle PyDataset;

    bool is_pandas_dataframe(py::handle pyobject);

    std::shared_ptr<arrow::RecordBatch> to_record_batch(py::handle pyobject);

    class Column {
    public:
        Column(Array_ptr column);

        Array_ptr operator->();
    private:
        Array_ptr m_column;
    };

    class DataFrame {
    public:
        DataFrame(std::shared_ptr<arrow::RecordBatch> rb);

        int64_t null_count() const;
        int64_t null_instances_count() const;
        DataFrame drop_null_instances();

        std::shared_ptr <arrow::Buffer> combined_bitmap() const;

        template<typename T>
        DataFrame loc(T cols) const;
        template<typename V>
        DataFrame loc(std::initializer_list<V> cols) const { return loc<std::initializer_list<V>>(cols); }
        Column loc(int i) const { return m_batch->column(i); }
        Column loc(const std::string& name) const { return m_batch->GetColumnByName(name); }


//        template<typename CType>
//        Matrix<CType, Dynamic, Dynamic> to_eigen();
//        template<typename CType>
//        Matrix<CType, Dynamic, Dynamic> to_eigen(Buffer_ptr bitmap);

        std::shared_ptr<arrow::RecordBatch> operator->();
    private:
        std::shared_ptr <arrow::Buffer> combined_bitmap_with_null() const;
        std::shared_ptr <arrow::RecordBatch> m_batch;
    };

    template<typename T>
    DataFrame DataFrame::loc(T cols) const {
        static_assert(util::is_integral_container_v<T> || util::is_string_container_v<T>,
                      "loc() only accepts integral or string containers.");

        auto size = cols.size();

        std::vector<std::shared_ptr<arrow::Field>> new_fields;
        new_fields.reserve(size);

        std::vector<Array_ptr> new_cols;
        new_cols.reserve(size);
        for (auto c : cols) {
            if constexpr (util::is_integral_container_v<T>) {
                auto field = m_batch->schema()->field(c);
//                if (!field)
//                    throw std::out_of_range("Wrong index field selected. Index (" + c + ") can not be indexed from"
//                                                        " a DataFrame with " + m_batch->num_columns() + " columns");

                new_cols.push_back(m_batch->column(c));
                new_fields.push_back(field);
            }
            else if constexpr (util::is_string_container_v<T>) {
                auto field = m_batch->schema()->GetFieldByName(c);
//                if (!field)
//                    throw std::out_of_range(std::string("Column \"") + c + "\" not present in the DataFrame with columns:\n"
//                                                           + m_batch->schema()->ToString() );

                new_cols.push_back(m_batch->GetColumnByName(c));
                new_fields.push_back(field);
            }
        }

        auto new_schema = std::make_shared<arrow::Schema>(new_fields);
        return DataFrame(arrow::RecordBatch::Make(new_schema, m_batch->num_rows(), new_cols));
    }

//    template<typename CType>
//    Matrix<CType, Dynamic, Dynamic> DataFrame::to_eigen() {
//
//    }
}


#endif //PGM_DATASET_DATASET_HPP
