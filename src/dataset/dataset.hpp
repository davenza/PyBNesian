#ifndef PGM_DATASET_DATASET_HPP
#define PGM_DATASET_DATASET_HPP

#include <variant>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
//#include <arrow/python/pyarrow.h>
#include <arrow/api.h>
#include <util/util.hpp>
#include <util/bit_util.hpp>

namespace py = pybind11;

using Eigen::MatrixXd, Eigen::MatrixXf, Eigen::VectorXd, Eigen::VectorXf, Eigen::Matrix, Eigen::Dynamic;
using arrow::Type, arrow::Buffer, arrow::DoubleType, arrow::FloatType;

typedef std::shared_ptr<arrow::Array> Array_ptr;
typedef std::shared_ptr<arrow::Buffer> Buffer_ptr;

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
//        DataFrame drop_null_instances();

        template<typename T>
        DataFrame loc(T cols) const;
        template<typename V>
        DataFrame loc(std::initializer_list<V> cols) const { return loc<std::initializer_list<V>>(cols); }
        Column loc(int i) const { return m_batch->column(i); }
        Column loc(const std::string& name) const { return m_batch->GetColumnByName(name); }

        template<typename T>
        Buffer_ptr combined_bitmap(T cols) const;
        template<typename V>
        Buffer_ptr combined_bitmap(std::initializer_list<V> cols) const { return combined_bitmap<std::initializer_list<V>>(cols); }
        Buffer_ptr combined_bitmap(int i) const { return m_batch->column(i)->null_bitmap(); }
        Buffer_ptr combined_bitmap(const std::string& name) const { return m_batch->GetColumnByName(name)->null_bitmap(); }

        template<typename T>
        int64_t null_count(T cols) const;
        template<typename V>
        int64_t null_count(std::initializer_list<V> cols) const { return null_count<std::initializer_list<V>>(cols); }
        int64_t null_count(int i) const { return m_batch->column(i)->null_count(); }
        int64_t null_count(const std::string& name) const { return m_batch->GetColumnByName(name)->null_count(); }

        template<bool append_ones, typename T>
        std::variant<MatrixXd, MatrixXf> to_eigen(T cols) const;
//        template<typename V, bool append_ones>
//        std::variant<MatrixXd, MatrixXf> to_eigen(std::initializer_list<V> cols) const {
//            return to_eigen<std::initializer_list<V>, append_ones>(cols);
//        }
        template<bool append_ones>
        std::variant<Matrix<DoubleType::c_type, Dynamic, 1+append_ones>, Matrix<FloatType::c_type, Dynamic, 1+append_ones>>
        to_eigen(int i) const;

        template<bool append_ones>
        std::variant<Matrix<DoubleType::c_type, Dynamic, 1+append_ones>, Matrix<FloatType::c_type, Dynamic, 1+append_ones>>
        to_eigen(const std::string& name) const;

        std::shared_ptr<arrow::RecordBatch> operator->();
    private:
        std::shared_ptr <arrow::Buffer> combined_bitmap_with_null() const;

//        template<typename T, bool append_ones, typename ArrowType>
//        Matrix<typename ArrowType::c_type, Dynamic, Dynamic> to_eigen_typed(T cols) const;

        template<bool append_ones, typename ArrowType>
        Matrix<typename ArrowType::c_type, Dynamic, 1+append_ones> to_eigen_typed(int i) const;

        template<bool append_ones, typename ArrowType>
        Matrix<typename ArrowType::c_type, Dynamic, 1+append_ones> to_eigen_typed(const std::string& name) const;

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

    template<typename T>
    int64_t DataFrame::null_count(T cols) const {
        static_assert(util::is_integral_container_v<T> || util::is_string_container_v<T>,
                      "null_count() only accepts integral or string containers.");
        int64_t r = 0;
        for (auto c : cols) {
            if constexpr (util::is_integral_container_v<T>)
                r += m_batch->column(c)->null_count();
            else if constexpr (util::is_string_container_v<T>)
                r += m_batch->GetColumnByName(c)->null_count();
        }
        return r;
    }

    template<typename T>
    Buffer_ptr DataFrame::combined_bitmap(T cols) const {
        static_assert(util::is_integral_container_v<T> || util::is_string_container_v<T>,
                      "combined_bitmap() only accepts integral or string containers.");
        if (null_count(cols) > 0) {
            auto first_null_col = 0;
            for(auto i = 0; i < cols.size(); ++i) {
                if constexpr (util::is_integral_container_v<T>) {
                    if (m_batch->column(cols[i])->null_count() != 0) {
                        first_null_col = i;
                        break;
                    }
                }
                else if constexpr (util::is_string_container_v<T>) {
                    if (m_batch->GetColumnByName(cols[i])->null_count() != 0) {
                        first_null_col = i;
                        break;
                    }
                }
            }

            auto res = [this, &cols, first_null_col]() -> auto {
                if constexpr (util::is_integral_container_v<T>)
                    return Buffer::Copy(m_batch->column(cols[first_null_col])->null_bitmap(), arrow::default_cpu_memory_manager());
                else if constexpr (util::is_string_container_v<T>)
                    return Buffer::Copy(m_batch->GetColumnByName(cols[first_null_col])->null_bitmap(), arrow::default_cpu_memory_manager());
            };

            auto bitmap = std::move(res).ValueOrDie();

            for(int i = first_null_col + 1; i < cols.size(); ++i) {
                auto col = [this, &cols, i]() -> auto {
                    if constexpr (util::is_integral_container_v<T>)
                        return m_batch->column(cols[i]);
                    else if constexpr (util::is_string_container_v<T>)
                        return m_batch->GetColumnByName(cols[i]);
                };

                if (col->null_count()) {
                    auto other_bitmap = col->null_bitmap();

                    arrow::internal::BitmapAnd(bitmap->data(), 0,
                                               other_bitmap->data(), 0,
                                               m_batch->num_rows(),
                                               0, bitmap->mutable_data());
                }
            }
        } else {
            return nullptr;
        }
    }


//    template<typename T, bool append_ones, typename ArrowType>
//    Matrix<typename ArrowType::c_type, Dynamic, Dynamic> DataFrame::to_eigen_typed(T cols) const {
//        static_assert(util::is_integral_container_v<T> || util::is_string_container_v<T>,
//                      "to_eigen_typed() only accepts integral or string containers.");
//        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, Dynamic>;
//        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
//
//        auto buffer_bitmap = combined_bitmap(cols);
//
//        if (buffer_bitmap) {
//
//            auto rows = m_batch->num_rows();
//            auto valid_rows = util::bit_util::non_null_count(buffer_bitmap, rows);
//
//            auto m = [&cols, valid_rows]() {
//                if constexpr(append_ones) return MatrixType(valid_rows, cols.size()+1);
//                else return MatrixType(valid_rows, cols.size());
//            };
//
//            auto m_ptr = m.data();
//
//            auto offset_ptr = 0;
//            if constexpr(append_ones) {
//                std::fill_n(m_ptr, valid_rows, 1);
//                offset_ptr += valid_rows;
//            }
//
//            auto combined_bitmap = buffer_bitmap->data();
//            for (auto i = 0; i < cols.size(); ++i) {
//                auto col = [this, &cols, i]() -> auto {
//                    if constexpr (util::is_integral_container_v<T>)
//                        return m_batch->column(cols[i]);
//                    else if constexpr (util::is_string_container_v<T>)
//                        return m_batch->GetColumnByName(cols[i]);
//                };
//
//                auto dwn_col = std::static_pointer_cast<ArrayType>(col);
//                auto raw_values = dwn_col->raw_values();
//
//                auto k = 0;
//
//                for (auto j = 0; j < rows; ++j) {
//                    if (arrow::BitUtil::GetBit(combined_bitmap, j))
//                        m_ptr[offset_ptr + k++] = raw_values[j];
//                }
//
//                offset_ptr += valid_rows;
//            }
//
//            return m;
//
//        } else {
//
//            auto rows = m_batch->num_rows();
//            auto m = [&cols, rows]() {
//                if constexpr(append_ones) return MatrixType(rows, cols.size()+1);
//                else return MatrixType(rows, cols.size());
//            };
//
//            auto m_ptr = m.data();
//
//            auto offset_ptr = 0;
//            if constexpr(append_ones) {
//                std::fill_n(m_ptr, rows, 1);
//                offset_ptr += rows;
//            }
//
//            for (auto i = 0; i < cols.size(); ++i) {
//                auto col = [this, &cols, i]() -> auto {
//                    if constexpr (util::is_integral_container_v<T>)
//                        return m_batch->column(cols[i]);
//                    else if constexpr (util::is_string_container_v<T>)
//                        return m_batch->GetColumnByName(cols[i]);
//                };
//
//                auto dwn_col = std::static_pointer_cast<ArrayType>(col);
//
//                std::memcpy(m_ptr + offset_ptr, dwn_col->raw_values(), sizeof(typename ArrowType::c_type)*rows);
//                offset_ptr += rows;
//            }
//
//            return m;
//        }
//
//    }

    template<bool append_ones, typename T>
    std::variant<MatrixXd, MatrixXf> DataFrame::to_eigen(T cols) const {
        static_assert(util::is_integral_container_v<T> || util::is_string_container_v<T>,
                      "to_eigen() only accepts integral or string containers.");

        std::cout << "To eigen list" << std::endl;
        auto dt_id = [this, &cols]() {
            if constexpr(util::is_integral_container_v<T>) return m_batch->column(cols[0])->type_id();
            else if constexpr (util::is_string_container_v<T>) return m_batch->GetColumnByName(cols[0])->type_id();
        }();

        switch (dt_id) {
            case Type::DOUBLE:
//                return to_eigen_typed<T, append_ones, arrow::DoubleType>(cols);
                break;
            case Type::FLOAT:
//                return to_eigen_typed<T, append_ones, arrow::FloatType>(cols);
                break;
            default:
                throw pybind11::value_error("Only numeric data types can be transformed to Eigen matrix.");
        }
    }

    template<bool append_ones, typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1+append_ones> DataFrame::to_eigen_typed(int i) const {
        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, 1+append_ones>;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto rows = m_batch->num_rows();
        auto buffer_bitmap = m_batch->column(i)->null_bitmap();

        if(buffer_bitmap) {

            auto valid_rows = util::bit_util::non_null_count(buffer_bitmap, rows);

            MatrixType m(valid_rows, 1+append_ones);

            auto m_ptr = m.data();

            if constexpr(append_ones) {
                std::fill_n(m_ptr, valid_rows, 1);
                m_ptr += valid_rows;
            }

            auto col = m_batch->column(i);
            auto dwn_col = std::static_pointer_cast<ArrayType>(col);

            auto raw_values = dwn_col->raw_values();
            auto k = 0;
            auto combined_bitmap = buffer_bitmap->data();

            for (auto j = 0; j < rows; ++j) {
                if (arrow::BitUtil::GetBit(combined_bitmap, j))
                    m_ptr[k++] = raw_values[j];
            }

            return m;

        } else {
            auto rows = m_batch->num_rows();
            MatrixType m(rows, 1+append_ones);

            auto m_ptr = m.data();

            if constexpr(append_ones) {
                std::fill_n(m_ptr, rows, 1);
                m_ptr += rows;
            }

            auto col = m_batch->column(i);
            auto dwn_col = std::static_pointer_cast<ArrayType>(col);

            std::memcpy(m_ptr, dwn_col->raw_values(), sizeof(typename ArrowType::c_type)*rows);

            return m;
        }
    }

    template<bool append_ones>
    std::variant<Matrix<DoubleType::c_type, Dynamic, 1+append_ones>,
                 Matrix<FloatType::c_type, Dynamic, 1+append_ones>>
     DataFrame::to_eigen(int i) const {
        auto dt_id = m_batch->column(i)->type_id();
        switch (dt_id) {
            case Type::DOUBLE:
                return to_eigen_typed<append_ones, DoubleType>(i);
            case Type::FLOAT:
                return to_eigen_typed<append_ones, FloatType>(i);
            default:
                throw pybind11::value_error("Only numeric data types can be transformed to Eigen matrix.");
        }
    }


    template<bool append_ones, typename ArrowType>
    Matrix<typename ArrowType::c_type, Dynamic, 1+append_ones> DataFrame::to_eigen_typed(const std::string& name) const {
        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, 1+append_ones>;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto rows = m_batch->num_rows();
        auto buffer_bitmap = m_batch->GetColumnByName(name)->null_bitmap();

        if(buffer_bitmap) {

            auto valid_rows = util::bit_util::non_null_count(buffer_bitmap, rows);

            MatrixType m(valid_rows, 1+append_ones);

            auto m_ptr = m.data();

            if constexpr(append_ones) {
                std::fill_n(m_ptr, valid_rows, 1);
                m_ptr += valid_rows;
            }

            auto col = m_batch->GetColumnByName(name);
            auto dwn_col = std::static_pointer_cast<ArrayType>(col);

            auto raw_values = dwn_col->raw_values();
            auto k = 0;
            auto combined_bitmap = buffer_bitmap->data();

            for (auto j = 0; j < rows; ++j) {
                if (arrow::BitUtil::GetBit(combined_bitmap, j))
                    m_ptr[k++] = raw_values[j];
            }

            return m;

        } else {
            auto rows = m_batch->num_rows();
            MatrixType m(rows, 1+append_ones);

            auto m_ptr = m.data();

//            TODO Use map if append_ones is false.
            if constexpr(append_ones) {
                std::fill_n(m_ptr, rows, 1);
                m_ptr += rows;
            }

            auto col = m_batch->GetColumnByName(name);
            auto dwn_col = std::static_pointer_cast<ArrayType>(col);

            std::memcpy(m_ptr, dwn_col->raw_values(), sizeof(typename ArrowType::c_type)*rows);

            return m;
        }
    }

    template<bool append_ones>
    std::variant<Matrix<DoubleType::c_type, Dynamic, 1+append_ones>, Matrix<FloatType::c_type, Dynamic, 1+append_ones>>
    DataFrame::to_eigen(const std::string& name) const {
        auto dt_id = m_batch->GetColumnByName(name)->type_id();
        switch (dt_id) {
            case Type::DOUBLE:
                return to_eigen_typed<append_ones, DoubleType>(name);
            case Type::FLOAT:
                return to_eigen_typed<append_ones, FloatType>(name);
            default:
                throw pybind11::value_error("Only numeric data types can be transformed to Eigen matrix.");
        }
    }

}


#endif //PGM_DATASET_DATASET_HPP
