#ifndef PGM_DATASET_DATASET_HPP
#define PGM_DATASET_DATASET_HPP

#include <variant>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
//#include <arrow/python/pyarrow.h>
#include <arrow/api.h>
#include <util/util.hpp>
#include <util/bit_util.hpp>
#include <util/variant_util.hpp>

namespace py = pybind11;

using Eigen::MatrixXd, Eigen::MatrixXf, Eigen::VectorXd, Eigen::VectorXf, Eigen::Matrix, Eigen::Dynamic, Eigen::Map,
        Eigen::DenseBase;
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

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        DataFrame loc(T cols) const;
        template<typename V>
        DataFrame loc(std::initializer_list<V> cols) const { return loc<std::initializer_list<V>>(cols); }
        Column loc(int i) const { return m_batch->column(i); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Column loc(StringType name) const { return m_batch->GetColumnByName(name); }

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        Buffer_ptr combined_bitmap(T cols) const;
        template<typename V>
        Buffer_ptr combined_bitmap(std::initializer_list<V> cols) const { return combined_bitmap<std::initializer_list<V>>(cols); }
        template<int = 0>
        Buffer_ptr combined_bitmap(int i) const { return m_batch->column(i)->null_bitmap(); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Buffer_ptr combined_bitmap(StringType name) const { return m_batch->GetColumnByName(name)->null_bitmap(); }

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        int64_t null_count(T cols) const;
        template<typename V>
        int64_t null_count(std::initializer_list<V> cols) const { return null_count<std::initializer_list<V>>(cols); }
        template<int = 0>
        int64_t null_count(int i) const { return m_batch->column(i)->null_count(); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        int64_t null_count(StringType name) const { return m_batch->GetColumnByName(name)->null_count(); }


        using ReturnedEigenVector = std::variant<
                                        std::unique_ptr<Matrix<double, Dynamic, 2>>, // append ones
                                        std::unique_ptr<Matrix<double, Dynamic, 1>>, // !append_ones but null_bitmap != nullptr
                                        std::unique_ptr<Matrix<float, Dynamic, 2>>, // append ones
                                        std::unique_ptr<Matrix<float, Dynamic, 1>>, // !append_ones but null_bitmap != nullptr
                                        std::unique_ptr<Map<const Matrix<double, Dynamic, 1>>>, // !append_ones and null_bitmap == nullptr
                                        std::unique_ptr<Map<const Matrix<float, Dynamic, 1>>> // !append_ones and null_bitmap == nullptr
                                    >;

//        FIXME: Keeping Map may not worth it. Check benchmarks.
        using ReturnedEigenMatrix = std::variant<
                                        std::unique_ptr<MatrixXd>,
                                        std::unique_ptr<MatrixXf>,
//                                        If cols.size() == 1, same variants as the ReturnedEigenVector
                                        std::unique_ptr<Matrix<double, Dynamic, 2>>, // append ones
                                        std::unique_ptr<Matrix<double, Dynamic, 1>>, // !append_ones but null_bitmap != nullptr
                                        std::unique_ptr<Matrix<float, Dynamic, 2>>, // append ones
                                        std::unique_ptr<Matrix<float, Dynamic, 1>>, // !append_ones but null_bitmap != nullptr
                                        std::unique_ptr<Map<const Matrix<double, Dynamic, 1>>>, // !append_ones and null_bitmap == nullptr
                                        std::unique_ptr<Map<const Matrix<float, Dynamic, 1>>> // !append_ones and null_bitmap == nullptr
                                    >;

        template<bool append_ones, typename T, util::enable_if_index_container_t<T, int> = 0>
        ReturnedEigenMatrix to_eigen(T cols) const;
        template<bool append_ones, typename V>
        ReturnedEigenMatrix to_eigen(std::initializer_list<V> cols) const {
            return to_eigen<append_ones, std::initializer_list<V>>(cols);
        }
        template<bool append_ones, int = 0>
        ReturnedEigenVector to_eigen(int i) const;

        template<bool append_ones, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        ReturnedEigenVector to_eigen(StringType name) const;

        std::shared_ptr<arrow::RecordBatch> operator->();
    private:
        std::shared_ptr <arrow::Buffer> combined_bitmap_with_null() const;

        template<bool append_ones, typename T, typename ArrowType, util::enable_if_index_container_t<T, int> = 0>
        ReturnedEigenMatrix to_eigen_typed(T cols) const;

        template<bool append_ones, typename ArrowType>
        ReturnedEigenVector to_eigen_typed(Array_ptr c) const;

        std::shared_ptr <arrow::RecordBatch> m_batch;
    };

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    DataFrame DataFrame::loc(T cols) const {
        static_assert(util::is_integral_container_v<T> || util::is_string_container_v<T>,
                      "loc() only accepts integral or string containers.");
        auto size = cols.size();

        std::vector<std::shared_ptr<arrow::Field>> new_fields;
        new_fields.reserve(size);

        std::vector<Array_ptr> new_cols;
        new_cols.reserve(size);
        for (auto &c : cols) {
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

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    int64_t DataFrame::null_count(T cols) const {
        static_assert(util::is_integral_container_v<T> || util::is_string_container_v<T>,
                      "null_count() only accepts integral or string containers.");
        int64_t r = 0;
        for (auto &c : cols) {
            if constexpr (util::is_integral_container_v<T>)
                r += m_batch->column(c)->null_count();
            else if constexpr (util::is_string_container_v<T>)
                r += m_batch->GetColumnByName(c)->null_count();
        }
        return r;
    }

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    Buffer_ptr DataFrame::combined_bitmap(T cols) const {
        static_assert(util::is_integral_container_v<T> || util::is_string_container_v<T>,
                      "combined_bitmap() only accepts integral or string containers.");
        if (null_count(cols) > 0) {

            typename T::iterator first_null_col = cols.end();

            for(auto it = cols.begin(); it < cols.end(); ++it) {
                if constexpr (util::is_integral_container_v<T>) {
                    if (m_batch->column(*it)->null_count() != 0) {
                        first_null_col = it;
                        break;
                    }
                }
                else if constexpr (util::is_string_container_v<T>) {
                    if (m_batch->GetColumnByName(*it)->null_count() != 0) {
                        first_null_col = it;
                        break;
                    }
                }
            }

            auto res = [this, first_null_col]() -> auto {
                if constexpr (util::is_integral_container_v<T>)
                    return Buffer::Copy(m_batch->column(*first_null_col)->null_bitmap(), arrow::default_cpu_memory_manager());
                else if constexpr (util::is_string_container_v<T>)
                    return Buffer::Copy(m_batch->GetColumnByName(*first_null_col)->null_bitmap(), arrow::default_cpu_memory_manager());
            }();

            auto bitmap = std::move(res).ValueOrDie();

            for(auto it = first_null_col + 1; it < cols.end(); ++it) {
                auto col = [this, it]() -> auto {
                    if constexpr (util::is_integral_container_v<T>)
                        return m_batch->column(*it);
                    else if constexpr (util::is_string_container_v<T>)
                        return m_batch->GetColumnByName(*it);
                }();

                if (col->null_count()) {
                    auto other_bitmap = col->null_bitmap();

                    arrow::internal::BitmapAnd(bitmap->data(), 0,
                                               other_bitmap->data(), 0,
                                               m_batch->num_rows(),
                                               0, bitmap->mutable_data());
                }
            }
            return bitmap;
        } else {
            return nullptr;
        }
    }


    template<bool append_ones, typename T, typename ArrowType, util::enable_if_index_container_t<T, int> = 0>
    DataFrame::ReturnedEigenMatrix DataFrame::to_eigen_typed(T cols) const {
//        TODO: Review static asserts.
//        static_assert(util::is_integral_container_v<T> || util::is_string_container_v<T>,
//                      "to_eigen_typed() only accepts integral or string containers.");
        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, Dynamic>;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto buffer_bitmap = combined_bitmap(cols);

        if (buffer_bitmap) {

            auto rows = m_batch->num_rows();
            auto valid_rows = util::bit_util::non_null_count(buffer_bitmap, rows);

            auto m = [&cols, valid_rows]() {
                if constexpr(append_ones) return std::make_unique<MatrixType>(valid_rows, cols.size()+1);
                else return std::make_unique<MatrixType>(valid_rows, cols.size());
            }();

            auto m_ptr = m->data();

            auto offset_ptr = 0;
            if constexpr(append_ones) {
                std::fill_n(m_ptr, valid_rows, 1);
                offset_ptr += valid_rows;
            }

            auto combined_bitmap = buffer_bitmap->data();

            for (auto &col_index : cols) {
                auto col = [this, &col_index]() -> auto {
                    if constexpr (util::is_integral_container_v<T>)
                        return m_batch->column(col_index);
                    else if constexpr (util::is_string_container_v<T>)
                        return m_batch->GetColumnByName(col_index);
                }();

                auto dwn_col = std::static_pointer_cast<ArrayType>(col);
                auto raw_values = dwn_col->raw_values();

                auto k = 0;
                for (auto j = 0; j < rows; ++j) {
                    if (arrow::BitUtil::GetBit(combined_bitmap, j))
                        m_ptr[offset_ptr + k++] = raw_values[j];
                }
                offset_ptr += valid_rows;
            }

            return std::move(m);

        } else {

            auto rows = m_batch->num_rows();
            auto m = [&cols, rows]() {
                if constexpr(append_ones) return std::make_unique<MatrixType>(rows, cols.size()+1);
                else return std::make_unique<MatrixType>(rows, cols.size());
            }();

            auto m_ptr = m->data();

            auto offset_ptr = 0;
            if constexpr(append_ones) {
                std::fill_n(m_ptr, rows, 1);
                offset_ptr += rows;
            }

            for(auto &col_index : cols) {
                auto col = [this, &col_index]() -> auto {
                    if constexpr (util::is_integral_container_v<T>)
                        return m_batch->column(col_index);
                    else if constexpr (util::is_string_container_v<T>)
                        return m_batch->GetColumnByName(col_index);
                }();

                auto dwn_col = std::static_pointer_cast<ArrayType>(col);

                std::memcpy(m_ptr + offset_ptr, dwn_col->raw_values(), sizeof(typename ArrowType::c_type)*rows);
                offset_ptr += rows;
            }

            return std::move(m);
        }

    }

    template<bool append_ones, typename T, util::enable_if_index_container_t<T, int> = 0>
    DataFrame::ReturnedEigenMatrix DataFrame::to_eigen(T cols) const {
//        static_assert(!std::is_convertible_v<T,std::string> && (util::is_integral_container_v<T> || util::is_string_container_v<T>),
//                      "to_eigen() only accepts integral or string containers.");

        if (cols.size() == 0) {
//            TODO return empty matrix.
        }

        if (cols.size() == 1) {
            return util::variant_cast(to_eigen<append_ones>(*cols.begin()));
        }

        auto dt_id = [this, &cols]() {
            if constexpr(util::is_integral_container_v<T>) return m_batch->column(*cols.begin())->type_id();
            else if constexpr (util::is_string_container_v<T>) return m_batch->GetColumnByName(*cols.begin())->type_id();
        }();

        switch (dt_id) {
            case Type::DOUBLE:
                return to_eigen_typed<append_ones, T, arrow::DoubleType>(cols);
                break;
            case Type::FLOAT:
                return to_eigen_typed<append_ones, T, arrow::FloatType>(cols);
                break;
            default:
                throw pybind11::value_error("Only numeric data types can be transformed to Eigen matrix.");
        }
    }

    template<bool append_ones, typename ArrowType>
    DataFrame::ReturnedEigenVector DataFrame::to_eigen_typed(Array_ptr c) const {
        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, 1+append_ones>;
        using MapType = Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto rows = c->length();
        auto buffer_bitmap = c->null_bitmap();

        if(buffer_bitmap) {
            auto valid_rows = util::bit_util::non_null_count(buffer_bitmap, rows);

            auto m = std::make_unique<MatrixType>(valid_rows, 1+append_ones);

            auto m_ptr = m->data();

            if constexpr(append_ones) {
                std::fill_n(m_ptr, valid_rows, 1);
                m_ptr += valid_rows;
            }

            auto dwn_col = std::static_pointer_cast<ArrayType>(c);

            auto raw_values = dwn_col->raw_values();
            auto k = 0;
            auto combined_bitmap = buffer_bitmap->data();

            for (auto j = 0; j < rows; ++j) {
                if (arrow::BitUtil::GetBit(combined_bitmap, j))
                    m_ptr[k++] = raw_values[j];
            }
            return std::move(m);

        } else {
            auto dwn_col = std::static_pointer_cast<ArrayType>(c);
            if constexpr(append_ones) {
                auto m = std::make_unique<MatrixType>(rows, 1+append_ones);
                auto m_ptr = m->data();

                std::fill_n(m_ptr, rows, 1);
                m_ptr += rows;

                std::memcpy(m_ptr, dwn_col->raw_values(), sizeof(typename ArrowType::c_type) * rows);
                return std::move(m);
            } else {
                return std::make_unique<MapType>(dwn_col->raw_values(), rows);
            }
        }
    }

    template<bool append_ones, int = 0>
    DataFrame::ReturnedEigenVector DataFrame::to_eigen(int i) const {
        auto col = m_batch->column(i);
        auto dt_id = col->type_id();
        switch (dt_id) {
            case Type::DOUBLE:
            {
                auto tmp = to_eigen_typed<append_ones, DoubleType>(col);
                return tmp;
            }
            case Type::FLOAT:
                return to_eigen_typed<append_ones, FloatType>(col);
            default:
                throw pybind11::value_error("Only numeric data types can be transformed to Eigen matrix.");
        }
    }

    template<bool append_ones, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    DataFrame::ReturnedEigenVector DataFrame::to_eigen(StringType name) const {
        //static_assert(std::is_convertible_v<StringType,std::string>, "to_eigen() only accepts integral or std::string.");
        auto col = m_batch->GetColumnByName(name);
        auto dt_id = col->type_id();
        switch (dt_id) {
            case Type::DOUBLE:
                return to_eigen_typed<append_ones, DoubleType>(col);
            case Type::FLOAT:
                return to_eigen_typed<append_ones, FloatType>(col);
            default:
                throw pybind11::value_error("Only numeric data types can be transformed to Eigen matrix.");
        }
    }

}


#endif //PGM_DATASET_DATASET_HPP
