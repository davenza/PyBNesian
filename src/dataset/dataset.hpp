#ifndef PGM_DATASET_DATASET_HPP
#define PGM_DATASET_DATASET_HPP

#include <arrow/api.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <util/parameter_traits.hpp>
#include <util/bit_util.hpp>


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

    template<typename ArrowType>
    using EigenMatrix = std::unique_ptr<Matrix<typename ArrowType::c_type, Dynamic, Dynamic>>;

    template<bool append_ones, typename ArrowType>
    using EigenVectorOrMatrix = std::unique_ptr<Matrix<typename ArrowType::c_type, Dynamic, append_ones+1>>;
    template<bool append_ones, typename ArrowType, bool contains_null>
    using MapOrMatrixType = typename std::conditional_t<append_ones || contains_null,
                                            EigenVectorOrMatrix<append_ones, ArrowType>,
                                            std::unique_ptr<Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>>>;

    class DataFrame {
    public:
        DataFrame(std::shared_ptr<arrow::RecordBatch> rb);

        std::vector<std::string> column_names() const;

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        DataFrame loc(T cols) const;
        template<typename V>
        DataFrame loc(std::initializer_list<V> cols) const { return loc<std::initializer_list<V>>(cols); }
        Array_ptr loc(int i) const { return m_batch->column(i); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Array_ptr loc(StringType name) const { return m_batch->GetColumnByName(name); }

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        Buffer_ptr combined_bitmap(T cols) const { return combined_bitmap(cols.begin(), cols.end()); }
        template<typename V>
        Buffer_ptr combined_bitmap(std::initializer_list<V> cols) const { return combined_bitmap(cols.begin(), cols.end()); }
        template<int = 0>
        Buffer_ptr combined_bitmap(int i) const { return m_batch->column(i)->null_bitmap(); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Buffer_ptr combined_bitmap(StringType name) const { return m_batch->GetColumnByName(name)->null_bitmap(); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        Buffer_ptr combined_bitmap(IndexIter begin, IndexIter end) const;

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        int64_t null_count(T cols) const { return null_count(cols.begin(), cols.end()); }
        template<typename V>
        int64_t null_count(std::initializer_list<V> cols) const { return null_count(cols.begin(), cols.end()); }
        template<int = 0>
        int64_t null_count(int i) const { return m_batch->column(i)->null_count(); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        int64_t null_count(StringType name) const { return m_batch->GetColumnByName(name)->null_count(); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        int64_t null_count(IndexIter begin, IndexIter end) const;

        template<bool append_ones, typename ArrowType, bool contains_null, typename T, util::enable_if_index_container_t<T, int> = 0>
        EigenMatrix<ArrowType> to_eigen(T cols) const { return to_eigen(cols.begin(), cols.end()); }
        template<bool append_ones, typename ArrowType, bool contains_null, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        EigenMatrix<ArrowType> to_eigen(IndexIter begin, IndexIter end) const;
        
        template<bool append_ones, typename ArrowType, typename T, util::enable_if_index_container_t<T, int> = 0>
        EigenMatrix<ArrowType> to_eigen(T cols, Buffer_ptr bitmap) const { return to_eigen(cols.begin(), cols.end(), bitmap); }
        template<bool append_ones, typename ArrowType, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        EigenMatrix<ArrowType> to_eigen(IndexIter begin, IndexIter end, Buffer_ptr bitmap) const;

        template<bool append_ones, typename ArrowType, bool contains_null>
        MapOrMatrixType<append_ones, ArrowType, contains_null> to_eigen(int i) const;
        template<bool append_ones, typename ArrowType>
        EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(int i, Buffer_ptr bitmap) const;

        template<bool append_ones, typename ArrowType, bool contains_null, typename StringType,
                util::enable_if_stringable_t<StringType, int> = 0>
        MapOrMatrixType<append_ones, ArrowType, contains_null> to_eigen(StringType name) const;
        template<bool append_ones, typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(StringType name, Buffer_ptr bitmap) const;

        std::shared_ptr<arrow::RecordBatch> operator->() const;
    private:
        std::shared_ptr <arrow::Buffer> combined_bitmap_with_null() const;

        template<bool append_ones, typename ArrowType, bool contains_null>
        MapOrMatrixType<append_ones, ArrowType, contains_null> to_eigen(Array_ptr col) const;

        template<bool append_ones, typename ArrowType>
        EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(Array_ptr col, Buffer_ptr bitmap) const;

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

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    int64_t DataFrame::null_count(IndexIter begin, IndexIter end) const {
        static_assert(util::is_integral_iterator_v<IndexIter> || util::is_string_iterator_v<IndexIter>,
                      "null_count() only accepts integral or string containers.");
        int64_t r = 0;

        for (auto it = begin; it != end; it++) {
            if constexpr (util::is_integral_iterator_v<IndexIter>)
                r += m_batch->column(*it)->null_count();
            else if constexpr (util::is_string_iterator_v<IndexIter>)
                r += m_batch->GetColumnByName(*it)->null_count();
        }
        return r;
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    Buffer_ptr DataFrame::combined_bitmap(IndexIter begin, IndexIter end) const {
        static_assert(util::is_integral_iterator_v<IndexIter> || util::is_string_iterator_v<IndexIter>,
                      "combined_bitmap() only accepts integral or string containers.");
        if (null_count(begin, end) > 0) {

            IndexIter first_null_col = end;

            for(auto it = begin; it < end; ++it) {
                if constexpr (util::is_integral_iterator_v<IndexIter>) {
                    if (m_batch->column(*it)->null_count() != 0) {
                        first_null_col = it;
                        break;
                    }
                }
                else if constexpr (util::is_string_iterator_v<IndexIter>) {
                    if (m_batch->GetColumnByName(*it)->null_count() != 0) {
                        first_null_col = it;
                        break;
                    }
                }
            }

            auto res = [this, first_null_col]() -> auto {
                if constexpr (util::is_integral_iterator_v<IndexIter>)
                    return Buffer::Copy(m_batch->column(*first_null_col)->null_bitmap(), arrow::default_cpu_memory_manager());
                else if constexpr (util::is_string_iterator_v<IndexIter>)
                    return Buffer::Copy(m_batch->GetColumnByName(*first_null_col)->null_bitmap(), arrow::default_cpu_memory_manager());
            }();

            auto bitmap = std::move(res).ValueOrDie();

            for(auto it = first_null_col + 1; it < end; ++it) {
                auto col = [this, it]() -> auto {
                    if constexpr (util::is_integral_iterator_v<IndexIter>)
                        return m_batch->column(*it);
                    else if constexpr (util::is_string_iterator_v<IndexIter>)
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


    template<bool append_ones, typename ArrowType, bool contains_null>
    MapOrMatrixType<append_ones, ArrowType, contains_null> DataFrame::to_eigen(int i) const {
        auto col = m_batch->column(i);
        return to_eigen<append_ones, ArrowType, contains_null>(col);
    }


    template<bool append_ones, typename ArrowType, bool contains_null, typename StringType,
            util::enable_if_stringable_t<StringType, int> = 0>
    MapOrMatrixType<append_ones, ArrowType, contains_null> DataFrame::to_eigen(StringType name) const {
        auto col = m_batch->GetColumnByName(name);
        return to_eigen<append_ones, ArrowType, contains_null>(col);
    }


    template<bool append_ones, typename ArrowType, bool contains_null>
    MapOrMatrixType<append_ones, ArrowType, contains_null> DataFrame::to_eigen(Array_ptr c) const {

        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, 1 + append_ones>;
        using MapType = Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto rows = c->length();

        if constexpr(contains_null) {
            auto bitmap = c->null_bitmap();

            return to_eigen<append_ones, ArrowType>(c, bitmap);
        } else {
            auto dwn_col = std::static_pointer_cast<ArrayType>(c);
            if constexpr(append_ones) {
                auto m = std::make_unique<MatrixType>(rows, 1 + append_ones);
                auto m_ptr = m->data();

                std::fill_n(m_ptr, rows, 1);
                m_ptr += rows;

                std::memcpy(m_ptr, dwn_col->raw_values(), sizeof(typename ArrowType::c_type) * rows);
                return m;
            } else {
                return std::make_unique<MapType>(dwn_col->raw_values(), rows);
            }
        }
    }

    template<bool append_ones, typename ArrowType>
    EigenVectorOrMatrix<append_ones, ArrowType> DataFrame::to_eigen(int i, Buffer_ptr bitmap) const {
        auto col = m_batch->column(i);
        return to_eigen<append_ones, ArrowType>(col, bitmap);
    }

    template<bool append_ones, typename ArrowType, typename StringType,
            util::enable_if_stringable_t<StringType, int> = 0>
    EigenVectorOrMatrix<append_ones, ArrowType> DataFrame::to_eigen(StringType name, Buffer_ptr bitmap) const {
        auto col = m_batch->GetColumnByName(name);
        return to_eigen<append_ones, ArrowType>(col, bitmap);
    }


    template<bool append_ones, typename ArrowType>
    EigenVectorOrMatrix<append_ones, ArrowType>
    DataFrame::to_eigen(Array_ptr c, Buffer_ptr bitmap) const {
        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, 1 + append_ones>;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto rows = c->length();

        auto valid_rows = util::bit_util::non_null_count(bitmap, rows);
        auto m = std::make_unique<MatrixType>(valid_rows, 1 + append_ones);
        auto m_ptr = m->data();

        if constexpr(append_ones) {
            std::fill_n(m_ptr, valid_rows, 1);
            m_ptr += valid_rows;
        }

        auto dwn_col = std::static_pointer_cast<ArrayType>(c);
        auto raw_values = dwn_col->raw_values();

        auto k = 0;
        auto combined_bitmap = bitmap->data();

        for (auto j = 0; j < rows; ++j) {
            if (arrow::BitUtil::GetBit(combined_bitmap, j))
                m_ptr[k++] = raw_values[j];
        }
        return m;
    }

    template<bool append_ones, typename ArrowType, bool contains_null, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> DataFrame::to_eigen(IndexIter begin, IndexIter end) const {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        if constexpr (contains_null) {
            auto bitmap = combined_bitmap(begin, end);
            return to_eigen<append_ones, ArrowType>(begin, end, bitmap);
        } else {
            auto rows = m_batch->num_rows();
            auto ncols = std::distance(begin, end);
            auto m = [rows, ncols]() {
                if constexpr(append_ones) return std::make_unique<typename EigenMatrix<ArrowType>::element_type>(rows, ncols+1);
                else return std::make_unique<typename EigenMatrix<ArrowType>::element_type>(rows, ncols);
            }();

            auto m_ptr = m->data();

            auto offset_ptr = 0;
            if constexpr(append_ones) {
                std::fill_n(m_ptr, rows, 1);
                offset_ptr += rows;
            }

            for(auto col_index = begin; col_index != end; ++col_index) {
                auto col = [this, &col_index]() -> auto {
                    if constexpr (util::is_integral_iterator_v<IndexIter>)
                        return m_batch->column(*col_index);
                    else if constexpr (util::is_string_iterator_v<IndexIter>)
                        return m_batch->GetColumnByName(*col_index);
                }();

                auto dwn_col = std::static_pointer_cast<ArrayType>(col);

                std::memcpy(m_ptr + offset_ptr, dwn_col->raw_values(), sizeof(typename ArrowType::c_type)*rows);
                offset_ptr += rows;
            }
            return std::move(m);
        }
    }

    template<bool append_ones, typename ArrowType, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> DataFrame::to_eigen(IndexIter begin, IndexIter end, Buffer_ptr bitmap) const {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        auto rows = m_batch->num_rows();
        auto valid_rows = util::bit_util::non_null_count(bitmap, rows);

        auto ncols = std::distance(begin, end);
        auto m = [valid_rows, ncols]() {
            if constexpr(append_ones) return std::move(std::make_unique<typename EigenMatrix<ArrowType>::element_type>(valid_rows, ncols+1));
            else return std::move(std::make_unique<typename EigenMatrix<ArrowType>::element_type>(valid_rows, ncols));
        }();

        auto m_ptr = m->data();

        auto offset_ptr = 0;
        if constexpr(append_ones) {
            std::fill_n(m_ptr, valid_rows, 1);
            offset_ptr += valid_rows;
        }

        auto bitmap_data = bitmap->data();

        for (auto col_index = begin; col_index != end; ++col_index) {
            auto col = [this, &col_index]() -> auto {
                if constexpr (util::is_integral_iterator_v<IndexIter>)
                    return m_batch->column(*col_index);
                else if constexpr (util::is_string_iterator_v<IndexIter>)
                    return m_batch->GetColumnByName(*col_index);
            }();

            auto dwn_col = std::static_pointer_cast<ArrayType>(col);
            auto raw_values = dwn_col->raw_values();

            auto k = 0;
            for (auto j = 0; j < rows; ++j) {
                if (arrow::BitUtil::GetBit(bitmap_data, j))
                    m_ptr[offset_ptr + k++] = raw_values[j];
            }
            offset_ptr += valid_rows;
        }

        return std::move(m);
    }

}


#endif //PGM_DATASET_DATASET_HPP
