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


using Array_ptr = std::shared_ptr<arrow::Array>;
using Array_vector =  std::vector<Array_ptr>;
using Array_iterator =  Array_vector::iterator;
using Buffer_ptr = std::shared_ptr<arrow::Buffer>;

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




    int64_t null_count(Array_iterator begin, Array_iterator end);
    Buffer_ptr combined_bitmap(Array_iterator begin, Array_iterator end);


    template<bool append_ones, typename ArrowType>
    inline typename ArrowType::c_type* fill_ones(typename ArrowType::c_type* ptr, int rows [[maybe_unused]]) {
        if constexpr(append_ones) {
            std::fill_n(ptr, rows, 1);
            return ptr + rows;
        } else {
            return ptr;
        }
    }

    template<typename ArrowType>
    typename ArrowType::c_type* fill_data_bitmap(typename ArrowType::c_type* ptr, 
                                                 Array_ptr input_array, 
                                                 const uint8_t* bitmap_data, 
                                                 int total_rows) 
    {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        auto dwn_col = std::static_pointer_cast<ArrayType>(input_array);
        auto raw_values = dwn_col->raw_values();

        auto k = 0;
        for (auto j = 0; j < total_rows; ++j) {
            if (arrow::BitUtil::GetBit(bitmap_data, j))
                ptr[k++] = raw_values[j];
        }
        return ptr + k;
    }

    template<bool append_ones, typename ArrowType>
    EigenMatrix<ArrowType> to_eigen(Array_iterator begin, Array_iterator end, Buffer_ptr bitmap) {
        auto ncols = std::distance(begin, end);

        if (ncols == 0) {
            return nullptr;
        }

        auto rows = (*begin)->length();
        auto valid_rows = util::bit_util::non_null_count(bitmap, rows);

        auto m = [valid_rows, ncols]() {
            if constexpr(append_ones) return std::move(std::make_unique<typename EigenMatrix<ArrowType>::element_type>(valid_rows, ncols+1));
            else return std::move(std::make_unique<typename EigenMatrix<ArrowType>::element_type>(valid_rows, ncols));
        }();

        auto m_ptr = m->data();
        m_ptr = fill_ones<append_ones, ArrowType>(m_ptr, valid_rows);
        const uint8_t* bitmap_data = bitmap->data();

        for (auto it = begin; it != end; ++it) {
            m_ptr = fill_data_bitmap<ArrowType>(m_ptr, *it, bitmap_data, rows);
        }

        return std::move(m);
    }

    template<bool append_ones, typename ArrowType, bool contains_null>
    EigenMatrix<ArrowType> to_eigen(Array_iterator begin, Array_iterator end) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        if constexpr (contains_null) {
            auto bitmap = combined_bitmap(begin, end);
            return to_eigen<append_ones, ArrowType>(begin, end, bitmap);
        } else {
            auto ncols = std::distance(begin, end);
            if (ncols == 0) {
                return nullptr;
            }

            auto rows = (*begin)->length();
            auto m = [rows, ncols]() {
                if constexpr(append_ones) return std::make_unique<typename EigenMatrix<ArrowType>::element_type>(rows, ncols+1);
                else return std::make_unique<typename EigenMatrix<ArrowType>::element_type>(rows, ncols);
            }();

            auto m_ptr = m->data();
            m_ptr = fill_ones<append_ones, ArrowType>(m_ptr, rows);

            for(auto it = begin; it != end; ++it) {
                auto dwn_col = std::static_pointer_cast<ArrayType>(*it);
                std::memcpy(m_ptr, dwn_col->raw_values(), sizeof(typename ArrowType::c_type)*rows);
                m_ptr += rows;
            }
            return std::move(m);
        }
    }

    template<bool append_ones, typename ArrowType>
    EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(Array_ptr col, Buffer_ptr bitmap) {
        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, 1 + append_ones>;
        auto rows = col->length();

        auto valid_rows = util::bit_util::non_null_count(bitmap, rows);
        auto m = std::make_unique<MatrixType>(valid_rows, 1 + append_ones);
        auto m_ptr = m->data();
        m_ptr = fill_ones<append_ones, ArrowType>(m_ptr, valid_rows);

        const uint8_t* bitmap_data = bitmap->data();
        fill_data_bitmap<ArrowType>(m_ptr, col, bitmap_data, rows);

        return m;
    }

    template<bool append_ones, typename ArrowType, bool contains_null>
    MapOrMatrixType<append_ones, ArrowType, contains_null> to_eigen(Array_ptr col) {
        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, 1 + append_ones>;
        using MapType = Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto rows = col->length();

        if constexpr (contains_null) {
            auto bitmap = col->null_bitmap();
            return to_eigen<append_ones, ArrowType>(col, bitmap);
        } else {
            auto dwn_col = std::static_pointer_cast<ArrayType>(col);
            if constexpr (append_ones) {
                auto m = std::make_unique<MatrixType>(rows, 1 + append_ones);
                auto m_ptr = m->data();
                m_ptr = fill_ones<append_ones, ArrowType>(m_ptr, rows);
                std::memcpy(m_ptr, dwn_col->raw_values(), sizeof(typename ArrowType::c_type) * rows);
                return m;
            } else {
                return std::make_unique<MapType>(dwn_col->raw_values(), rows);
            }
        }
    }

    class DataFrame {
    public:

        DataFrame() : m_batch() {};

        DataFrame(std::shared_ptr<arrow::RecordBatch> rb);

        std::vector<std::string> column_names() const;

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        DataFrame loc(T cols) const;
        template<typename V>
        DataFrame loc(std::initializer_list<V> cols) const { return loc<std::initializer_list<V>>(cols); }
        Array_ptr loc(int i) const { return m_batch->column(i); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Array_ptr loc(StringType name) const { return m_batch->GetColumnByName(name); }

        
        Buffer_ptr combined_bitmap() const { Array_vector cols = m_batch->columns(); return dataset::combined_bitmap(cols.begin(), cols.end()); }
        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        Buffer_ptr combined_bitmap(T cols) const { Array_vector v = indices_to_columns(cols); return dataset::combined_bitmap(v.begin(), v.end()); }
        template<typename V>
        Buffer_ptr combined_bitmap(std::initializer_list<V> cols) const { return combined_bitmap(cols.begin(), cols.end()); }
        Buffer_ptr combined_bitmap(int i) const { return m_batch->column(i)->null_bitmap(); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Buffer_ptr combined_bitmap(StringType name) const { return m_batch->GetColumnByName(name)->null_bitmap(); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        Buffer_ptr combined_bitmap(IndexIter begin, IndexIter end) const { Array_vector v = indices_to_columns(begin, end);
                                                                             return dataset::combined_bitmap(v.begin(), v.end());
                                                                         }

        int64_t null_count() const { auto cols = m_batch->columns(); return dataset::null_count(cols.begin(), cols.end()); }
        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        int64_t null_count(T cols) const { Array_vector v = indices_to_columns(cols); return dataset::null_count(v.begin(), v.end()); }
        template<typename V>
        int64_t null_count(std::initializer_list<V> cols) const { return null_count(cols.begin(), cols.end()); }
        int64_t null_count(int i) const { return m_batch->column(i)->null_count(); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        int64_t null_count(StringType name) const { return m_batch->GetColumnByName(name)->null_count(); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        int64_t null_count(IndexIter begin, IndexIter end) const { Array_vector v = indices_to_columns(begin, end); return dataset::null_count(v.begin(), v.end()); }

        template<bool append_ones, typename ArrowType, bool contains_null, typename T, util::enable_if_index_container_t<T, int> = 0>
        EigenMatrix<ArrowType> to_eigen(T cols) const { 
            Array_vector v = indices_to_columns(cols); 
            return dataset::to_eigen<append_ones, ArrowType, contains_null>(v.begin(), v.end()); 
        }
        template<bool append_ones, typename ArrowType, bool contains_null, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        EigenMatrix<ArrowType> to_eigen(IndexIter begin, IndexIter end) const {
            Array_vector v = indices_to_columns(begin, end); 
            return dataset::to_eigen<append_ones, ArrowType, contains_null>(v.begin(), v.end());
        }
        
        template<bool append_ones, typename ArrowType, typename T, util::enable_if_index_container_t<T, int> = 0>
        EigenMatrix<ArrowType> to_eigen(T cols, Buffer_ptr bitmap) const { 
            Array_vector v = indices_to_columns(cols); 
            return dataset::to_eigen<append_ones, ArrowType>(v.begin(), v.end(), bitmap); 
        }

        template<bool append_ones, typename ArrowType, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        EigenMatrix<ArrowType> to_eigen(IndexIter begin, IndexIter end, Buffer_ptr bitmap) const {
            Array_vector v = indices_to_columns(begin, end); 
            return dataset::to_eigen<append_ones, ArrowType>(v.begin(), v.end(), bitmap); 
        }

        template<bool append_ones, typename ArrowType, bool contains_null>
        MapOrMatrixType<append_ones, ArrowType, contains_null> to_eigen(int i) const {
            auto col = m_batch->column(i);
            return dataset::to_eigen<append_ones, ArrowType, contains_null>(col);
        }

        template<bool append_ones, typename ArrowType>
        EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(int i, Buffer_ptr bitmap) const {
            auto col = m_batch->column(i);
            return dataset::to_eigen<append_ones, ArrowType>(col, bitmap);
        }

        template<bool append_ones, typename ArrowType, bool contains_null, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        MapOrMatrixType<append_ones, ArrowType, contains_null> to_eigen(StringType name) const {
            auto col = m_batch->GetColumnByName(name);
            return dataset::to_eigen<append_ones, ArrowType, contains_null>(col);
        }

        template<bool append_ones, typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(StringType name, Buffer_ptr bitmap) const {
            auto col = m_batch->GetColumnByName(name);
            return dataset::to_eigen<append_ones, ArrowType>(col, bitmap);
        }

        std::shared_ptr<arrow::RecordBatch> operator->() const;

        friend std::pair<DataFrame, DataFrame> generate_cv_pair(const DataFrame& df, int fold, const std::vector<int>& indices, 
                                                                const std::vector<std::vector<int>::iterator>& test_limits);
    private:
        std::shared_ptr <arrow::Buffer> combined_bitmap_with_null() const;


        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        Array_vector indices_to_columns(T cols) const;

        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        Array_vector indices_to_columns(IndexIter begin, IndexIter end) const;

        std::shared_ptr <arrow::RecordBatch> m_batch;
    };

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    Array_vector DataFrame::indices_to_columns(T cols) const {
        Array_vector v;
        v.reserve(cols.size());

        for (auto c : cols) {
            if constexpr (util::is_integral_container_v<T>)
                v.push_back(m_batch->column(c));
            else if constexpr (util::is_string_container_v<T>)
                v.push_back(m_batch->GetColumnByName(c));
        }

        return v;    
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    Array_vector DataFrame::indices_to_columns(IndexIter begin, IndexIter end) const {
        Array_vector v;
        v.reserve(std::distance(begin, end));

        for (auto it = begin; it != end; ++it) {
            if constexpr (util::is_integral_iterator_v<IndexIter>)
                v.push_back(m_batch->column(*it));
            else if constexpr (util::is_string_iterator_v<IndexIter>)
                v.push_back(m_batch->GetColumnByName(*it));
        }

        return v;    
    }


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
                new_cols.push_back(m_batch->column(c));
                new_fields.push_back(field);
            }
            else if constexpr (util::is_string_container_v<T>) {
                auto field = m_batch->schema()->GetFieldByName(c);
                new_cols.push_back(m_batch->GetColumnByName(c));
                new_fields.push_back(field);
            }
        }

        auto new_schema = std::make_shared<arrow::Schema>(new_fields);
        return DataFrame(arrow::RecordBatch::Make(new_schema, m_batch->num_rows(), new_cols));
    }
}


#endif //PGM_DATASET_DATASET_HPP
