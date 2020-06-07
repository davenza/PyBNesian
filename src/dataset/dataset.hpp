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
using arrow::Type, arrow::Buffer, arrow::DoubleType, arrow::FloatType, arrow::RecordBatch;


using Array_ptr = std::shared_ptr<arrow::Array>;
using Array_vector =  std::vector<Array_ptr>;
using Array_iterator =  Array_vector::iterator;
using Buffer_ptr = std::shared_ptr<arrow::Buffer>;
using RecordBatch_ptr = std::shared_ptr<RecordBatch>;

namespace dataset {
    typedef py::handle PyDataset;

    bool is_pandas_dataframe(py::handle pyobject);

    std::shared_ptr<RecordBatch> to_record_batch(py::handle pyobject);

    template<typename ArrowType>
    using EigenMatrix = std::unique_ptr<Matrix<typename ArrowType::c_type, Dynamic, Dynamic>>;

    template<bool append_ones, typename ArrowType>
    using EigenVectorOrMatrix = std::unique_ptr<Matrix<typename ArrowType::c_type, Dynamic, append_ones+1>>;
    template<typename ArrowType>
    using MapType = std::unique_ptr<Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>>;
    template<bool append_ones, typename ArrowType, bool contains_null>
    using MapOrMatrixType = typename std::conditional_t<append_ones || contains_null,
                                            EigenVectorOrMatrix<append_ones, ArrowType>,
                                            MapType<ArrowType>
                                            >;

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
    EigenMatrix<ArrowType> to_eigen(Buffer_ptr bitmap, Array_iterator begin, Array_iterator end) {
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
            return to_eigen<append_ones, ArrowType>(bitmap, begin, end);
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
    EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(Buffer_ptr bitmap, Array_ptr col) {
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
            return to_eigen<append_ones, ArrowType>(bitmap, col);
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

    template<typename ArrowType, typename MatrixObject>
    EigenMatrix<ArrowType> compute_cov(std::vector<MatrixObject>& v) {
        auto n = v.size();
        EigenMatrix<ArrowType> res = std::make_unique<typename EigenMatrix<ArrowType>::element_type>(n, n);
        
        typename ArrowType::c_type inv_n = 1 / (n - 1);
        int i = 0;
        for(auto it = v.begin(); it != v.end(); ++it, ++i) {
            (*res)(i, i) = (*it).squaredNorm() * inv_n;
            int j = 0;
            for(auto it2 = v.begin(); it2 != it; ++it2, ++j) {
                (*res)(i, j) = (*res)(j, i) = (*it).dot(*it2) * inv_n;
            }
        }

        return res;
    }

    template<typename ArrowType>
    EigenMatrix<ArrowType> cov(Buffer_ptr bitmap, Array_iterator begin, Array_iterator end) {
        using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
        std::vector<EigenVector> columns;
        auto n = std::distance(begin, end);
        columns.reserve(n);

        for(auto it = begin; it != end; ++it) {
            auto c = to_eigen<false, ArrowType>(bitmap, *it);
            auto m = c->mean();
            columns.push_back(c->array() - m);
        }

        return compute_cov<ArrowType>(columns);
    }

    template<typename ArrowType, bool contains_null>
    EigenMatrix<ArrowType> cov(Array_iterator begin, Array_iterator end) {
        if constexpr (contains_null) {
            auto bitmap = combined_bitmap(begin, end);
            return cov<ArrowType>(bitmap, begin, end);
        } else {
            using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
            std::vector<EigenVector> columns;
            auto n = std::distance(begin, end);
            columns.reserve(n);

            for(auto it = begin; it != end; ++it) {
                auto c = to_eigen<false, ArrowType, false>(*it);
                auto m = c->mean();
                columns.push_back(c->array() - m);
            }

            return compute_cov<ArrowType>(columns);
        }
    }

    template<typename ArrowType>
    EigenMatrix<ArrowType> cov(Buffer_ptr bitmap, Array_ptr col) {
        using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
        std::vector<EigenVector> columns;
        columns.reserve(1);

        auto c = to_eigen<false, ArrowType>(col, bitmap);
        auto m = c->mean();
        columns.push_back(c->array() - m);

        return compute_cov<ArrowType>(columns);
    }

    template<typename ArrowType, bool contains_null>
    EigenMatrix<ArrowType> cov(Array_ptr col) {
        if constexpr (contains_null) {
            auto bitmap = col->null_bitmap();
            return cov<ArrowType>(col, bitmap);
        } else {
            using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
            std::vector<EigenVector> columns;
            columns.reserve(1);

            auto c = to_eigen<false, ArrowType, false>(col);
            auto m = c->mean();
            columns.push_back(c->array() - m);

            return compute_cov<ArrowType>(columns);
        }
    }


    class DataFrame {
    public:

        DataFrame() : m_batch() {};

        DataFrame(std::shared_ptr<RecordBatch> rb);

        std::vector<std::string> column_names() const;

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        DataFrame loc(const T cols) const { return loc(cols.begin(), cols.end()); }
        template<typename V>
        DataFrame loc(std::initializer_list<V> cols) const { return loc(cols.begin(), cols.end()); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        DataFrame loc(const IndexIter begin, const IndexIter end) const;
        DataFrame loc(int i) const;
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        DataFrame loc(const StringType name) const;
        template<typename ...Args>
        DataFrame loc(Args... args) const;

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        arrow::Type::type same_type(const T cols) const { return same_type(cols.begin(), cols.end()); }
        template<typename V>
        arrow::Type::type same_type(std::initializer_list<V> cols) const { return same_type(cols.begin(), cols.end()); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        arrow::Type::type same_type(const IndexIter begin, const IndexIter end) const {
            auto v = indices_to_columns(begin, end);
            return same_type(v.begin(), v.end());
        }
        arrow::Type::type same_type(int i) const { return m_batch->column(i)->type_id(); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        arrow::Type::type same_type(const StringType name) const { return m_batch->GetColumnByName(name)->type_id(); }
        template<typename ...Args>
        arrow::Type::type same_type(Args... args) const {
            auto v = indices_to_columns(args...);
            return same_type(v.begin(), v.end());
        }


        Array_ptr col(int i) const { return m_batch->column(i); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Array_ptr col(const StringType name) const { return m_batch->GetColumnByName(name); }

        const std::string& name(int i) const { return m_batch->column_name(i); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        const std::string& name(const StringType n) const { return n; }

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        std::vector<std::string> names(const T n) const { return names(n.begin(), n.end()); }
        template<typename V>
        std::vector<std::string> names(std::initializer_list<V> n) const { return names(n.begin(), n.end()); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        std::vector<std::string> names(const IndexIter begin, const IndexIter end) const {
            std::vector<std::string> res;
            res.reserve(std::distance(begin, end));

            for (auto it = begin; it != end; ++it) {
                res.push_back(name(*it));
            }
            return res;
        }
        
        Buffer_ptr combined_bitmap() const { 
            Array_vector cols = m_batch->columns(); 
            return dataset::combined_bitmap(cols.begin(), cols.end());
        }
        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        Buffer_ptr combined_bitmap(const T cols) const { 
            Array_vector v = indices_to_columns(cols); 
            return dataset::combined_bitmap(v.begin(), v.end()); 
        }
        template<typename V>
        Buffer_ptr combined_bitmap(std::initializer_list<V> cols) const { 
            return combined_bitmap(cols.begin(), cols.end()); 
        }
        Buffer_ptr combined_bitmap(int i) const { 
            return m_batch->column(i)->null_bitmap(); 
        }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Buffer_ptr combined_bitmap(const StringType name) const { 
            return m_batch->GetColumnByName(name)->null_bitmap(); 
        }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        Buffer_ptr combined_bitmap(const IndexIter begin, const IndexIter end) const { 
            Array_vector v = indices_to_columns(begin, end);
            return dataset::combined_bitmap(v.begin(), v.end());
        }

        template<typename ...Args>
        Buffer_ptr combined_bitmap(Args... args) const {
            Array_vector v = indices_to_columns(args...);
            return dataset::combined_bitmap(v.begin(), v.end());
        }

        int64_t null_count() const { 
            auto cols = m_batch->columns(); 
            return dataset::null_count(cols.begin(), cols.end()); 
        }
        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        int64_t null_count(const T cols) const { 
            Array_vector v = indices_to_columns(cols); 
            return dataset::null_count(v.begin(), v.end()); 
        }
        template<typename V>
        int64_t null_count(std::initializer_list<V> cols) const { 
            return null_count(cols.begin(), cols.end()); 
        }
        int64_t null_count(int i) const { return m_batch->column(i)->null_count(); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        int64_t null_count(const StringType name) const { return m_batch->GetColumnByName(name)->null_count(); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        int64_t null_count(const IndexIter begin, const IndexIter end) const { 
            Array_vector v = indices_to_columns(begin, end); 
            return dataset::null_count(v.begin(), v.end());
        }
        template<typename ...Args>
        int64_t null_count(Args... args) const {
            Array_vector v = indices_to_columns(args...); 
            return dataset::null_count(v.begin(), v.end());
        }

        template<bool append_ones, typename ArrowType, typename T, util::enable_if_index_container_t<T, int> = 0>
        EigenMatrix<ArrowType> to_eigen(const T cols) const { 
            if (null_count(cols) == 0) {
                return to_eigen<append_ones, ArrowType, false>(cols);
            } else {
                return to_eigen<append_ones, ArrowType, true>(cols);
            }
        }
        template<bool append_ones, typename ArrowType, bool contains_null, typename T, util::enable_if_index_container_t<T, int> = 0>
        EigenMatrix<ArrowType> to_eigen(const T cols) const { 
            Array_vector v = indices_to_columns(cols); 
            return dataset::to_eigen<append_ones, ArrowType, contains_null>(v.begin(), v.end()); 
        }

        template<bool append_ones, typename ArrowType, typename T, util::enable_if_index_container_t<T, int> = 0>
        EigenMatrix<ArrowType> to_eigen(Buffer_ptr bitmap, const T cols) const { 
            Array_vector v = indices_to_columns(cols); 
            return dataset::to_eigen<append_ones, ArrowType>(bitmap, v.begin(), v.end()); 
        }
        
        template<bool append_ones, typename ArrowType, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        EigenMatrix<ArrowType> to_eigen(IndexIter begin, IndexIter end) const {
            if (null_count(begin, end) == 0) {
                return to_eigen<append_ones, ArrowType, false>(begin, end);
            } else {
                return to_eigen<append_ones, ArrowType, true>(begin, end);
            }
        }
        
        template<bool append_ones, typename ArrowType, bool contains_null, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        EigenMatrix<ArrowType> to_eigen(IndexIter begin, IndexIter end) const {
            Array_vector v = indices_to_columns(begin, end); 
            return dataset::to_eigen<append_ones, ArrowType, contains_null>(v.begin(), v.end());
        }
        
        template<bool append_ones, typename ArrowType, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        EigenMatrix<ArrowType> to_eigen(Buffer_ptr bitmap, IndexIter begin, IndexIter end) const {
            Array_vector v = indices_to_columns(begin, end); 
            return dataset::to_eigen<append_ones, ArrowType>(bitmap, v.begin(), v.end()); 
        }

        template<bool append_ones, typename ArrowType, bool contains_null>
        MapOrMatrixType<append_ones, ArrowType, contains_null> to_eigen(int i) const {
            auto col = m_batch->column(i);
            return dataset::to_eigen<append_ones, ArrowType, contains_null>(col);
        }

        template<bool append_ones, typename ArrowType>
        EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(Buffer_ptr bitmap, int i) const {
            auto col = m_batch->column(i);
            return dataset::to_eigen<append_ones, ArrowType>(bitmap, col);
        }

        template<bool append_ones, typename ArrowType, bool contains_null, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        MapOrMatrixType<append_ones, ArrowType, contains_null> to_eigen(StringType name) const {
            auto col = m_batch->GetColumnByName(name);
            return dataset::to_eigen<append_ones, ArrowType, contains_null>(col);
        }

        template<bool append_ones, typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(Buffer_ptr bitmap, StringType name) const {
            auto col = m_batch->GetColumnByName(name);
            return dataset::to_eigen<append_ones, ArrowType>(bitmap, col);
        }
        
        template<bool append_ones, typename ArrowType, typename ...Args>
        EigenMatrix<ArrowType> to_eigen(Args... args) const {
            if (null_count(args...) == 0) {
                return to_eigen<append_ones, ArrowType, false>(args...);
            } else {
                return to_eigen<append_ones, ArrowType, true>(args...);
            }
        }
        
        template<bool append_ones, typename ArrowType, bool contains_null, typename ...Args>
        EigenMatrix<ArrowType> to_eigen(Args... args) const {
            Array_vector v = indices_to_columns(args...); 
            return dataset::to_eigen<append_ones, ArrowType, contains_null>(v.begin(), v.end());
        }

        template<bool append_ones, typename ArrowType, typename ...Args>
        EigenMatrix<ArrowType> to_eigen(Buffer_ptr bitmap, Args... args) const {
            Array_vector v = indices_to_columns(args...); 
            return dataset::to_eigen<append_ones, ArrowType>(bitmap, v.begin(), v.end());
        }


        template<typename ArrowType, typename T, util::enable_if_index_container_t<T, int> = 0>
        EigenMatrix<ArrowType> cov(const T cols) {
            if (null_count(cols) == 0) {
                return cov<ArrowType, false>(cols);
            } else {
                return cov<ArrowType, true>(cols);
            }
        }
        template<typename ArrowType, bool contains_null, typename T, util::enable_if_index_container_t<T, int> = 0>
        EigenMatrix<ArrowType> cov(const T cols) {
            auto c = indices_to_columns(cols);
            return dataset::cov<ArrowType, contains_null>(c.begin(), c.end());
        }
        template<typename ArrowType, typename T, util::enable_if_index_container_t<T, int> = 0>
        EigenMatrix<ArrowType> cov(Buffer_ptr bitmap, const T cols) {
            auto c = indices_to_columns(cols);
            return dataset::cov<ArrowType>(bitmap, c.begin(), c.end());
        }
        template<typename ArrowType, typename V>
        EigenMatrix<ArrowType> cov(std::initializer_list<V> cols) {
            if (null_count(cols) == 0) {
                return cov<ArrowType, false>(cols);
            } else {
                return cov<ArrowType, true>(cols);
            }
        }
        template<typename ArrowType, bool contains_null, typename V>
        EigenMatrix<ArrowType> cov(std::initializer_list<V> cols) {
            auto c = indices_to_columns(cols);
            return dataset::cov<ArrowType, contains_null>(c.begin(), c.end());
        }
        template<typename ArrowType, typename V>
        EigenMatrix<ArrowType> cov(Buffer_ptr bitmap, std::initializer_list<V> cols) {
            auto c = indices_to_columns(cols);
            return dataset::cov<ArrowType>(bitmap, c.begin(), c.end());
        }
        template<typename ArrowType, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        EigenMatrix<ArrowType> cov(const IndexIter begin, const IndexIter end) const {
            if (null_count(begin, end) == 0) {
                return cov<ArrowType, false>(begin, end);
            } else {
                return cov<ArrowType, true>(begin, end);
            }
        }
        template<typename ArrowType, bool contains_null, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        EigenMatrix<ArrowType> cov(const IndexIter begin, const IndexIter end) const {
            auto c = indices_to_columns(begin, end);
            return dataset::cov<ArrowType, contains_null>(c.begin(), c.end());
        }
        template<typename ArrowType, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        EigenMatrix<ArrowType> cov(Buffer_ptr bitmap, const IndexIter begin, const IndexIter end) const {
            auto c = indices_to_columns(begin, end);
            return dataset::cov<ArrowType>(bitmap, c.begin(), c.end());
        }
        template<typename ArrowType>
        EigenMatrix<ArrowType> cov(int i) const {
            if (null_count(i) == 0) {
                return cov<ArrowType, false>(i);
            } else {
                return cov<ArrowType, true>(i);
            }
        }
        template<typename ArrowType, bool contains_null>
        EigenMatrix<ArrowType> cov(int i) const {
            dataset::cov<ArrowType, contains_null>(m_batch->column(i));
        }
        template<typename ArrowType>
        EigenMatrix<ArrowType> cov(Buffer_ptr bitmap, int i) const {
            dataset::cov<ArrowType>(bitmap, m_batch->column(i));
        }
        template<typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        EigenMatrix<ArrowType> cov(const StringType name) const {
            if (null_count(name) == 0) {
                return cov<ArrowType, false>(name);
            } else {
                return cov<ArrowType, true>(name);
            }
        }
        template<typename ArrowType, bool contains_null, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        EigenMatrix<ArrowType> cov(const StringType name) const {
            dataset::cov<ArrowType, contains_null>(m_batch->GetColumnByName(name));
        }
        template<typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        EigenMatrix<ArrowType> cov(Buffer_ptr bitmap, const StringType name) const {
            dataset::cov<ArrowType>(bitmap, m_batch->GetColumnByName(name));
        }
        template<typename ArrowType, typename ...Args>
        EigenMatrix<ArrowType> cov(Args... args) const {
            if (null_count(args...) == 0) {
                return cov<ArrowType, false>(args...);
            } else {
                return cov<ArrowType, true>(args...);
            }
        }
        template<typename ArrowType, bool contains_null, typename ...Args>
        EigenMatrix<ArrowType> cov(Args... args) const {
            auto c = indices_to_columns(args...);
            return dataset::cov<ArrowType, contains_null>(c.begin(), c.end());
        }
        template<typename ArrowType, typename ...Args>
        EigenMatrix<ArrowType> cov(Buffer_ptr bitmap, Args... args) const {
            auto c = indices_to_columns(args...);
            return dataset::cov<ArrowType>(bitmap, c.begin(), c.end());
        }

        std::shared_ptr<RecordBatch> operator->() const;

        friend std::pair<DataFrame, DataFrame> generate_cv_pair(const DataFrame& df, int fold, const std::vector<int>& indices, 
                                                                const std::vector<std::vector<int>::iterator>& test_limits);
    private:
        std::shared_ptr <arrow::Buffer> combined_bitmap_with_null() const;

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        Array_vector indices_to_columns(const T cols) const;

        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        Array_vector indices_to_columns(const IndexIter begin, const IndexIter end) const;

        template<typename ...Args>
        Array_vector indices_to_columns(Args... args) const;

        arrow::Type::type same_type(Array_iterator begin, Array_iterator end) const;

        std::shared_ptr<RecordBatch> m_batch;
    };

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    Array_vector DataFrame::indices_to_columns(const T cols) const {
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
    Array_vector DataFrame::indices_to_columns(const IndexIter begin, const IndexIter end) const {
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
    inline int size_argument(const T arg) { return arg.size(); }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline int size_argument(const std::pair< IndexIter, IndexIter> it) { return std::distance(it.first, it.second); }

    inline int size_argument(int) { return 1; }

    template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    inline int size_argument(const StringType) { return 1; }

    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, int i) {
        arrays.push_back(rb->column(i));
    }

    template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const StringType name) {
        arrays.push_back(rb->GetColumnByName(name));
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const IndexIter begin, const IndexIter end) {
        for (auto it = begin; it != end; ++it) {
            append_columns(rb, arrays, *it);
        }
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const std::pair<IndexIter, IndexIter> it) {
        append_columns(rb, arrays, it.first, it.second);
    }

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const T arg) { 
        return append_columns(rb, arrays, arg.begin(), arg.end()); 
    }

    inline void append_schema(const RecordBatch_ptr& rb, arrow::SchemaBuilder& b, int i) {
        b.AddField(rb->schema()->field(i));
    }

    template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    void append_schema(const RecordBatch_ptr& rb, arrow::SchemaBuilder& b, const StringType name) {
        b.AddField(rb->schema()->GetFieldByName(name));
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline void append_schema(const RecordBatch_ptr& rb, arrow::SchemaBuilder& b, IndexIter begin, IndexIter end) {
        for (auto it = begin; it != end; ++it) {
            append_schema(rb, b, *it);
        }
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline void append_schema(const RecordBatch_ptr& rb, arrow::SchemaBuilder& b, std::pair<IndexIter, IndexIter> it) {
        append_schema(rb, b, it.first, it.second);
    }

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    inline void append_schema(const RecordBatch_ptr& rb, Array_vector& arrays, const T arg) { 
        return append_schema(rb, arrays, arg.begin(), arg.end()); 
    }

    template<typename ...Args>
    Array_vector DataFrame::indices_to_columns(Args... args) const {
        Array_vector cols;

        int total_size = (size_argument(args) + ...);
        cols.reserve(total_size);

        (append_columns(m_batch, cols, args), ...);

        return cols;
    }

    template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    DataFrame DataFrame::loc(const StringType name) const {
        arrow::SchemaBuilder b;
        b.AddField(m_batch->schema()->GetFieldByName(name));
        auto r = b.Finish();
        if (!r.ok()) {
            throw std::domain_error("Schema could not be created for column " + name);
        }
        Array_vector c = { m_batch->GetColumnByName(name) };
        return RecordBatch::Make(std::move(r).ValueOrDie(), m_batch->num_rows(), c);
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    DataFrame DataFrame::loc(const IndexIter begin, const IndexIter end) const {
        arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);
        Array_vector new_cols;
        new_cols.reserve(std::distance(begin, end));

        append_schema(m_batch, b, begin, end);
        append_columns(m_batch, begin, end);

        auto r = b.Finish();
        if (!r.ok()) {
            throw std::domain_error("Schema could not be created for selected columns.");
        }
        return DataFrame(RecordBatch::Make(std::move(r).ValueOrDie(), m_batch->num_rows(), new_cols));
    }

    template<typename ...Args>
    DataFrame DataFrame::loc(Args... args) const {
        arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_ERROR);
        Array_vector new_cols;

        int total_size = (size_argument(args) + ...);
        new_cols.reserve(total_size);

        (append_schema(m_batch, b, args),...);
        (append_columns(m_batch, new_cols, args),...);

        auto r = b.Finish();
        if (!r.ok()) {
            throw std::domain_error("Schema could not be created for selected columns.");
        }
        return DataFrame(RecordBatch::Make(std::move(r).ValueOrDie(), m_batch->num_rows(), new_cols));
    }

    // template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    // MatrixXd DataFrame::cov(Array_vector begin, Array_ end) const {
    //     auto n = std::distance(begin, end);
    //     MatrixXd c(n, n);

    //     std::vector<EigenVectorOrMatrix> d_vector;
    //     d_vector.reserve(std::distance(begin, end));



    //     for (auto it = begin; it != end; ++it) {
    //         auto d1 = to_eigen(*it);
    //         auto m1 =  d1.mean();
    //         d1 = d1 - m1;
    //     }

    //     int i, j;
    //     for (auto it = begin, i = 0; it != end; ++it, ++i) {


    //         cov(i, i) = d1.squaredNorm();

    //         for (auto it2 = begin, j = 0; it2 < it; ++it2, ++j) {
    //             VectorXd d2 = to_eigen(*it);
    //             auto m1 =  d1.mean();
    //             d1 = d1 - m1;
    //         }
    //     }

    // }

    // template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    // MatrixXd DataFrame::cov(const IndexIter begin, const IndexIter end) const {
    //     auto n = std::distance(begin, end);
    //     MatrixXd c(n, n);

    //     std::vector<EigenVectorOrMatrix> d_vector;
    //     d_vector.reserve(std::distance(begin, end));



    //     for (auto it = begin; it != end; ++it) {
    //         auto d1 = to_eigen(*it);
    //         auto m1 =  d1.mean();
    //         d1 = d1 - m1;
    //     }

    //     int i, j;
    //     for (auto it = begin, i = 0; it != end; ++it, ++i) {


    //         cov(i, i) = d1.squaredNorm();

    //         for (auto it2 = begin, j = 0; it2 < it; ++it2, ++j) {
    //             VectorXd d2 = to_eigen(*it);
    //             auto m1 =  d1.mean();
    //             d1 = d1 - m1;
    //         }
    //     }

    // }
}


#endif //PGM_DATASET_DATASET_HPP
