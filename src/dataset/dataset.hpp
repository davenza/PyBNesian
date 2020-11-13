#ifndef PYBNESIAN_DATASET_HPP
#define PYBNESIAN_DATASET_HPP

#include <tuple>
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <util/parameter_traits.hpp>
#include <util/bit_util.hpp>
#include <util/arrow_macros.hpp>

namespace pyarrow = arrow::py;
namespace py = pybind11;

using Eigen::MatrixXd, Eigen::MatrixXf, Eigen::VectorXd, Eigen::VectorXf, Eigen::Matrix, Eigen::Dynamic, Eigen::Map,
        Eigen::DenseBase;
using arrow::Type, arrow::Buffer, arrow::DoubleType, arrow::FloatType, arrow::RecordBatch;


using Field_ptr = std::shared_ptr<arrow::Field>;
using Array_ptr = std::shared_ptr<arrow::Array>;
using Array_vector =  std::vector<Array_ptr>;
using Array_iterator =  Array_vector::iterator;
using Buffer_ptr = std::shared_ptr<arrow::Buffer>;
using RecordBatch_ptr = std::shared_ptr<RecordBatch>;

namespace dataset {
    typedef py::handle PyDataset;

    bool is_pandas_dataframe(py::handle pyobject);
    bool is_pandas_series(py::handle pyobject);

    std::shared_ptr<RecordBatch> to_record_batch(py::handle pyobject);
    py::object pandas_to_pyarrow_record_batch(py::handle pyobject);
    py::object pandas_to_pyarrow_array(py::handle pyobject);

    Array_ptr copy_array(const Array_ptr& array);

    Array_ptr copy_array_dictionary(const Array_ptr& array);
    Array_ptr copy_array_string(const Array_ptr& array);

    template<typename ArrowType>
    Array_ptr copy_array_numeric(const Array_ptr& array) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        arrow::NumericBuilder<ArrowType> builder;
        auto dwn_array = std::static_pointer_cast<ArrayType>(array);

        if (array->null_count() > 0) {
            RAISE_STATUS_ERROR(builder.AppendValues(dwn_array->raw_values(), array->length(), array->null_bitmap_data()));
        } else {
            RAISE_STATUS_ERROR(builder.AppendValues(dwn_array->raw_values(), array->length()));
        }

        Array_ptr out;
        RAISE_STATUS_ERROR(builder.Finish(&out));
        return out;
    }

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
    int64_t valid_rows(Array_iterator begin, Array_iterator end);

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

    template<typename ArrowType>
    typename ArrowType::c_type min(Array_ptr a) {
        using CType = typename ArrowType::c_type;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using MapType = Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>;
        auto dwn = std::static_pointer_cast<ArrayType>(a);

        auto raw = dwn->raw_values();
        if (a->null_count() == 0) {
            MapType map(raw, a->length());
            return map.minCoeff();
        } else {
            auto bitmap = a->null_bitmap_data();
            auto res = std::numeric_limits<CType>::infinity();
            for (auto i = 0; i < a->length(); ++i) {
                if (arrow::BitUtil::GetBit(bitmap, i) && raw[i] < res)
                    res = raw[i];
            }
            return res;
        }
    }

    template<typename ArrowType>
    typename ArrowType::c_type max(Array_ptr a) {
        using CType = typename ArrowType::c_type;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using MapType = Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>;
        auto dwn = std::static_pointer_cast<ArrayType>(a);

        auto raw = dwn->raw_values();
        if (a->null_count() == 0) {
            MapType map(raw, a->length());
            return map.maxCoeff();
        } else {
            auto bitmap = a->null_bitmap_data();
            auto res = -std::numeric_limits<CType>::infinity();
            for (auto i = 0; i < a->length(); ++i) {
                if (arrow::BitUtil::GetBit(bitmap, i) && raw[i] > res)
                    res = raw[i];
            }
            return res;
        }

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

        return m;
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
            return m;
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
        using CType = typename ArrowType::c_type;
        auto N = v[0].rows();
        auto n = v.size();
        EigenMatrix<ArrowType> res = std::make_unique<typename EigenMatrix<ArrowType>::element_type>(n, n);

        CType inv_N = 1 / static_cast<CType>(N - 1);

        for (size_t i = 0; i < v.size(); ++i) {
            (*res)(i, i) = v[i].squaredNorm() * inv_N;

            for (size_t j = i+1; j < v.size(); ++j) {
                (*res)(i, j) = (*res)(j, i) = v[i].dot(v[j]) * inv_N;
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
            return cov<ArrowType>(bitmap, col);
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


    template<bool copy, typename... Args>
    class IndexLOC {
    public:
        IndexLOC(Args... args) : m_args(args...) {}

        const std::tuple<Args...>& columns() const { return m_args; }
    private:
        const std::tuple<Args...> m_args;
    };

    template<typename... Args>
    using CopyLOC = IndexLOC<true, Args...>;
    template<typename... Args>
    using MoveLOC = IndexLOC<false, Args...>;

    // This is necessary because C++17 do not allow alias template type deduction.
    template<typename... Args>
    CopyLOC<Args...> Copy(Args... args) { 
        return CopyLOC<Args...>(args...);
    }
    template<typename... Args>
    MoveLOC<Args...> Move(Args... args) { 
        return MoveLOC<Args...>(args...);
    }

    class DataFrame {
    public:
        DataFrame() = default;

        DataFrame(int64_t num_rows) : m_batch(arrow::RecordBatch::Make(
                                                arrow::schema({}),
                                                num_rows,
                                                Array_vector()
                                                )) {}

        DataFrame(std::shared_ptr<RecordBatch> rb);

        const std::shared_ptr<RecordBatch>& record_batch() const { return m_batch; }
        std::vector<std::string> column_names() const;

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        void has_columns(const T& cols) const { has_columns(cols.begin(), cols.end()); }
        template<typename V>
        void has_columns(const std::initializer_list<V>& cols) const { has_columns(cols.begin(), cols.end()); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        void has_columns(const IndexIter& begin, const IndexIter& end) const;
        void has_columns(int i) const;
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        void has_columns(const StringType& name) const;
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        void has_columns(const std::pair<IndexIter, IndexIter>& it);
        template<typename ...Args>
        void has_columns(const Args&... args) const;

        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        DataFrame loc(const T& cols) const { return loc(cols.begin(), cols.end()); }
        template<typename V>
        DataFrame loc(const std::initializer_list<V>& cols) const { return loc(cols.begin(), cols.end()); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        DataFrame loc(const IndexIter& begin, const IndexIter& end) const;
        DataFrame loc(int i) const;
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        DataFrame loc(const StringType& name) const;
        template<typename ...Args>
        DataFrame loc(const Args&... args) const;

        arrow::Type::type same_type() const {
            Array_vector cols = m_batch->columns();
            return same_type(cols.begin(), cols.end());
        }
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
        arrow::Type::type same_type(const StringType& name) const { return m_batch->GetColumnByName(name)->type_id(); }
        template<typename ...Args>
        arrow::Type::type same_type(Args... args) const {
            auto v = indices_to_columns(args...);
            return same_type(v.begin(), v.end());
        }

        Array_ptr col(int i) const { 
            if (i < m_batch->num_columns())
                return m_batch->column(i);
            else
                throw std::invalid_argument("Column index " + std::to_string(i) + " do not exist in DataFrame.");
        }

        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Array_ptr col(const StringType& name) const { 
            return m_batch->GetColumnByName(name); 
        }

        const std::string& name(int i) const { return m_batch->column_name(i); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        const std::string& name(const StringType& n) const { return n; }

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

        template<typename ArrowType>
        const typename ArrowType::c_type* data(int i) const {
            return m_batch->column_data(i)->template GetValues<typename ArrowType::c_type>(1);
        }

        template<typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        const typename ArrowType::c_type* data(const StringType& name) const {
            return m_batch->GetColumnByName(name)->data()->template GetValues<typename ArrowType::c_type>(1);
        }

        template<typename ArrowType>
        typename ArrowType::c_type* mutable_data(int i) const {
            return m_batch->column_data(i)->template GetMutableValues<typename ArrowType::c_type>(1);
        }

        template<typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        typename ArrowType::c_type* mutable_data(const StringType& name) const {
            return m_batch->GetColumnByName(name)->data()->template GetMutableValues<typename ArrowType::c_type>(1);
        }

        template<typename ArrowType>
        std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>
        downcast(int i) const {
            using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
            auto a = m_batch->column(i);
            return std::static_pointer_cast<ArrayType>(a);
        }
        template<typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>
        downcast(const StringType& name) const {
            using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
            auto a = m_batch->GetColumnByName(name);
            return std::static_pointer_cast<ArrayType>(a);
        }

        template<typename ArrowType>
        std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>>
        downcast_vector() const { 
            using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
            Array_vector v = indices_to_columns();
            std::vector<std::shared_ptr<ArrayType>> res;
            res.reserve(v.size());
            for (auto& array : v) {
                res.push_back(std::static_pointer_cast<ArrayType>(array));
            }

            return res;
        }
        template<typename ArrowType, typename T, util::enable_if_index_container_t<T, int> = 0>
        std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>>
        downcast_vector(const T n) const { 
            return downcast_vector<ArrowType>(n.begin(), n.end());
        }
        template<typename ArrowType, typename V>
        std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> 
        downcast_vector(std::initializer_list<V> n) const { 
            return downcast_vector<ArrowType>(n.begin(), n.end());
        }

        template<typename ArrowType, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> 
        downcast_vector(const IndexIter begin, const IndexIter end) const {
            using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
            Array_vector v = indices_to_columns(begin, end);
            std::vector<std::shared_ptr<ArrayType>> res;
            res.reserve(v.size());
            for (auto& array : v) {
                res.push_back(std::static_pointer_cast<ArrayType>(array));
            }

            return res;
        }

        template<typename ArrowType, typename ...Args>
        std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> 
        downcast_vector(Args... args) const {
            using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
            Array_vector v = indices_to_columns(args...);
            std::vector<std::shared_ptr<ArrayType>> res;
            res.reserve(v.size());
            for (auto& array : v) {
                res.push_back(std::static_pointer_cast<ArrayType>(array));
            }

            return res;
        }
        
        template<typename ArrowType>
        std::unordered_map<std::string, std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> 
        downcast_map(int i) const {
            using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
            auto a = m_batch->column(i);
            return { { name(i), std::static_pointer_cast<ArrayType>(a) } };
        }
        template<typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        std::unordered_map<std::string, std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> 
        downcast_map(const StringType& name) const {
            using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
            auto a = m_batch->GetColumnByName(name);
            return { {name, std::static_pointer_cast<ArrayType>(a) } };
        }
        template<typename ArrowType, typename T, util::enable_if_index_container_t<T, int> = 0>
        std::unordered_map<std::string, std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> 
        downcast_map(const T n) const { 
            return downcast_map<ArrowType>(n.begin(), n.end());
        }
        template<typename ArrowType, typename V>
        std::unordered_map<std::string, std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> 
        downcast_map(std::initializer_list<V> n) const { 
            return downcast_map<ArrowType>(n.begin(), n.end());
        }
        template<typename ArrowType, typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        std::unordered_map<std::string, std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> 
        downcast_map(const IndexIter begin, const IndexIter end) const {
            using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
            std::unordered_map<std::string, std::shared_ptr<ArrayType>> res;
            
            for (auto it = begin; it != end; ++it) {
                res.insert({name(*it), std::static_pointer_cast<ArrayType>(col(*it))});
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
        Buffer_ptr combined_bitmap(const StringType& name) const { 
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
        int64_t null_count(const StringType& name) const { return m_batch->GetColumnByName(name)->null_count(); }
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

        int64_t valid_rows() const {
            auto cols = m_batch->columns();
            return dataset::valid_rows(cols.begin(), cols.end()); 
        }
        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        int64_t valid_rows(const T cols) const { 
            return valid_rows(cols.begin(), cols.end()); 
        }
        template<typename V>
        int64_t valid_rows(std::initializer_list<V> cols) const { 
            return valid_rows(cols.begin(), cols.end()); 
        }
        int64_t valid_rows(int i) const { return m_batch->num_rows() - m_batch->column(i)->null_count(); }
        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        int64_t valid_rows(const StringType& name) const { return m_batch->num_rows() - m_batch->GetColumnByName(name)->null_count(); }
        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        int64_t valid_rows(const IndexIter begin, const IndexIter end) const {
            auto v = indices_to_columns(begin, end);
            return dataset::valid_rows(v.begin(), v.end());
        }
        template<typename ...Args>
        int64_t valid_rows(Args... args) const {
            auto v = indices_to_columns(args...);
            return dataset::valid_rows(v.begin(), v.end());
        }

        template<typename ArrowType>
        typename ArrowType::c_type min(int i) const {
            return dataset::min<ArrowType>(m_batch->column(i));
        }

        template<typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        typename ArrowType::c_type min(const StringType& name) const {
            return dataset::min<ArrowType>(m_batch->GetColumnByName(name));
        }

        template<typename ArrowType>
        typename ArrowType::c_type max(int i) const {
            return dataset::max<ArrowType>(m_batch->column(i));
        }

        template<typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        typename ArrowType::c_type max(const StringType& name) const {
            return dataset::max<ArrowType>(m_batch->GetColumnByName(name));
        }


        template<bool append_ones, typename ArrowType>
        EigenMatrix<ArrowType> to_eigen() const {
            auto cols = m_batch->columns();
            if (null_count() == 0) {
                return dataset::to_eigen<append_ones, ArrowType, false>(cols.begin(), cols.end());
            } else {
                return dataset::to_eigen<append_ones, ArrowType, true>(cols.begin(), cols.end());
            }
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
        MapOrMatrixType<append_ones, ArrowType, contains_null> to_eigen(const StringType& name) const {
            auto col = m_batch->GetColumnByName(name);
            return dataset::to_eigen<append_ones, ArrowType, contains_null>(col);
        }

        template<bool append_ones, typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(Buffer_ptr bitmap, const StringType& name) const {
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
                return cov<ArrowType, false, std::initializer_list<V>>(cols);
            } else {
                return cov<ArrowType, true, std::initializer_list<V>>(cols);
            }
        }
        template<typename ArrowType, bool contains_null, typename V>
        EigenMatrix<ArrowType> cov(std::initializer_list<V> cols) {
            auto c = indices_to_columns(cols);
            return dataset::cov<ArrowType, contains_null, std::initializer_list<V>>(c.begin(), c.end());
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
        EigenMatrix<ArrowType> cov(const StringType& name) const {
            if (null_count(name) == 0) {
                return cov<ArrowType, false>(name);
            } else {
                return cov<ArrowType, true>(name);
            }
        }
        template<typename ArrowType, bool contains_null, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        EigenMatrix<ArrowType> cov(const StringType& name) const {
            dataset::cov<ArrowType, contains_null>(m_batch->GetColumnByName(name));
        }
        template<typename ArrowType, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        EigenMatrix<ArrowType> cov(Buffer_ptr bitmap, const StringType& name) const {
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

        std::vector<int> continuous_columns() const;

        std::shared_ptr<RecordBatch> operator->() const;
        friend std::pair<DataFrame, DataFrame> generate_cv_pair(const DataFrame& df, int fold, const std::vector<int>& indices, 
                                                                const std::vector<std::vector<int>::iterator>& test_limits);

    private:

        Array_vector indices_to_columns() const {
            return m_batch->columns();
        }
        template<typename T, util::enable_if_index_container_t<T, int> = 0>
        Array_vector indices_to_columns(const T& cols) const {
            return indices_to_columns(cols.begin(), cols.end());
        }

        template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
        Array_vector indices_to_columns(const IndexIter& begin, const IndexIter& end) const;

        template<typename ...Args>
        Array_vector indices_to_columns(const Args&... args) const;

        arrow::Type::type same_type(Array_iterator begin, Array_iterator end) const;

        std::shared_ptr<RecordBatch> m_batch;
    };

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    inline int size_argument(const RecordBatch_ptr&, const T& arg) { return arg.size(); }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline int size_argument(const RecordBatch_ptr&, const std::pair<IndexIter, IndexIter>& it) { 
        return std::distance(it.first, it.second); 
    }

    inline int size_argument(const RecordBatch_ptr&, int) { return 1; }

    template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    inline int size_argument(const RecordBatch_ptr&, const StringType&) { return 1; }

    template<bool copy, typename... Args>
    inline int size_argument(const RecordBatch_ptr& rb, const IndexLOC<copy, Args...>& cols) {
        if constexpr(std::tuple_size_v<std::remove_reference_t<decltype(cols.columns())>> == 0) {
            return rb->num_columns();
        } else {
            return std::apply([&rb](const auto&... args) {
                return (size_argument(rb, args) + ...);
            }, cols.columns());
        }
    }

    inline void append_copy_columns(const RecordBatch_ptr& rb, Array_vector& arrays, int i) {
        if (i < rb->num_columns())
            arrays.push_back(copy_array(rb->column(i)));
        else
            throw std::invalid_argument("Column index " + std::to_string(i) + " do not exist in DataFrame.");
    }

    template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    inline void append_copy_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const StringType& name) {
        auto c = rb->GetColumnByName(name);
        if (c)
            arrays.push_back(copy_array(c));
        else
            throw std::invalid_argument("Column \"" + name + "\" do not exist in DataFrame.");
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline void append_copy_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const IndexIter& begin, const IndexIter& end) {
        for (auto it = begin; it != end; ++it) {
            append_copy_columns(rb, arrays, *it);
        }
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline void append_copy_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const std::pair<IndexIter, IndexIter>& it) {
        return append_copy_columns(rb, arrays, it.first, it.second);
    }

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    inline void append_copy_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const T& arg) { 
        return append_copy_columns(rb, arrays, arg.begin(), arg.end()); 
    }

    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, int i) {
        if (i < rb->num_columns())
            arrays.push_back(rb->column(i));
        else
            throw std::invalid_argument("Column index " + std::to_string(i) + " do not exist in DataFrame.");
    }

    template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const StringType& name) {
        auto c = rb->GetColumnByName(name);
        if (c)
            arrays.push_back(c);
        else
            throw std::invalid_argument("Column \"" + name + "\" do not exist in DataFrame.");
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const IndexIter& begin, const IndexIter& end) {
        for (auto it = begin; it != end; ++it) {
            append_columns(rb, arrays, *it);
        }
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const std::pair<IndexIter, IndexIter>& it) {
        append_columns(rb, arrays, it.first, it.second);
    }

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const T& arg) { 
        append_columns(rb, arrays, arg.begin(), arg.end()); 
    }

    template<typename ...Args>
    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const CopyLOC<Args...>& cols) {
        if constexpr(std::tuple_size_v<std::remove_reference_t<decltype(cols.columns())>> == 0) {
            for (int i = 0; i < rb->num_columns(); ++i) {
                append_copy_columns(rb, arrays, i);
            }
        } else {
            std::apply([&rb, &arrays](const auto&...args) {
                (append_copy_columns(rb, arrays, args),...);
            }, cols.columns());
        }
    }

    template<typename ...Args>
    inline void append_columns(const RecordBatch_ptr& rb, Array_vector& arrays, const MoveLOC<Args...>& cols) {
        if constexpr(std::tuple_size_v<std::remove_reference_t<decltype(cols.columns())>> == 0) {
            for (int i = 0; i < rb->num_columns(); ++i) {
                append_columns(rb, arrays, i);
            }
        } else {
            std::apply([&rb, &arrays](const auto&...args) {
                (append_columns(rb, arrays, args),...);
            }, cols.columns());
        }
    }

    inline void append_schema(const RecordBatch_ptr& rb, arrow::SchemaBuilder& b, int i) {
        RAISE_STATUS_ERROR(b.AddField(rb->schema()->field(i)));
    }

    template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    void append_schema(const RecordBatch_ptr& rb, arrow::SchemaBuilder& b, const StringType& name) {
        RAISE_STATUS_ERROR(b.AddField(rb->schema()->GetFieldByName(name)));
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline void append_schema(const RecordBatch_ptr& rb, arrow::SchemaBuilder& b, const IndexIter& begin, const IndexIter& end) {
        for (auto it = begin; it != end; ++it) {
            append_schema(rb, b, *it);
        }
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    inline void append_schema(const RecordBatch_ptr& rb, arrow::SchemaBuilder& b, const std::pair<IndexIter, IndexIter>& it) {
        append_schema(rb, b, it.first, it.second);
    }

    template<typename T, util::enable_if_index_container_t<T, int> = 0>
    inline void append_schema(const RecordBatch_ptr& rb, arrow::SchemaBuilder& b, const T& arg) { 
        append_schema(rb, b, arg.begin(), arg.end()); 
    }

    template<bool copy, typename ...Args>
    inline void append_schema(const RecordBatch_ptr& rb, arrow::SchemaBuilder& b, const IndexLOC<copy, Args...>& cols) {
        if constexpr(std::tuple_size_v<std::remove_reference_t<decltype(cols.columns())>> == 0) {
            for (int i = 0; i < rb->num_columns(); ++i) {
                append_schema(rb, b, i);
            }
        } else {
            std::apply([&rb, &b](const auto&...args) {
                (append_schema(rb, b, args),...);
            }, cols.columns());
        }
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int>>
    Array_vector DataFrame::indices_to_columns(const IndexIter& begin, const IndexIter& end) const {
        Array_vector v;
        v.reserve(std::distance(begin, end));
        append_columns(m_batch, v, begin, end);
        return v;
    }

    template<typename ...Args>
    Array_vector DataFrame::indices_to_columns(const Args&... args) const {
        Array_vector cols;

        int total_size = (size_argument(m_batch, args) + ...);
        cols.reserve(total_size);

        (append_columns(m_batch, cols, args), ...);

        return cols;
    }

    template<typename StringType, util::enable_if_stringable_t<StringType, int>>
    void DataFrame::has_columns(const StringType& name) const {
        auto c = m_batch->GetColumnByName(name);
        if (!c) {
            throw std::domain_error("Column \"" + name + "\" not found in DataFrame");
        }
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int>>
    void DataFrame::has_columns(const IndexIter& begin, const IndexIter& end) const {
        for(auto it = begin; it != end; ++it) {
            has_columns(*it);
        }
    }

    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int>>
    void DataFrame::has_columns(const std::pair<IndexIter, IndexIter>& it) {
        has_columns(it.first, it.second);
    }

    template<typename ...Args>
    void DataFrame::has_columns(const Args&... args) const {
        (has_columns(args),...);
    }

    template<typename StringType, util::enable_if_stringable_t<StringType, int>>
    DataFrame DataFrame::loc(const StringType& name) const {
        arrow::SchemaBuilder b;
        auto f = m_batch->schema()->GetFieldByName(name);
        if (!f) {
            throw std::invalid_argument("Column \"" + name + "\" do not exist in DataFrame.");
        }

        RAISE_STATUS_ERROR(b.AddField(f));

        auto r = b.Finish();
        if (!r.ok()) {
            throw std::domain_error("Schema could not be created for column " + name);
        }
        Array_vector c = { m_batch->GetColumnByName(name) };
        return RecordBatch::Make(std::move(r).ValueOrDie(), m_batch->num_rows(), c);
    }


    template<typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int>>
    DataFrame DataFrame::loc(const IndexIter& begin, const IndexIter& end) const {
        arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_APPEND);
        Array_vector new_cols;
        new_cols.reserve(std::distance(begin, end));

        append_columns(m_batch, new_cols, begin, end);
        append_schema(m_batch, b, begin, end);

        auto r = b.Finish();
        if (!r.ok()) {
            throw std::domain_error("Schema could not be created for selected columns.");
        }
        return DataFrame(RecordBatch::Make(std::move(r).ValueOrDie(), m_batch->num_rows(), new_cols));
    }

    template<typename ...Args>
    DataFrame DataFrame::loc(const Args&... args) const {
        arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_APPEND);
        Array_vector new_cols;

        int total_size = (size_argument(m_batch, args) + ...);
        new_cols.reserve(total_size);

        (append_columns(m_batch, new_cols, args),...);
        (append_schema(m_batch, b, args),...);

        auto r = b.Finish();
        if (!r.ok()) {
            throw std::domain_error("Schema could not be created for selected columns.");
        }
        return DataFrame(RecordBatch::Make(std::move(r).ValueOrDie(), m_batch->num_rows(), new_cols));
    }
}

namespace pybind11::detail {
    template <> struct type_caster<dataset::DataFrame> {
    public:
        /**
         * This macro establishes the name 'inty' in
         * function signatures and declares a local variable
         * 'value' of type inty
         */
        PYBIND11_TYPE_CASTER(dataset::DataFrame, _("DataFrame"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a inty
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
            PyObject* py_ptr = src.ptr();

            if (pyarrow::is_batch(py_ptr)) {
                auto result = pyarrow::unwrap_batch(py_ptr);
                if (result.ok()) {
                    value = result.ValueOrDie();
                    return true;
                } else {
                    return false;
                }
            }
            else if (dataset::is_pandas_dataframe(src)) {
                auto a = dataset::pandas_to_pyarrow_record_batch(src);
                auto result = pyarrow::unwrap_batch(a.ptr());

                if (result.ok()) {
                    value = result.ValueOrDie();
                    return true;
                } else {
                    return false;
                }
            }

            return false;
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an inty instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(dataset::DataFrame src, return_value_policy /* policy */, handle /* parent */) {
            PyObject* wrapped_rb = pyarrow::wrap_batch(src.record_batch());
            return wrapped_rb;
        }
    };
} // namespace pybind11::detail

namespace pybind11::detail {
    template <> struct type_caster<Array_ptr> {
    public:
        /**
         * This macro establishes the name 'inty' in
         * function signatures and declares a local variable
         * 'value' of type inty
         */
        PYBIND11_TYPE_CASTER(Array_ptr, _("pyarrow.Array"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a inty
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
            PyObject* py_ptr = src.ptr();

            if (pyarrow::is_array(py_ptr)) {
                auto result = pyarrow::unwrap_array(py_ptr);
                if (result.ok()) {
                    value = result.ValueOrDie();
                    return true;
                } else {
                    return false;
                }
            }
            else if (dataset::is_pandas_series(src)) {
                auto a = dataset::pandas_to_pyarrow_array(src);
                auto result = pyarrow::unwrap_array(a.ptr());

                if (result.ok()) {
                    value = result.ValueOrDie();
                    return true;
                } else {
                    return false;
                }
            }

            return false;
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an inty instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(Array_ptr src, return_value_policy /* policy */, handle /* parent */) {
            PyObject* wrapped_rb = pyarrow::wrap_array(src);
            return wrapped_rb;
        }
    };
} // namespace pybind11::detail

#endif //PYBNESIAN_DATASET_HPP