#ifndef PYBNESIAN_DATASET_DATASET_HPP
#define PYBNESIAN_DATASET_DATASET_HPP

#include <Eigen/Dense>
#include <arrow/c/bridge.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <pybind11/pybind11.h>
#include <util/parameter_traits.hpp>
#include <util/bit_util.hpp>
#include <util/arrow_macros.hpp>

namespace py = pybind11;

using arrow::Type, arrow::Buffer, arrow::DoubleType, arrow::FloatType, arrow::RecordBatch;
using Eigen::MatrixXd, Eigen::MatrixXf, Eigen::VectorXd, Eigen::VectorXf, Eigen::Matrix, Eigen::Dynamic, Eigen::Map,
    Eigen::DenseBase;

using Field_ptr = std::shared_ptr<arrow::Field>;
using Array_ptr = std::shared_ptr<arrow::Array>;
using Array_vector = std::vector<Array_ptr>;
using Array_iterator = Array_vector::iterator;
using Buffer_ptr = std::shared_ptr<arrow::Buffer>;
using RecordBatch_ptr = std::shared_ptr<RecordBatch>;

namespace dataset {

struct ArrowCAPIObjects {
    struct ArrowSchema* arrow_schema;
    struct ArrowArray* arrow_array;
};

bool is_pyarrow_instance(py::handle pyobject, const char* class_name);
bool is_pandas_dataframe(py::handle pyobject);
bool is_pandas_series(py::handle pyobject);

py::object pandas_to_pyarrow_record_batch(py::handle pyobject);
py::object pandas_to_pyarrow_array(py::handle pyobject);

struct ArrowSchema* extract_pycapsule_schema(py::handle pyobject);
ArrowCAPIObjects extract_pycapsule_array(py::handle pyobject);
void ReleaseArrowSchemaPyCapsule(PyObject* capsule);
void ReleaseArrowArrayPyCapsule(PyObject* capsule);

Array_ptr copy_array(const Array_ptr& array);
Array_ptr copy_array_dictionary(const Array_ptr& array);
Array_ptr copy_array_string(const Array_ptr& array);

template <typename ArrowType>
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

template <typename ArrowType>
using EigenMatrix = std::unique_ptr<Matrix<typename ArrowType::c_type, Dynamic, Dynamic>>;

template <bool append_ones, typename ArrowType>
using EigenVectorOrMatrix = std::unique_ptr<Matrix<typename ArrowType::c_type, Dynamic, append_ones + 1>>;
template <typename ArrowType>
using MapType = std::unique_ptr<Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>>;
template <bool append_ones, typename ArrowType, bool contains_null>
using MapOrMatrixType = typename std::
    conditional_t<append_ones || contains_null, EigenVectorOrMatrix<append_ones, ArrowType>, MapType<ArrowType>>;

int64_t null_count(Array_iterator begin, Array_iterator end);
Buffer_ptr combined_bitmap(Array_iterator begin, Array_iterator end);
int64_t valid_rows(Array_iterator begin, Array_iterator end);

template <bool append_ones, typename ArrowType>
inline typename ArrowType::c_type* fill_ones(typename ArrowType::c_type* ptr, int rows [[maybe_unused]]) {
    if constexpr (append_ones) {
        std::fill_n(ptr, rows, 1);
        return ptr + rows;
    } else {
        return ptr;
    }
}

template <typename ArrowType>
typename ArrowType::c_type* fill_data_bitmap(typename ArrowType::c_type* ptr,
                                             Array_ptr input_array,
                                             const uint8_t* bitmap_data,
                                             int total_rows) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto dwn_col = std::static_pointer_cast<ArrayType>(input_array);
    auto raw_values = dwn_col->raw_values();

    auto k = 0;
    for (auto j = 0; j < total_rows; ++j) {
        if (util::bit_util::GetBit(bitmap_data, j)) ptr[k++] = raw_values[j];
    }
    return ptr + k;
}

std::shared_ptr<arrow::DataType> same_type(Array_iterator begin, Array_iterator end);

template <typename ArrowType>
typename ArrowType::c_type min(Array_ptr& a) {
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
            if (util::bit_util::GetBit(bitmap, i) && raw[i] < res) res = raw[i];
        }
        return res;
    }
}

template <typename ArrowType>
typename ArrowType::c_type min(Array_ptr&& a) {
    return min<ArrowType>(a);
}

template <typename ArrowType>
typename ArrowType::c_type max(Array_ptr& a) {
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
            if (util::bit_util::GetBit(bitmap, i) && raw[i] > res) res = raw[i];
        }
        return res;
    }
}

template <typename ArrowType>
typename ArrowType::c_type max(Array_ptr&& a) {
    return max<ArrowType>(a);
}

// //////////////////////////////////// mean() //////////////////////////////
double mean(Array_ptr& a);
double mean(const Buffer_ptr& bitmap, Array_ptr& a);
double mean(Array_ptr&& a);
double mean(const Buffer_ptr&& bitmap, Array_ptr&& a);
VectorXd means(Array_iterator begin, Array_iterator end);
VectorXd means(const Buffer_ptr& bitmap, Array_iterator begin, Array_iterator end);

template <typename ArrowType>
typename ArrowType::c_type mean(const Buffer_ptr& bitmap, Array_ptr& a) {
    using CType = typename ArrowType::c_type;
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto dwn = std::static_pointer_cast<ArrayType>(a);

    auto raw = dwn->raw_values();
    auto bitmap_data = bitmap->data();
    CType res = 0;
    for (auto i = 0; i < a->length(); ++i) {
        if (util::bit_util::GetBit(bitmap_data, i)) res += raw[i];
    }

    return res / static_cast<CType>(util::bit_util::non_null_count(bitmap, a->length()));
}

template <typename ArrowType>
typename ArrowType::c_type mean(Array_ptr& a) {
    using CType = typename ArrowType::c_type;
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using MapType = Map<const Matrix<CType, Dynamic, 1>>;
    auto dwn = std::static_pointer_cast<ArrayType>(a);

    auto raw = dwn->raw_values();
    if (a->null_count() == 0) {
        MapType map(raw, a->length());
        return map.mean();
    } else {
        auto bitmap = a->null_bitmap();
        return mean<ArrowType>(bitmap, a);
    }
}

template <typename ArrowType>
typename ArrowType::c_type mean(Array_ptr&& a) {
    return mean<ArrowType>(a);
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> means(Buffer_ptr bitmap, Array_iterator begin, Array_iterator end) {
    using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;

    EigenVector res(std::distance(begin, end));

    int i = 0;
    for (auto it = begin; it != end; ++it, ++i) {
        res(i) = mean<ArrowType>(bitmap, *it);
    }

    return res;
}

template <typename ArrowType>
Matrix<typename ArrowType::c_type, Dynamic, 1> means(Array_iterator begin, Array_iterator end) {
    using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;

    EigenVector res(std::distance(begin, end));

    int i = 0;
    for (auto it = begin; it != end; ++it, ++i) {
        res(i) = mean<ArrowType>(*it);
    }

    return res;
}

// //////////////////////////////////// to_eigen() //////////////////////////////
template <bool append_ones, typename ArrowType>
EigenMatrix<ArrowType> to_eigen(Buffer_ptr bitmap, Array_iterator begin, Array_iterator end) {
    auto ncols = std::distance(begin, end);

    if (ncols == 0) {
        return nullptr;
    }

    auto rows = (*begin)->length();
    auto valid_rows = util::bit_util::non_null_count(bitmap, rows);

    auto m = [valid_rows, ncols]() {
        if constexpr (append_ones)
            return std::move(std::make_unique<typename EigenMatrix<ArrowType>::element_type>(valid_rows, ncols + 1));
        else
            return std::move(std::make_unique<typename EigenMatrix<ArrowType>::element_type>(valid_rows, ncols));
    }();

    auto m_ptr = m->data();
    m_ptr = fill_ones<append_ones, ArrowType>(m_ptr, valid_rows);
    const uint8_t* bitmap_data = bitmap->data();

    for (auto it = begin; it != end; ++it) {
        m_ptr = fill_data_bitmap<ArrowType>(m_ptr, *it, bitmap_data, rows);
    }

    return m;
}

template <bool append_ones, typename ArrowType, bool contains_null>
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
            if constexpr (append_ones)
                return std::make_unique<typename EigenMatrix<ArrowType>::element_type>(rows, ncols + 1);
            else
                return std::make_unique<typename EigenMatrix<ArrowType>::element_type>(rows, ncols);
        }();

        auto m_ptr = m->data();
        m_ptr = fill_ones<append_ones, ArrowType>(m_ptr, rows);

        for (auto it = begin; it != end; ++it) {
            auto dwn_col = std::static_pointer_cast<ArrayType>(*it);
            std::memcpy(m_ptr, dwn_col->raw_values(), sizeof(typename ArrowType::c_type) * rows);
            m_ptr += rows;
        }
        return m;
    }
}

template <bool append_ones, typename ArrowType>
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

template <bool append_ones, typename ArrowType, bool contains_null>
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

// //////////////////////////////////// cov() //////////////////////////////
template <typename ArrowType, typename MatrixObject>
EigenMatrix<ArrowType> compute_cov(std::vector<MatrixObject>& v) {
    using CType = typename ArrowType::c_type;
    auto N = v[0].rows();
    auto n = v.size();
    EigenMatrix<ArrowType> res = std::make_unique<typename EigenMatrix<ArrowType>::element_type>(n, n);

    CType inv_N = 1 / static_cast<CType>(N - 1);

    for (size_t i = 0; i < v.size(); ++i) {
        (*res)(i, i) = v[i].squaredNorm() * inv_N;

        for (size_t j = i + 1; j < v.size(); ++j) {
            (*res)(i, j) = (*res)(j, i) = v[i].dot(v[j]) * inv_N;
        }
    }

    return res;
}

template <typename ArrowType>
EigenMatrix<ArrowType> cov(Buffer_ptr bitmap, Array_iterator begin, Array_iterator end) {
    using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
    std::vector<EigenVector> columns;
    auto n = std::distance(begin, end);
    columns.reserve(n);

    for (auto it = begin; it != end; ++it) {
        auto c = to_eigen<false, ArrowType>(bitmap, *it);
        auto m = c->mean();
        columns.push_back(c->array() - m);
    }

    return compute_cov<ArrowType>(columns);
}

template <typename ArrowType, bool contains_null>
EigenMatrix<ArrowType> cov(Array_iterator begin, Array_iterator end) {
    if constexpr (contains_null) {
        auto bitmap = combined_bitmap(begin, end);
        return cov<ArrowType>(bitmap, begin, end);
    } else {
        using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
        std::vector<EigenVector> columns;
        auto n = std::distance(begin, end);
        columns.reserve(n);

        for (auto it = begin; it != end; ++it) {
            auto c = to_eigen<false, ArrowType, false>(*it);
            auto m = c->mean();
            columns.push_back(c->array() - m);
        }

        return compute_cov<ArrowType>(columns);
    }
}

template <typename ArrowType>
typename ArrowType::c_type cov(Buffer_ptr bitmap, Array_ptr col) {
    using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
    std::vector<EigenVector> columns;
    columns.reserve(1);

    auto c = to_eigen<false, ArrowType>(bitmap, col);
    auto m = c->mean();
    columns.push_back(c->array() - m);

    return *compute_cov<ArrowType>(columns)->data();
}

template <typename ArrowType, bool contains_null>
typename ArrowType::c_type cov(Array_ptr col) {
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

        return *compute_cov<ArrowType>(columns)->data();
    }
}

// //////////////////////////////////// sse() //////////////////////////////
template <typename ArrowType, typename MatrixObject>
EigenMatrix<ArrowType> compute_sse(std::vector<MatrixObject>& v) {
    auto n = v.size();
    EigenMatrix<ArrowType> res = std::make_unique<typename EigenMatrix<ArrowType>::element_type>(n, n);

    for (size_t i = 0; i < v.size(); ++i) {
        (*res)(i, i) = v[i].squaredNorm();

        for (size_t j = i + 1; j < v.size(); ++j) {
            (*res)(i, j) = (*res)(j, i) = v[i].dot(v[j]);
        }
    }

    return res;
}

template <typename ArrowType>
EigenMatrix<ArrowType> sse(Buffer_ptr bitmap, Array_iterator begin, Array_iterator end) {
    using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
    std::vector<EigenVector> columns;
    auto n = std::distance(begin, end);
    columns.reserve(n);

    for (auto it = begin; it != end; ++it) {
        auto c = to_eigen<false, ArrowType>(bitmap, *it);
        auto m = c->mean();
        columns.push_back(c->array() - m);
    }

    return compute_sse<ArrowType>(columns);
}

template <typename ArrowType, bool contains_null>
EigenMatrix<ArrowType> sse(Array_iterator begin, Array_iterator end) {
    if constexpr (contains_null) {
        auto bitmap = combined_bitmap(begin, end);
        return sse<ArrowType>(bitmap, begin, end);
    } else {
        using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
        std::vector<EigenVector> columns;
        auto n = std::distance(begin, end);
        columns.reserve(n);

        for (auto it = begin; it != end; ++it) {
            auto c = to_eigen<false, ArrowType, false>(*it);
            auto m = c->mean();
            columns.push_back(c->array() - m);
        }

        return compute_sse<ArrowType>(columns);
    }
}

template <typename ArrowType>
EigenMatrix<ArrowType> sse(Buffer_ptr bitmap, Array_ptr col) {
    using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
    std::vector<EigenVector> columns;
    columns.reserve(1);

    auto c = to_eigen<false, ArrowType>(bitmap, col);
    auto m = c->mean();
    columns.push_back(c->array() - m);

    return compute_sse<ArrowType>(columns);
}

template <typename ArrowType, bool contains_null>
EigenMatrix<ArrowType> sse(Array_ptr col) {
    if constexpr (contains_null) {
        auto bitmap = col->null_bitmap();
        return sse<ArrowType>(bitmap, col);
    } else {
        using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;
        std::vector<EigenVector> columns;
        columns.reserve(1);

        auto c = to_eigen<false, ArrowType, false>(col);
        auto m = c->mean();
        columns.push_back(c->array() - m);

        return compute_sse<ArrowType>(columns);
    }
}

// //////////////////////////////////// IndexLOC //////////////////////////////
template <bool copy, typename... Args>
class IndexLOC {
public:
    IndexLOC(Args... args) : m_args(args...) {}

    const std::tuple<Args...>& columns() const { return m_args; }

private:
    const std::tuple<Args...> m_args;
};

template <typename... Args>
using CopyLOC = IndexLOC<true, Args...>;
template <typename... Args>
using MoveLOC = IndexLOC<false, Args...>;

// This is necessary because C++17 do not allow alias template type deduction.
template <typename... Args>
CopyLOC<Args...> Copy(Args... args) {
    return CopyLOC<Args...>(args...);
}
template <typename... Args>
MoveLOC<Args...> Move(Args... args) {
    return MoveLOC<Args...>(args...);
}

template <typename T>
struct dataframe_traits;

class DataFrame;
template <>
struct dataframe_traits<DataFrame> {
    template <typename T, typename R>
    using enable_if_index_t = util::enable_if_index_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_container_t = util::enable_if_index_container_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_iterator_t = util::enable_if_index_iterator_t<T, R>;
    using loc_return = DataFrame;
};

std::string index_to_string(int i);
std::string index_to_string(const std::string& name);
std::string index_to_string(const DynamicVariable<int>& i);
std::string index_to_string(const DynamicVariable<std::string>& name);

template <typename T, util::enable_if_index_container_t<T, int> = 0>
inline int size_argument(int, const T& arg) {
    return arg.size();
}

template <typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
inline int size_argument(int, const std::pair<IndexIter, IndexIter>& it) {
    return std::distance(it.first, it.second);
}

inline int size_argument(int, int) { return 1; }

template <typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
inline int size_argument(int, const StringType&) {
    return 1;
}

template <typename Index>
inline int size_argument(int, const DynamicVariable<Index>&) {
    return 1;
}

template <typename T, util::enable_if_dynamic_index_container_t<T, int> = 0>
inline int size_argument(int, const T& arg) {
    return arg.size();
}

template <typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int> = 0>
inline int size_argument(int, const std::pair<IndexIter, IndexIter>& it) {
    return std::distance(it.first, it.second);
}

template <bool copy, typename... Args>
inline int size_argument(int total_columns, const IndexLOC<copy, Args...>& cols) {
    if constexpr (std::tuple_size_v<std::remove_reference_t<decltype(cols.columns())>> == 0) {
        return total_columns;
    } else {
        return std::apply([total_columns](const auto&... args) { return (size_argument(total_columns, args) + ...); },
                          cols.columns());
    }
}

template <typename Composition,
          typename FuncObj,
          typename Index,
          typename FuncObj::template enable_if_index_t<Index, int> = 0>
inline void append_generic(Composition& comp, const FuncObj& f, const Index& index) {
    f(comp, index);
}

template <typename Composition,
          typename FuncObj,
          typename IndexIter,
          typename FuncObj::template enable_if_index_iterator_t<IndexIter, int> = 0>
inline void append_generic(Composition& comp, const FuncObj& f, const IndexIter& begin, const IndexIter& end) {
    for (auto it = begin; it != end; ++it) {
        f(comp, *it);
    }
}

template <typename Composition,
          typename FuncObj,
          typename IndexIter,
          typename FuncObj::template enable_if_index_iterator_t<IndexIter, int> = 0>
inline void append_generic(Composition& comp, const FuncObj& f, const std::pair<IndexIter, IndexIter>& t) {
    append_generic(comp, f, t.first, t.second);
}

template <typename Composition,
          typename FuncObj,
          typename T,
          typename FuncObj::template enable_if_index_container_t<T, int> = 0>
inline void append_generic(Composition& comp, const FuncObj& f, const T& arg) {
    append_generic(comp, f, arg.begin(), arg.end());
}

template <typename Composition,
          typename FuncObj,
          typename... Args,
          std::enable_if_t<(... && !util::is_iterator_v<Args>), int> = 0>
inline void append_generic(Composition& comp, const FuncObj& f, const CopyLOC<Args...>& cols) {
    if constexpr (std::tuple_size_v<std::remove_reference_t<decltype(cols.columns())>> == 0) {
        f(comp);
    } else {
        f(comp, cols);
    }
}

template <typename Composition,
          typename FuncObj,
          typename... Args,
          std::enable_if_t<(... && !util::is_iterator_v<Args>), int> = 0>
inline void append_generic(Composition& comp, const FuncObj& f, const MoveLOC<Args...>& cols) {
    if constexpr (std::tuple_size_v<std::remove_reference_t<decltype(cols.columns())>> == 0) {
        f(comp);
    } else {
        f(comp, cols);
    }
}

template <typename Composition,
          typename FuncObj,
          typename... Args,
          typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
inline void append_generic(Composition& comp, const FuncObj& f, const Args&... args) {
    (append_generic(comp, f, args), ...);
}

template <typename DataFrame>
struct AppendCopyColumns {
    template <typename T, typename R>
    using enable_if_index_t = typename DataFrame::template enable_if_index_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_container_t = typename DataFrame::template enable_if_index_container_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_iterator_t = typename DataFrame::template enable_if_index_iterator_t<T, R>;

    template <typename Composition>
    void operator()(Composition& vector) const {
        for (const auto& col : self.derived().columns()) {
            vector.push_back(copy_array(col));
        }
    }

    template <typename Composition, typename Index, enable_if_index_t<Index, int> = 0>
    void operator()(Composition& vector, const Index& index) const {
        return vector.push_back(copy_array(self.derived().col(index)));
    }

    template <typename Composition, typename... Args>
    void operator()(Composition& vector, const CopyLOC<Args...>& indices) const {
        std::apply([&vector, this](const auto&... args) { (append_generic(vector, *this, args), ...); },
                   indices.columns());
    }

    const DataFrame& self;
};

template <typename DataFrame>
struct AppendColumns {
    template <typename T, typename R>
    using enable_if_index_t = typename DataFrame::template enable_if_index_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_container_t = typename DataFrame::template enable_if_index_container_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_iterator_t = typename DataFrame::template enable_if_index_iterator_t<T, R>;

    template <typename Composition>
    void operator()(Composition& comp) const {
        for (const auto& col : self.derived().columns()) {
            comp.push_back(col);
        }
    }

    template <typename Composition, typename Index, enable_if_index_t<Index, int> = 0>
    void operator()(Composition& comp, const Index& index) const {
        comp.push_back(self.derived().col(index));
    }

    template <typename Composition, typename... Args>
    void operator()(Composition& comp, const CopyLOC<Args...>& indices) const {
        AppendCopyColumns<DataFrame> copy{/*.self = */ self};

        std::apply([&comp, &copy](const auto&... args) { (append_generic(comp, copy, args), ...); }, indices.columns());
    }

    template <typename Composition, typename... Args>
    void operator()(Composition& comp, const MoveLOC<Args...>& indices) const {
        std::apply([&comp, this](const auto&... args) { (append_generic(comp, *this, args), ...); }, indices.columns());
    }

    const DataFrame& self;
};

template <typename DataFrame>
struct AppendSchema {
    template <typename T, typename R>
    using enable_if_index_t = typename DataFrame::template enable_if_index_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_container_t = typename DataFrame::template enable_if_index_container_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_iterator_t = typename DataFrame::template enable_if_index_iterator_t<T, R>;

    template <typename Composition>
    void operator()(Composition& comp) const {
        auto schema = self.derived().schema();

        for (auto i = 0; i < schema->num_fields(); ++i) {
            RAISE_STATUS_ERROR(comp.AddField(schema->field(i)));
        }
    }

    template <typename Composition, typename Index, enable_if_index_t<Index, int> = 0>
    void operator()(Composition& comp, const Index& index) const {
        RAISE_STATUS_ERROR(comp.AddField(self.derived().field(index)));
    }

    template <typename Composition, bool copy, typename... Args>
    void operator()(Composition& comp, const IndexLOC<copy, Args...>& indices) const {
        std::apply([&comp, this](const auto&... args) { (append_generic(comp, *this, args), ...); }, indices.columns());
    }

    const DataFrame& self;
};

template <typename DataFrame>
struct AppendNames {
    template <typename T, typename R>
    using enable_if_index_t = typename DataFrame::template enable_if_index_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_container_t = typename DataFrame::template enable_if_index_container_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_iterator_t = typename DataFrame::template enable_if_index_iterator_t<T, R>;

    template <typename Composition, typename Index, enable_if_index_t<Index, int> = 0>
    void operator()(Composition& comp, const Index& index) const {
        comp.push_back(self.derived().name(index));
    }

    const DataFrame& self;
};

template <typename Derived>
class DataFrameBase {
public:
    template <typename T, typename R>
    using enable_if_index_t = typename dataframe_traits<Derived>::template enable_if_index_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_container_t = typename dataframe_traits<Derived>::template enable_if_index_container_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_iterator_t = typename dataframe_traits<Derived>::template enable_if_index_iterator_t<T, R>;
    using loc_return = typename dataframe_traits<Derived>::loc_return;

    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    Derived& derived() { return static_cast<Derived&>(*this); }

    ///////////////////////////// has_columns /////////////////////////
    template <typename Index, enable_if_index_t<Index, int> = 0>
    bool has_columns(const Index& index) const {
        return derived().has_column(index);
    }
    template <typename T, enable_if_index_container_t<T, int> = 0>
    bool has_columns(const T& cols) const {
        return has_columns(cols.begin(), cols.end());
    }
    template <typename V>
    bool has_columns(const std::initializer_list<V>& cols) const {
        return has_columns(cols.begin(), cols.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    bool has_columns(const IndexIter& begin, const IndexIter& end) const;
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    bool has_columns(const std::pair<IndexIter, IndexIter>& tuple) const {
        return has_columns(tuple.first, tuple.second);
    }

#ifdef _MSC_VER
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
#else
    template <typename... Args, std::enable_if_t<(... && !util::is_iterator_v<Args>), int> = 0>
#endif
    bool has_columns(const Args&... args) const;

    ///////////////////////////// raise_has_columns /////////////////////////
    template <typename Index, enable_if_index_t<Index, int> = 0>
    void raise_has_columns(const Index& index) const {
        return derived().raise_has_column(index);
    }
    template <typename T, enable_if_index_container_t<T, int> = 0>
    void raise_has_columns(const T& cols) const {
        return raise_has_columns(cols.begin(), cols.end());
    }
    template <typename V>
    void raise_has_columns(const std::initializer_list<V>& cols) const {
        return raise_has_columns(cols.begin(), cols.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    void raise_has_columns(const IndexIter& begin, const IndexIter& end) const;
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    void raise_has_columns(const std::pair<IndexIter, IndexIter>& tuple) const {
        return raise_has_columns(tuple.first, tuple.second);
    }

#ifdef _MSC_VER
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
#else
    template <typename... Args, std::enable_if_t<(... && !util::is_iterator_v<Args>), int> = 0>
#endif
    void raise_has_columns(const Args&... args) const;

    ///////////////////////////// loc /////////////////////////
    template <typename Index, enable_if_index_t<Index, int> = 0>
    loc_return loc(const Index& index) const;
    template <typename T, enable_if_index_container_t<T, int> = 0>
    loc_return loc(const T& cols) const {
        return loc(cols.begin(), cols.end());
    }
    template <typename V>
    loc_return loc(const std::initializer_list<V>& cols) const {
        return loc(cols.begin(), cols.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    loc_return loc(const IndexIter& begin, const IndexIter& end) const;
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    loc_return loc(const std::pair<IndexIter, IndexIter>& tuple) const {
        return loc(tuple.first, tuple.second);
    }

#ifdef _MSC_VER
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
#else
    template <typename... Args, std::enable_if_t<(... && !util::is_iterator_v<Args>), int> = 0>
#endif
    loc_return loc(const Args&... args) const;

    ///////////////////////////// same_type /////////////////////////
    std::shared_ptr<arrow::DataType> same_type() const {
        Array_vector cols = derived().columns();
        return dataset::same_type(cols.begin(), cols.end());
    }
    template <typename Index, enable_if_index_t<Index, int> = 0>
    std::shared_ptr<arrow::DataType> same_type(const Index& index) const {
        return derived().col(index)->type();
    }
    template <typename T, enable_if_index_container_t<T, int> = 0>
    std::shared_ptr<arrow::DataType> same_type(const T& cols) const {
        return same_type(cols.begin(), cols.end());
    }
    template <typename V>
    std::shared_ptr<arrow::DataType> same_type(const std::initializer_list<V>& cols) const {
        return same_type(cols.begin(), cols.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    std::shared_ptr<arrow::DataType> same_type(const IndexIter& begin, const IndexIter& end) const {
        auto v = indices_to_columns(begin, end);
        return dataset::same_type(v.begin(), v.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    std::shared_ptr<arrow::DataType> same_type(const std::pair<IndexIter, IndexIter>& tuple) const {
        return same_type(tuple.first, tuple.second);
    }
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    std::shared_ptr<arrow::DataType> same_type(const Args&... args) const {
        auto v = indices_to_columns(args...);
        return dataset::same_type(v.begin(), v.end());
    }

    ///////////////////////////// is_discrete/is_continuous /////////////////////////
    template <typename Index, enable_if_index_t<Index, int> = 0>
    bool is_discrete(const Index& index) const {
        if (derived().col(index)->type_id() == Type::DICTIONARY) {
            return true;
        }

        return false;
    }

    template <typename Index, enable_if_index_t<Index, int> = 0>
    bool is_continuous(const Index& index) const {
        switch (derived().col(index)->type_id()) {
            case Type::DOUBLE:
            case Type::FLOAT:
                return true;
            default:
                return false;
        }
    }

    ///////////////////////////// names /////////////////////////
    std::vector<std::string> names() const { return derived().column_names(); }
    template <typename Index, enable_if_index_t<Index, int> = 0>
    std::vector<std::string> names(const Index& index) const {
        return {derived().name(index)};
    }
    template <typename T, enable_if_index_container_t<T, int> = 0>
    std::vector<std::string> names(const T& n) const {
        return names(n.begin(), n.end());
    }
    template <typename V>
    std::vector<std::string> names(const std::initializer_list<V>& n) const {
        return names(n.begin(), n.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    std::vector<std::string> names(const IndexIter& begin, const IndexIter& end) const {
        std::vector<std::string> res;
        res.reserve(std::distance(begin, end));

        AppendNames<DataFrameBase<Derived>> func{*this};
        append_generic(res, func, begin, end);
        return res;
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    std::vector<std::string> names(const std::pair<IndexIter, IndexIter>& tuple) const {
        return names(tuple.first, tuple.second);
    }
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    std::vector<std::string> names(const Args&... args) {
        int total_size = (size_argument(args) + ...);
        std::vector<std::string> res;
        res.reserve(total_size);

        AppendNames<DataFrameBase<Derived>> func{*this};
        append_generic(res, func, args...);

        return res;
    }

    ///////////////////////////// data /////////////////////////
    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    const typename ArrowType::c_type* data(const Index& index) const {
        return derived().col(index)->data()->template GetValues<typename ArrowType::c_type>(1);
    }

    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    typename ArrowType::c_type* mutable_data(const Index& index) const {
        return derived().col(index)->data()->template GetMutableValues<typename ArrowType::c_type>(1);
    }

    ///////////////////////////// downcast_vector /////////////////////////
    template <typename ArrowType>
    std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> downcast_vector() const {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        Array_vector v = indices_to_columns();
        std::vector<std::shared_ptr<ArrayType>> res;
        res.reserve(v.size());
        for (auto& array : v) {
            res.push_back(std::static_pointer_cast<ArrayType>(array));
        }

        return res;
    }
    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType> downcast(const Index& index) const {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        auto a = derived().col(index);
        return std::static_pointer_cast<ArrayType>(a);
    }
    template <typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> downcast_vector(const T& n) const {
        return downcast_vector<ArrowType>(n.begin(), n.end());
    }
    template <typename ArrowType, typename V>
    std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> downcast_vector(
        const std::initializer_list<V>& n) const {
        return downcast_vector<ArrowType>(n.begin(), n.end());
    }

    template <typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> downcast_vector(
        const IndexIter& begin, const IndexIter& end) const {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        Array_vector v = indices_to_columns(begin, end);
        std::vector<std::shared_ptr<ArrayType>> res;
        res.reserve(v.size());
        for (auto& array : v) {
            res.push_back(std::static_pointer_cast<ArrayType>(array));
        }

        return res;
    }
    template <typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> downcast_vector(
        const std::pair<IndexIter, IndexIter>& tuple) const {
        return downcast_vector(tuple.first, tuple.second);
    }
    template <typename ArrowType,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    std::vector<std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>> downcast_vector(
        const Args&... args) const {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        Array_vector v = indices_to_columns(args...);
        std::vector<std::shared_ptr<ArrayType>> res;
        res.reserve(v.size());
        for (auto& array : v) {
            res.push_back(std::static_pointer_cast<ArrayType>(array));
        }

        return res;
    }

    ///////////////////////////// combined_bitmap /////////////////////////
    Buffer_ptr combined_bitmap() const {
        Array_vector cols = derived().columns();
        return dataset::combined_bitmap(cols.begin(), cols.end());
    }
    template <typename Index, enable_if_index_t<Index, int> = 0>
    Buffer_ptr combined_bitmap(const Index& index) const {
        return derived().col(index)->null_bitmap();
    }
    template <typename T, util::enable_if_index_container_t<T, int> = 0>
    Buffer_ptr combined_bitmap(const T& cols) const {
        Array_vector v = indices_to_columns(cols);
        return dataset::combined_bitmap(v.begin(), v.end());
    }
    template <typename V>
    Buffer_ptr combined_bitmap(const std::initializer_list<V>& cols) const {
        return combined_bitmap(cols.begin(), cols.end());
    }
    template <typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    Buffer_ptr combined_bitmap(const IndexIter& begin, const IndexIter& end) const {
        Array_vector v = indices_to_columns(begin, end);
        return dataset::combined_bitmap(v.begin(), v.end());
    }
    template <typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    Buffer_ptr combined_bitmap(const std::pair<IndexIter, IndexIter>& tuple) const {
        return combined_bitmap(tuple.first, tuple.second);
    }
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    Buffer_ptr combined_bitmap(const Args&... args) const {
        Array_vector v = indices_to_columns(args...);
        return dataset::combined_bitmap(v.begin(), v.end());
    }

    ///////////////////////////// null_count /////////////////////////
    int64_t null_count() const {
        auto cols = derived().columns();
        return dataset::null_count(cols.begin(), cols.end());
    }
    template <typename Index, enable_if_index_t<Index, int> = 0>
    int64_t null_count(const Index& index) const {
        return derived().col(index)->null_count();
    }
    template <typename T, util::enable_if_index_container_t<T, int> = 0>
    int64_t null_count(const T& cols) const {
        Array_vector v = indices_to_columns(cols);
        return dataset::null_count(v.begin(), v.end());
    }
    template <typename V>
    int64_t null_count(const std::initializer_list<V>& cols) const {
        return null_count(cols.begin(), cols.end());
    }
    template <typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    int64_t null_count(const IndexIter& begin, const IndexIter& end) const {
        Array_vector v = indices_to_columns(begin, end);
        return dataset::null_count(v.begin(), v.end());
    }
    template <typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    int64_t null_count(const std::pair<IndexIter, IndexIter>& tuple) const {
        return null_count(tuple.first, tuple.second);
    }
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    int64_t null_count(const Args&... args) const {
        Array_vector v = indices_to_columns(args...);
        return dataset::null_count(v.begin(), v.end());
    }

    ///////////////////////////// valid_rows /////////////////////////
    int64_t valid_rows() const {
        auto cols = derived().columns();
        return dataset::valid_rows(cols.begin(), cols.end());
    }
    template <typename Index, enable_if_index_t<Index, int> = 0>
    int64_t valid_rows(const Index& index) const {
        return derived().num_rows() - derived().col(index)->null_count();
    }
    template <typename T, enable_if_index_container_t<T, int> = 0>
    int64_t valid_rows(const T& cols) const {
        return valid_rows(cols.begin(), cols.end());
    }
    template <typename V>
    int64_t valid_rows(const std::initializer_list<V>& cols) const {
        return valid_rows(cols.begin(), cols.end());
    }
    template <typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    int64_t valid_rows(const IndexIter& begin, const IndexIter& end) const {
        auto v = indices_to_columns(begin, end);
        return dataset::valid_rows(v.begin(), v.end());
    }
    template <typename IndexIter, util::enable_if_index_iterator_t<IndexIter, int> = 0>
    int64_t valid_rows(const std::pair<IndexIter, IndexIter>& tuple) const {
        return valid_rows(tuple.first, tuple.second);
    }
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    int64_t valid_rows(const Args&... args) const {
        auto v = indices_to_columns(args...);
        return dataset::valid_rows(v.begin(), v.end());
    }

    ///////////////////////////// min /////////////////////////
    template <typename Index, util::enable_if_index_t<Index, int> = 0>
    double min(const Index& index) const {
        auto a = derived().col(index);

        switch (a->type_id()) {
            case Type::DOUBLE:
                return dataset::min<arrow::DoubleType>(a);
            case Type::FLOAT:
                return static_cast<double>(dataset::min<arrow::FloatType>(a));
            default:
                throw std::invalid_argument("min() only implemented for \"double\" and \"float\" data types.");
        }
    }

    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    typename ArrowType::c_type min(const Index& index) const {
        return dataset::min<ArrowType>(derived().col(index));
    }

    ///////////////////////////// max /////////////////////////
    template <typename Index, util::enable_if_index_t<Index, int> = 0>
    double max(const Index& index) const {
        auto a = derived().col(index);

        switch (a->type_id()) {
            case Type::DOUBLE:
                return dataset::max<arrow::DoubleType>(a);
            case Type::FLOAT:
                return static_cast<double>(dataset::max<arrow::FloatType>(a));
            default:
                throw std::invalid_argument("max() only implemented for \"double\" and \"float\" data types.");
        }
    }

    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    typename ArrowType::c_type max(const Index& index) const {
        return dataset::max<ArrowType>(derived().col(index));
    }

    ///////////////////////////// means /////////////////////////
    template <typename Index, util::enable_if_index_t<Index, int> = 0>
    double mean(const Index& index) const {
        return dataset::mean(derived().col(index));
    }

    template <typename Index, util::enable_if_index_t<Index, int> = 0>
    double mean(const Buffer_ptr& bitmap, const Index& index) const {
        auto a = derived().col(index);
        return dataset::mean(bitmap, a);
    }

    template <typename ArrowType, typename Index, util::enable_if_index_t<Index, int> = 0>
    typename ArrowType::c_type mean(const Index& n) const {
        return dataset::mean<ArrowType>(derived().col(n));
    }

    template <typename ArrowType, typename Index, util::enable_if_index_t<Index, int> = 0>
    typename ArrowType::c_type mean(const Buffer_ptr& bitmap, const Index& n) const {
        return dataset::mean<ArrowType>(bitmap, derived().col(n));
    }

    template <typename Index, enable_if_index_t<Index, int> = 0>
    VectorXd means(const Index& index) const {
        VectorXd res(1);
        res(0) = mean(index);
        return res;
    }
    template <typename T, enable_if_index_container_t<T, int> = 0>
    VectorXd means() const {
        Array_vector cols = derived().columns();
        return means(cols.begin(), cols.end());
    }
    template <typename T, enable_if_index_container_t<T, int> = 0>
    VectorXd means(const T& cols) const {
        return means(cols.begin(), cols.end());
    }
    template <typename V>
    VectorXd means(const std::initializer_list<V>& cols) const {
        return means(cols.begin(), cols.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    VectorXd means(const IndexIter& begin, const IndexIter& end) const {
        auto c = indices_to_columns(begin, end);
        return dataset::means(c.begin(), c.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    VectorXd means(const std::pair<IndexIter, IndexIter>& tuple) const {
        return means(tuple.first, tuple.second);
    }
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    VectorXd means(const Args&... args) const {
        auto c = indices_to_columns(args...);
        return dataset::means(c.begin(), c.end());
    }

    template <typename Index, enable_if_index_t<Index, int> = 0>
    VectorXd means(const Buffer_ptr& bitmap, const Index& index) const {
        VectorXd res(1);
        res(0) = mean(bitmap, index);
        return res;
    }
    template <typename T, enable_if_index_container_t<T, int> = 0>
    VectorXd means(const Buffer_ptr& bitmap) const {
        Array_vector cols = derived().columns();
        return means(bitmap, cols.begin(), cols.end());
    }
    template <typename T, enable_if_index_container_t<T, int> = 0>
    VectorXd means(const Buffer_ptr& bitmap, const T& cols) const {
        return means(bitmap, cols.begin(), cols.end());
    }
    template <typename V>
    VectorXd means(const Buffer_ptr& bitmap, const std::initializer_list<V>& cols) const {
        return means(bitmap, cols.begin(), cols.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    VectorXd means(const Buffer_ptr& bitmap, const IndexIter& begin, const IndexIter& end) const {
        auto c = indices_to_columns(begin, end);
        return dataset::means(bitmap, c.begin(), c.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    VectorXd means(const Buffer_ptr& bitmap, const std::pair<IndexIter, IndexIter>& tuple) const {
        return means(bitmap, tuple.first, tuple.second);
    }
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    VectorXd means(const Buffer_ptr& bitmap, const Args&... args) const {
        auto c = indices_to_columns(args...);
        return dataset::means(bitmap, c.begin(), c.end());
    }

    ///////////////////////////// means<ArrowType> /////////////////////////
    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const Index& index) const {
        Matrix<typename ArrowType::c_type, Dynamic, 1> res(1);
        res(0) = mean<ArrowType>(index);
        return res;
    }
    template <typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means() const {
        Array_vector cols = derived().columns();
        return means<ArrowType>(cols.begin(), cols.end());
    }
    template <typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const T& cols) const {
        return means<ArrowType>(cols.begin(), cols.end());
    }
    template <typename ArrowType, typename V>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const std::initializer_list<V>& cols) const {
        return means(cols.begin(), cols.end());
    }
    template <typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const IndexIter& begin, const IndexIter& end) const {
        auto c = indices_to_columns(begin, end);
        return dataset::means<ArrowType>(c.begin(), c.end());
    }
    template <typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const std::pair<IndexIter, IndexIter>& tuple) const {
        return means(tuple.first, tuple.second);
    }
    template <typename ArrowType,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const Args&... args) const {
        auto c = indices_to_columns(args...);
        return dataset::means<ArrowType>(c.begin(), c.end());
    }

    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const Buffer_ptr& bitmap, const Index& index) const {
        Matrix<typename ArrowType::c_type, Dynamic, 1> res(1);
        res(0) = mean<ArrowType>(bitmap, index);
        return res;
    }
    template <typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const Buffer_ptr& bitmap) const {
        Array_vector cols = derived().columns();
        return means<ArrowType>(bitmap, cols.begin(), cols.end());
    }
    template <typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const Buffer_ptr& bitmap, const T& cols) const {
        return means<ArrowType>(bitmap, cols.begin(), cols.end());
    }
    template <typename ArrowType, typename V>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const Buffer_ptr& bitmap,
                                                         const std::initializer_list<V>& cols) const {
        return means(bitmap, cols.begin(), cols.end());
    }
    template <typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const Buffer_ptr& bitmap,
                                                         const IndexIter& begin,
                                                         const IndexIter& end) const {
        auto c = indices_to_columns(begin, end);
        return dataset::means<ArrowType>(bitmap, c.begin(), c.end());
    }
    template <typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const Buffer_ptr& bitmap,
                                                         const std::pair<IndexIter, IndexIter>& tuple) const {
        return means(bitmap, tuple.first, tuple.second);
    }
    template <typename ArrowType,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    Matrix<typename ArrowType::c_type, Dynamic, 1> means(const Buffer_ptr& bitmap, const Args&... args) const {
        auto c = indices_to_columns(args...);
        return dataset::means<ArrowType>(bitmap, c.begin(), c.end());
    }

    ///////////////////////////// to_eigen<ArrowType> /////////////////////////
    template <bool append_ones, typename ArrowType, bool contains_null>
    EigenMatrix<ArrowType> to_eigen() const {
        auto cols = derived().columns();
        return dataset::to_eigen<append_ones, ArrowType, contains_null>(cols.begin(), cols.end());
    }
    template <bool append_ones, typename ArrowType>
    EigenMatrix<ArrowType> to_eigen() const {
        auto cols = derived().columns();
        if (null_count() == 0) {
            return dataset::to_eigen<append_ones, ArrowType, false>(cols.begin(), cols.end());
        } else {
            return dataset::to_eigen<append_ones, ArrowType, true>(cols.begin(), cols.end());
        }
    }
    template <bool append_ones, typename ArrowType>
    EigenMatrix<ArrowType> to_eigen(const Buffer_ptr& bitmap) const {
        auto cols = derived().columns();
        return dataset::to_eigen<append_ones, ArrowType>(bitmap, cols.begin(), cols.end());
    }

    template <bool append_ones,
              typename ArrowType,
              bool contains_null,
              typename Index,
              enable_if_index_t<Index, int> = 0>
    MapOrMatrixType<append_ones, ArrowType, contains_null> to_eigen(const Index& index) const {
        auto col = derived().col(index);
        return dataset::to_eigen<append_ones, ArrowType, contains_null>(col);
    }
    template <bool append_ones, typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(const Index& index) const {
        auto col = derived().col(index);
        if (col->null_count() == 0) {
            using EigenReturnType = typename EigenVectorOrMatrix<append_ones, ArrowType>::element_type;
            auto map = dataset::to_eigen<append_ones, ArrowType, false>(col);
            return std::make_unique<EigenReturnType>(*map);
        } else {
            return dataset::to_eigen<append_ones, ArrowType, true>(col);
        }
    }
    template <bool append_ones, typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    EigenVectorOrMatrix<append_ones, ArrowType> to_eigen(const Buffer_ptr& bitmap, const Index& index) const {
        auto col = derived().col(index);
        return dataset::to_eigen<append_ones, ArrowType>(bitmap, col);
    }

    template <bool append_ones,
              typename ArrowType,
              bool contains_null,
              typename T,
              enable_if_index_container_t<T, int> = 0>
    EigenMatrix<ArrowType> to_eigen(const T& cols) const {
        Array_vector v = indices_to_columns(cols);
        return dataset::to_eigen<append_ones, ArrowType, contains_null>(v.begin(), v.end());
    }
    template <bool append_ones, typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    EigenMatrix<ArrowType> to_eigen(const T& cols) const {
        if (null_count(cols) == 0) {
            return to_eigen<append_ones, ArrowType, false>(cols);
        } else {
            return to_eigen<append_ones, ArrowType, true>(cols);
        }
    }
    template <bool append_ones, typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    EigenMatrix<ArrowType> to_eigen(const Buffer_ptr& bitmap, const T& cols) const {
        Array_vector v = indices_to_columns(cols);
        return dataset::to_eigen<append_ones, ArrowType>(bitmap, v.begin(), v.end());
    }

    template <bool append_ones, typename ArrowType, bool contains_null, typename V>
    EigenMatrix<ArrowType> to_eigen(const std::initializer_list<V>& cols) const {
        return to_eigen<append_ones, ArrowType, contains_null>(cols.begin(), cols.end());
    }
    template <bool append_ones, typename ArrowType, typename V>
    EigenMatrix<ArrowType> to_eigen(const std::initializer_list<V>& cols) const {
        return to_eigen<append_ones, ArrowType>(cols.begin(), cols.end());
    }
    template <bool append_ones, typename ArrowType, typename V>
    EigenMatrix<ArrowType> to_eigen(const Buffer_ptr& bitmap, const std::initializer_list<V>& cols) const {
        return to_eigen<append_ones, ArrowType>(bitmap, cols.begin(), cols.end());
    }

    template <bool append_ones,
              typename ArrowType,
              bool contains_null,
              typename IndexIter,
              enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> to_eigen(const IndexIter& begin, const IndexIter& end) const {
        Array_vector v = indices_to_columns(begin, end);
        return dataset::to_eigen<append_ones, ArrowType, contains_null>(v.begin(), v.end());
    }
    template <bool append_ones, typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> to_eigen(const IndexIter& begin, const IndexIter& end) const {
        if (null_count(begin, end) == 0) {
            return to_eigen<append_ones, ArrowType, false>(begin, end);
        } else {
            return to_eigen<append_ones, ArrowType, true>(begin, end);
        }
    }
    template <bool append_ones, typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> to_eigen(const Buffer_ptr& bitmap, const IndexIter& begin, const IndexIter& end) const {
        Array_vector v = indices_to_columns(begin, end);
        return dataset::to_eigen<append_ones, ArrowType>(bitmap, v.begin(), v.end());
    }

    template <bool append_ones,
              typename ArrowType,
              bool contains_null,
              typename IndexIter,
              enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> to_eigen(const std::pair<IndexIter, IndexIter>& tuple) const {
        return to_eigen<append_ones, ArrowType, contains_null>(tuple.first, tuple.second);
    }
    template <bool append_ones, typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> to_eigen(const std::pair<IndexIter, IndexIter>& tuple) const {
        return to_eigen<append_ones, ArrowType>(tuple.first, tuple.second);
    }
    template <bool append_ones, typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> to_eigen(const Buffer_ptr& bitmap, const std::pair<IndexIter, IndexIter>& tuple) const {
        return to_eigen<append_ones, ArrowType>(bitmap, tuple.first, tuple.second);
    }

    template <bool append_ones,
              typename ArrowType,
              bool contains_null,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    EigenMatrix<ArrowType> to_eigen(const Args&... args) const {
        Array_vector v = indices_to_columns(args...);
        return dataset::to_eigen<append_ones, ArrowType, contains_null>(v.begin(), v.end());
    }
    template <bool append_ones,
              typename ArrowType,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    EigenMatrix<ArrowType> to_eigen(const Args&... args) const {
        if (null_count(args...) == 0) {
            return to_eigen<append_ones, ArrowType, false>(args...);
        } else {
            return to_eigen<append_ones, ArrowType, true>(args...);
        }
    }
    template <bool append_ones,
              typename ArrowType,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    EigenMatrix<ArrowType> to_eigen(const Buffer_ptr& bitmap, const Args&... args) const {
        Array_vector v = indices_to_columns(args...);
        return dataset::to_eigen<append_ones, ArrowType>(bitmap, v.begin(), v.end());
    }

    ///////////////////////////// cov<ArrowType> /////////////////////////
    template <typename ArrowType, bool contains_null>
    EigenMatrix<ArrowType> cov() {
        auto c = derived().columns();
        return dataset::cov<ArrowType, contains_null>(c.begin(), c.end());
    }
    template <typename ArrowType>
    EigenMatrix<ArrowType> cov() {
        if (null_count() == 0) {
            return cov<ArrowType, false>();
        } else {
            return cov<ArrowType, true>();
        }
    }
    template <typename ArrowType>
    EigenMatrix<ArrowType> cov(const Buffer_ptr& bitmap) {
        auto c = derived().columns();
        return cov<ArrowType>(bitmap, c.begin(), c.end());
    }

    template <typename ArrowType, bool contains_null, typename Index, enable_if_index_t<Index, int> = 0>
    typename ArrowType::c_type cov(const Index& index) const {
        return dataset::cov<ArrowType, contains_null>(derived().col(index));
    }
    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    typename ArrowType::c_type cov(const Index& index) const {
        if (null_count(index) == 0) {
            return cov<ArrowType, false>(index);
        } else {
            return cov<ArrowType, true>(index);
        }
    }
    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    typename ArrowType::c_type cov(const Buffer_ptr& bitmap, const Index& index) const {
        return dataset::cov<ArrowType>(bitmap, derived().col(index));
    }

    template <typename ArrowType, bool contains_null, typename T, enable_if_index_container_t<T, int> = 0>
    EigenMatrix<ArrowType> cov(const T& cols) {
        auto c = indices_to_columns(cols);
        return dataset::cov<ArrowType, contains_null>(c.begin(), c.end());
    }
    template <typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    EigenMatrix<ArrowType> cov(const T& cols) {
        if (null_count(cols) == 0) {
            return cov<ArrowType, false>(cols);
        } else {
            return cov<ArrowType, true>(cols);
        }
    }
    template <typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    EigenMatrix<ArrowType> cov(const Buffer_ptr& bitmap, const T& cols) {
        auto c = indices_to_columns(cols);
        return dataset::cov<ArrowType>(bitmap, c.begin(), c.end());
    }

    template <typename ArrowType, bool contains_null, typename V>
    EigenMatrix<ArrowType> cov(const std::initializer_list<V>& cols) {
        auto c = indices_to_columns(cols);
        return dataset::cov<ArrowType, contains_null, std::initializer_list<V>>(c.begin(), c.end());
    }
    template <typename ArrowType, typename V>
    EigenMatrix<ArrowType> cov(const std::initializer_list<V>& cols) {
        if (null_count(cols) == 0) {
            return cov<ArrowType, false, std::initializer_list<V>>(cols);
        } else {
            return cov<ArrowType, true, std::initializer_list<V>>(cols);
        }
    }
    template <typename ArrowType, typename V>
    EigenMatrix<ArrowType> cov(const Buffer_ptr& bitmap, const std::initializer_list<V>& cols) {
        auto c = indices_to_columns(cols);
        return dataset::cov<ArrowType>(bitmap, c.begin(), c.end());
    }

    template <typename ArrowType,
              bool contains_null,
              typename IndexIter,
              enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> cov(const IndexIter& begin, const IndexIter& end) const {
        auto c = indices_to_columns(begin, end);
        return dataset::cov<ArrowType, contains_null>(c.begin(), c.end());
    }
    template <typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> cov(const IndexIter& begin, const IndexIter& end) const {
        if (null_count(begin, end) == 0) {
            return cov<ArrowType, false>(begin, end);
        } else {
            return cov<ArrowType, true>(begin, end);
        }
    }
    template <typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> cov(const Buffer_ptr& bitmap, const IndexIter& begin, const IndexIter& end) const {
        auto c = indices_to_columns(begin, end);
        return dataset::cov<ArrowType>(bitmap, c.begin(), c.end());
    }

    template <typename ArrowType,
              bool contains_null,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    EigenMatrix<ArrowType> cov(const Args&... args) const {
        auto c = indices_to_columns(args...);
        return dataset::cov<ArrowType, contains_null>(c.begin(), c.end());
    }
    template <typename ArrowType,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    EigenMatrix<ArrowType> cov(const Args&... args) const {
        if (null_count(args...) == 0) {
            return cov<ArrowType, false>(args...);
        } else {
            return cov<ArrowType, true>(args...);
        }
    }
    template <typename ArrowType,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    EigenMatrix<ArrowType> cov(const Buffer_ptr& bitmap, const Args&... args) const {
        auto c = indices_to_columns(args...);
        return dataset::cov<ArrowType>(bitmap, c.begin(), c.end());
    }

    ///////////////////////////// sse<ArrowType> /////////////////////////

    template <typename ArrowType, bool contains_null>
    EigenMatrix<ArrowType> sse() {
        auto c = derived().columns();
        return dataset::sse<ArrowType, contains_null>(c.begin(), c.end());
    }

    template <typename ArrowType>
    EigenMatrix<ArrowType> sse() {
        if (null_count() == 0) {
            return sse<ArrowType, false>();
        } else {
            return sse<ArrowType, true>();
        }
    }
    template <typename ArrowType>
    EigenMatrix<ArrowType> sse(const Buffer_ptr& bitmap) {
        auto c = derived().columns();
        return sse<ArrowType>(bitmap, c.begin(), c.end());
    }

    template <typename ArrowType, bool contains_null, typename Index, enable_if_index_t<Index, int> = 0>
    EigenMatrix<ArrowType> sse(const Index& index) const {
        return dataset::sse<ArrowType, contains_null>(derived().col(index));
    }
    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    EigenMatrix<ArrowType> sse(const Index& index) const {
        if (null_count(index) == 0) {
            return sse<ArrowType, false>(index);
        } else {
            return sse<ArrowType, true>(index);
        }
    }
    template <typename ArrowType, typename Index, enable_if_index_t<Index, int> = 0>
    EigenMatrix<ArrowType> sse(const Buffer_ptr& bitmap, const Index& index) const {
        return dataset::sse<ArrowType>(bitmap, derived().col(index));
    }

    template <typename ArrowType, bool contains_null, typename T, enable_if_index_container_t<T, int> = 0>
    EigenMatrix<ArrowType> sse(const T& cols) {
        auto c = indices_to_columns(cols);
        return dataset::sse<ArrowType, contains_null>(c.begin(), c.end());
    }
    template <typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    EigenMatrix<ArrowType> sse(const T& cols) {
        if (null_count(cols) == 0) {
            return sse<ArrowType, false>(cols);
        } else {
            return sse<ArrowType, true>(cols);
        }
    }
    template <typename ArrowType, typename T, enable_if_index_container_t<T, int> = 0>
    EigenMatrix<ArrowType> sse(const Buffer_ptr& bitmap, const T& cols) {
        auto c = indices_to_columns(cols);
        return dataset::sse<ArrowType>(bitmap, c.begin(), c.end());
    }

    template <typename ArrowType, bool contains_null, typename V>
    EigenMatrix<ArrowType> sse(const std::initializer_list<V>& cols) {
        auto c = indices_to_columns(cols);
        return dataset::sse<ArrowType, contains_null, std::initializer_list<V>>(c.begin(), c.end());
    }
    template <typename ArrowType, typename V>
    EigenMatrix<ArrowType> sse(const std::initializer_list<V>& cols) {
        if (null_count(cols) == 0) {
            return sse<ArrowType, false, std::initializer_list<V>>(cols);
        } else {
            return sse<ArrowType, true, std::initializer_list<V>>(cols);
        }
    }
    template <typename ArrowType, typename V>
    EigenMatrix<ArrowType> sse(const Buffer_ptr& bitmap, const std::initializer_list<V>& cols) {
        auto c = indices_to_columns(cols);
        return dataset::sse<ArrowType>(bitmap, c.begin(), c.end());
    }

    template <typename ArrowType,
              bool contains_null,
              typename IndexIter,
              enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> sse(const IndexIter& begin, const IndexIter& end) const {
        auto c = indices_to_columns(begin, end);
        return dataset::sse<ArrowType, contains_null>(c.begin(), c.end());
    }
    template <typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> sse(const IndexIter& begin, const IndexIter& end) const {
        if (null_count(begin, end) == 0) {
            return sse<ArrowType, false>(begin, end);
        } else {
            return sse<ArrowType, true>(begin, end);
        }
    }
    template <typename ArrowType, typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    EigenMatrix<ArrowType> sse(const Buffer_ptr& bitmap, const IndexIter& begin, const IndexIter& end) const {
        auto c = indices_to_columns(begin, end);
        return dataset::sse<ArrowType>(bitmap, c.begin(), c.end());
    }

    template <typename ArrowType,
              bool contains_null,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    EigenMatrix<ArrowType> sse(const Args&... args) const {
        auto c = indices_to_columns(args...);
        return dataset::sse<ArrowType, contains_null>(c.begin(), c.end());
    }
    template <typename ArrowType,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    EigenMatrix<ArrowType> sse(const Args&... args) const {
        if (null_count(args...) == 0) {
            return sse<ArrowType, false>(args...);
        } else {
            return sse<ArrowType, true>(args...);
        }
    }
    template <typename ArrowType,
              typename... Args,
              typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
    EigenMatrix<ArrowType> sse(const Buffer_ptr& bitmap, const Args&... args) const {
        auto c = indices_to_columns(args...);
        return dataset::sse<ArrowType>(bitmap, c.begin(), c.end());
    }

    ///////////////////////////// indices_to_columns /////////////////////////
    Array_vector indices_to_columns() const;
    template <typename T, enable_if_index_container_t<T, int> = 0>
    Array_vector indices_to_columns(const T& cols) const {
        return indices_to_columns(cols.begin(), cols.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    Array_vector indices_to_columns(const IndexIter& begin, const IndexIter& end) const;
    // https://stackoverflow.com/questions/39659127/restrict-variadic-template-arguments

#ifdef _MSC_VER
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
#else
    template <typename... Args, std::enable_if_t<(... && !util::is_iterator_v<Args>), int> = 0>
#endif
    Array_vector indices_to_columns(const Args&... args) const;

    ///////////////////////////// indices_to_schema /////////////////////////
    std::shared_ptr<arrow::Schema> indices_to_schema() const;
    template <typename T, enable_if_index_container_t<T, int> = 0>
    std::shared_ptr<arrow::Schema> indices_to_schema(const T& cols) const {
        return indices_to_schema(cols.begin(), cols.end());
    }
    template <typename IndexIter, enable_if_index_iterator_t<IndexIter, int> = 0>
    std::shared_ptr<arrow::Schema> indices_to_schema(const IndexIter& begin, const IndexIter& end) const;

#ifdef _MSC_VER
    template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
#else
    template <typename... Args, std::enable_if_t<(... && !util::is_iterator_v<Args>), int> = 0>
#endif
    std::shared_ptr<arrow::Schema> indices_to_schema(const Args&... args) const;
};

template <typename Derived>
// template<typename IndexIter, typename DataFrameBase<Derived>::template enable_if_index_iterator_t<IndexIter, int>>
template <typename IndexIter, typename dataframe_traits<Derived>::template enable_if_index_iterator_t<IndexIter, int>>
bool DataFrameBase<Derived>::has_columns(const IndexIter& begin, const IndexIter& end) const {
    for (auto it = begin; it != end; ++it) {
        if (!derived().has_column(*it)) return false;
    }

    return true;
}

template <typename Derived>
template <typename IndexIter, typename dataframe_traits<Derived>::template enable_if_index_iterator_t<IndexIter, int>>
void DataFrameBase<Derived>::raise_has_columns(const IndexIter& begin, const IndexIter& end) const {
    for (auto it = begin; it != end; ++it) {
        derived().raise_has_column(*it);
    }
}

template <typename Derived>
#ifdef _MSC_VER
template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
#else
template <typename... Args, std::enable_if_t<(... && !util::is_iterator_v<Args>), int>>
#endif
bool DataFrameBase<Derived>::has_columns(const Args&... args) const {
    return (has_columns(args) && ...);
}

template <typename Derived>
#ifdef _MSC_VER
template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
#else
template <typename... Args, std::enable_if_t<(... && !util::is_iterator_v<Args>), int>>
#endif
void DataFrameBase<Derived>::raise_has_columns(const Args&... args) const {
    return (raise_has_columns(args), ...);
}

template <typename Derived>
Array_vector DataFrameBase<Derived>::indices_to_columns() const {
    Array_vector v;
    v.reserve(derived().num_columns());

    for (const auto& col : derived().columns()) {
        v.push_back(col);
    }

    return v;
}

template <typename Derived>
template <typename IndexIter, typename dataframe_traits<Derived>::template enable_if_index_iterator_t<IndexIter, int>>
Array_vector DataFrameBase<Derived>::indices_to_columns(const IndexIter& begin, const IndexIter& end) const {
    Array_vector cols;
    cols.reserve(std::distance(begin, end));

    AppendColumns<DataFrameBase<Derived>> func{/*.self = */ *this};
    append_generic(cols, func, begin, end);

    return cols;
}

template <typename Derived>
#ifdef _MSC_VER
template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
#else
template <typename... Args, std::enable_if_t<(... && !util::is_iterator_v<Args>), int>>
#endif
Array_vector DataFrameBase<Derived>::indices_to_columns(const Args&... args) const {
    int total_size = (size_argument(derived().num_columns(), args) + ...);

    Array_vector cols;
    cols.reserve(total_size);

    AppendColumns<DataFrameBase<Derived>> func{/*.self = */ *this};
    append_generic(cols, func, args...);

    return cols;
}

template <typename Derived>
std::shared_ptr<arrow::Schema> DataFrameBase<Derived>::indices_to_schema() const {
    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_APPEND);
    auto schema = derived().schema();

    for (auto i = 0; i < schema->num_fields(); ++i) {
        RAISE_STATUS_ERROR(b.AddField(schema->field(i)));
    }

    auto r = b.Finish();
    if (!r.ok()) {
        throw std::domain_error("Schema could not be created for selected columns.");
    }

    return std::move(r).ValueOrDie();
}

template <typename Derived>
template <typename IndexIter, typename dataframe_traits<Derived>::template enable_if_index_iterator_t<IndexIter, int>>
std::shared_ptr<arrow::Schema> DataFrameBase<Derived>::indices_to_schema(const IndexIter& begin,
                                                                         const IndexIter& end) const {
    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_APPEND);

    AppendSchema<DataFrameBase<Derived>> func{*this};
    append_generic(b, func, begin, end);

    auto r = b.Finish();
    if (!r.ok()) {
        throw std::domain_error("Schema could not be created for selected columns.");
    }

    return std::move(r).ValueOrDie();
}

template <typename Derived>
#ifdef _MSC_VER
template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
#else
template <typename... Args, std::enable_if_t<(... && !util::is_iterator_v<Args>), int>>
#endif
std::shared_ptr<arrow::Schema> DataFrameBase<Derived>::indices_to_schema(const Args&... args) const {
    arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_APPEND);

    AppendSchema<DataFrameBase<Derived>> func{/*.self = */ *this};
    append_generic(b, func, args...);

    auto r = b.Finish();
    if (!r.ok()) {
        throw std::domain_error("Schema could not be created for selected columns.");
    }

    return std::move(r).ValueOrDie();
}

template <typename Derived>
template <typename IndexIter, typename dataframe_traits<Derived>::template enable_if_index_iterator_t<IndexIter, int>>
typename DataFrameBase<Derived>::loc_return DataFrameBase<Derived>::loc(const IndexIter& begin,
                                                                        const IndexIter& end) const {
    auto columns = indices_to_columns(begin, end);
    auto schema = indices_to_schema(begin, end);
    return DataFrame(RecordBatch::Make(schema, derived().num_rows(), columns));
}

template <typename Derived>
template <typename Index, typename dataframe_traits<Derived>::template enable_if_index_t<Index, int>>
typename DataFrameBase<Derived>::loc_return DataFrameBase<Derived>::loc(const Index& index) const {
    arrow::SchemaBuilder b;
    RAISE_STATUS_ERROR(b.AddField(derived().field(index)));

    auto r = b.Finish();
    if (!r.ok()) {
        throw std::domain_error("Schema could not be created for column index " + index_to_string(index));
    }

    Array_vector c = {derived().col(index)};
    return RecordBatch::Make(std::move(r).ValueOrDie(), derived().num_rows(), c);
}

template <typename Derived>
#ifdef _MSC_VER
template <typename... Args, typename = std::enable_if_t<(... && !util::is_iterator_v<Args>), void>>
#else
template <typename... Args, std::enable_if_t<(... && !util::is_iterator_v<Args>), int>>
#endif
typename DataFrameBase<Derived>::loc_return DataFrameBase<Derived>::loc(const Args&... args) const {
    auto columns = indices_to_columns(args...);
    auto schema = indices_to_schema(args...);
    return DataFrame(RecordBatch::Make(schema, derived().num_rows(), columns));
}

class DataFrame : public DataFrameBase<DataFrame> {
public:
    DataFrame() : m_batch(arrow::RecordBatch::Make(arrow::schema({}), 0, Array_vector())) {}
    DataFrame(int64_t num_rows) : m_batch(arrow::RecordBatch::Make(arrow::schema({}), num_rows, Array_vector())) {}

    DataFrame(std::shared_ptr<RecordBatch> rb) : m_batch(rb) {}

    const std::shared_ptr<RecordBatch>& record_batch() const { return m_batch; }

    DataFrame slice(int64_t offset) const { return DataFrame(m_batch->Slice(offset)); }

    DataFrame slice(int64_t offset, int64_t length) const { return DataFrame(m_batch->Slice(offset, length)); }

    int num_rows() const { return m_batch->num_rows(); }

    int num_columns() const { return m_batch->num_columns(); }

    int num_variables() const { return m_batch->num_columns(); }

    bool has_column(int index) const {
        if (index < 0 || index >= m_batch->num_columns()) {
            return false;
        }

        return true;
    }

    void raise_has_column(int index) const {
        if (!has_column(index)) {
            throw std::domain_error("Index " + std::to_string(index) + " do no exist for DataFrame with " +
                                    std::to_string(m_batch->num_columns()) + " columns.");
        }
    }

    template <typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    bool has_column(const StringType& name) const {
        auto a = m_batch->GetColumnByName(name);
        if (!a) {
            return false;
        }

        return true;
    }

    template <typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    void raise_has_column(const StringType& name) const {
        if (!has_column(name)) {
            throw std::domain_error("Column \"" + name + "\" not found in DataFrame");
        }
    }

    Field_ptr field(int i) const { return m_batch->schema()->field(i); }

    template <typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    Field_ptr field(const StringType& name) const {
        return m_batch->schema()->GetFieldByName(name);
    }

    int index(int i) const { return i; }

    template <typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    int index(const StringType& name) const {
        return m_batch->schema()->GetFieldIndex(name);
    }

    const std::string& name(int i) const { return m_batch->column_name(i); }
    template <typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    const std::string& name(const StringType& n) const {
        return n;
    }

    std::vector<std::string> column_names() const;

    Array_ptr col(int i) const {
        if (i >= 0 && i < m_batch->num_columns())
            return m_batch->column(i);
        else
            throw std::invalid_argument("Column index " + std::to_string(i) + " do not exist in DataFrame.");
    }

    template <typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    Array_ptr col(const StringType& name) const {
        auto a = m_batch->GetColumnByName(name);
        if (a)
            return a;
        else
            throw std::invalid_argument("Column index " + name + " do not exist in DataFrame.");
    }

    Array_vector columns() const { return m_batch->columns(); }

    std::vector<int> discrete_columns() const;
    std::vector<int> continuous_columns() const;

    DataFrame normalize() const;

    DataFrame filter_null() const {
        if (null_count() == 0) {
            return *this;
        } else {
            auto bitmap = combined_bitmap();
            auto bitmap_data = bitmap->data();

            int total_rows = num_rows();
            int valid_rows = util::bit_util::non_null_count(bitmap, total_rows);

            arrow::AdaptiveIntBuilder builder;
            RAISE_STATUS_ERROR(builder.Reserve(valid_rows));

            for (auto i = 0; i < total_rows; ++i) {
                if (util::bit_util::GetBit(bitmap_data, i)) RAISE_STATUS_ERROR(builder.Append(i));
            }

            Array_ptr take_ind;
            RAISE_STATUS_ERROR(builder.Finish(&take_ind));
            return take(take_ind);
        }
    }

    DataFrame take(const Array_ptr& indices) const {
        auto rfiltered = arrow::compute::Take(m_batch, indices, arrow::compute::TakeOptions::NoBoundsCheck());
        return DataFrame(std::move(rfiltered).ValueOrDie().record_batch());
    }

    std::shared_ptr<RecordBatch> operator->() const { return m_batch; }
    friend std::pair<DataFrame, DataFrame> generate_cv_pair(const DataFrame& df,
                                                            int fold,
                                                            const std::vector<int>& indices,
                                                            const std::vector<std::vector<int>::iterator>& test_limits);

private:
    std::shared_ptr<RecordBatch> m_batch;
};
}  // namespace dataset

namespace pybind11::detail {
template <>
struct type_caster<dataset::DataFrame> {
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
        if (dataset::is_pyarrow_instance(src, "RecordBatch")) {
            dataset::ArrowCAPIObjects capi = dataset::extract_pycapsule_array(src);
            RAISE_RESULT_ERROR(value, arrow::ImportRecordBatch(capi.arrow_array, capi.arrow_schema))
            return true;
        } else if (dataset::is_pandas_dataframe(src)) {
            auto a = dataset::pandas_to_pyarrow_record_batch(src);
            dataset::ArrowCAPIObjects capi = dataset::extract_pycapsule_array(a);
            RAISE_RESULT_ERROR(value, arrow::ImportRecordBatch(capi.arrow_array, capi.arrow_schema))
            return true;
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
        struct ArrowSchema* c_schema = (struct ArrowSchema*)malloc(sizeof(struct ArrowSchema));
        struct ArrowArray* c_array = (struct ArrowArray*)malloc(sizeof(struct ArrowArray));
        RAISE_STATUS_ERROR(arrow::ExportRecordBatch(*src.record_batch(), c_array, c_schema));

        PyObject* schema_capsule = PyCapsule_New(c_schema, "arrow_schema", dataset::ReleaseArrowSchemaPyCapsule);
        PyObject* array_capsule = PyCapsule_New(c_array, "arrow_array", dataset::ReleaseArrowArrayPyCapsule);
        PyObject* args = PyTuple_Pack(2, schema_capsule, array_capsule);

        py::handle method = py::module::import("pyarrow").attr("RecordBatch").attr("_import_from_c_capsule");

        PyObject* method_py = method.ptr();
        py::handle casted = PyObject_Call(method_py, args, NULL);

        Py_DECREF(args);
        return casted;
    }
};
}  // namespace pybind11::detail

namespace pybind11::detail {
template <>
struct type_caster<Array_ptr> {
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
        if (dataset::is_pyarrow_instance(src, "Array")) {
            dataset::ArrowCAPIObjects capi = dataset::extract_pycapsule_array(src);
            RAISE_RESULT_ERROR(value, arrow::ImportArray(capi.arrow_array, capi.arrow_schema))
            return true;
        } else if (dataset::is_pandas_series(src)) {
            auto a = dataset::pandas_to_pyarrow_array(src);
            dataset::ArrowCAPIObjects capi = dataset::extract_pycapsule_array(a);
            RAISE_RESULT_ERROR(value, arrow::ImportArray(capi.arrow_array, capi.arrow_schema))
            return true;
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
        struct ArrowSchema* c_schema = (struct ArrowSchema*)malloc(sizeof(struct ArrowSchema));
        struct ArrowArray* c_array = (struct ArrowArray*)malloc(sizeof(struct ArrowArray));
        RAISE_STATUS_ERROR(arrow::ExportArray(*src, c_array, c_schema));

        PyObject* schema_capsule = PyCapsule_New(c_schema, "arrow_schema", dataset::ReleaseArrowSchemaPyCapsule);
        PyObject* array_capsule = PyCapsule_New(c_array, "arrow_array", dataset::ReleaseArrowArrayPyCapsule);
        PyObject* args = PyTuple_Pack(2, schema_capsule, array_capsule);

        py::handle method = py::module::import("pyarrow").attr("Array").attr("_import_from_c_capsule");

        PyObject* method_py = method.ptr();
        py::handle casted = PyObject_Call(method_py, args, NULL);

        Py_DECREF(args);
        return casted;
    }
};
}  // namespace pybind11::detail

namespace pybind11::detail {
template <>
struct type_caster<std::shared_ptr<arrow::DataType>> {
public:
    /**
     * This macro establishes the name 'inty' in
     * function signatures and declares a local variable
     * 'value' of type inty
     */
    PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::DataType>, _("pyarrow.DataType"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a inty
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool) {
        if (dataset::is_pyarrow_instance(src, "DataType")) {
            struct ArrowSchema* capi = dataset::extract_pycapsule_schema(src);
            RAISE_RESULT_ERROR(value, arrow::ImportType(capi))
            return true;
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
    static handle cast(std::shared_ptr<arrow::DataType> src, return_value_policy /* policy */, handle /* parent */) {
        struct ArrowSchema* c_schema = (struct ArrowSchema*)malloc(sizeof(struct ArrowSchema));
        RAISE_STATUS_ERROR(arrow::ExportType(*src, c_schema));

        PyObject* schema_capsule = PyCapsule_New(c_schema, "arrow_schema", dataset::ReleaseArrowSchemaPyCapsule);

        py::handle method = py::module::import("pyarrow").attr("DataType").attr("_import_from_c_capsule");

        PyObject* method_py = method.ptr();

        #ifdef Python_MAJOR_VERSION == 3 && Python_MINOR_VERSION >= 9
        py::handle casted = PyObject_CallOneArg(method_py, schema_capsule);
        #else
        PyObject* args = PyTuple_Pack(1, schema_capsule);
        py::handle casted = PyObject_Call(method_py, args, NULL);
        Py_DECREF(args);
        #endif

        return casted;
    }
};

}  // namespace pybind11::detail

#endif  // PYBNESIAN_DATASET_DATASET_HPP
