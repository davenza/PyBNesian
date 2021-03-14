#ifndef PYBNESIAN_DATASET_DYNAMIC_DATASET_HPP
#define PYBNESIAN_DATASET_DYNAMIC_DATASET_HPP

#include <dataset/dataset.hpp>
#include <util/util_types.hpp>
#include <util/temporal.hpp>

using util::ArcStringVector;

using Field_ptr = std::shared_ptr<arrow::Field>;
using Field_vector = std::vector<Field_ptr>;
using Array_ptr = std::shared_ptr<arrow::Array>;
using Array_vector = std::vector<Array_ptr>;

namespace dataset {

DataFrame create_temporal_slice(const DataFrame& df, int slice_index, int slice_offset, int markovian_order);
std::vector<DataFrame> create_temporal_slices(const DataFrame& df, int markovian_order);
DataFrame create_static_df(const DataFrame& df, int markovian_order);
DataFrame create_transition_df(std::vector<DataFrame>& v, int markovian_order);

void append_slice(const DataFrame& slice, Array_vector& columns, Field_vector& fields);

template <typename Index, typename>
struct DynamicVariable {
    using variable_type = Index;

    DynamicVariable(Index v, int s) : variable(v), temporal_slice(s) {}
    DynamicVariable(std::pair<Index, int> t) : DynamicVariable(t.first, t.second) {}

    template <typename T = Index, util::enable_if_stringable_t<T, int> = 0>
    std::string temporal_name() const {
        return util::temporal_name(variable, temporal_slice);
    }

    Index variable;
    int temporal_slice;
};

class DynamicDataFrame;
template <>
struct dataframe_traits<DynamicDataFrame> {
    template <typename T, typename R>
    using enable_if_index_t = util::enable_if_dynamic_index_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_container_t = util::enable_if_dynamic_index_container_t<T, R>;
    template <typename T, typename R>
    using enable_if_index_iterator_t = util::enable_if_dynamic_index_iterator_t<T, R>;
    using loc_return = DataFrame;
};

class DynamicDataFrame : public DataFrameBase<DynamicDataFrame> {
public:
    DynamicDataFrame(DataFrame df, int markovian_order) : m_origin(df), m_markovian_order(markovian_order) {
        if (markovian_order < 1) {
            throw std::invalid_argument("Markovian order must be at least 1.");
        }

        m_temporal_slices = create_temporal_slices(m_origin, m_markovian_order);
        m_static = create_static_df(m_origin, m_markovian_order);
        m_transition = create_transition_df(m_temporal_slices, m_markovian_order);
    }

    int markovian_order() const { return m_markovian_order; }

    int num_rows() const { return m_transition->num_rows(); }

    int num_columns() const { return m_transition->num_columns(); }

    int num_variables() const { return m_origin->num_columns(); }

    template <typename Index, enable_if_index_t<Index, int> = 0>
    void check_temporal_slice(const Index& index) const {
        if (index.temporal_slice < 0 || index.temporal_slice > m_markovian_order) {
            throw std::invalid_argument("slice_index must be an index between 0 and " +
                                        std::to_string(m_markovian_order));
        }
    }

    template <typename Index, enable_if_index_t<Index, int> = 0>
    bool has_column(const Index& index) const {
        check_temporal_slice(index);

        if constexpr (util::is_stringable_v<typename Index::variable_type>) {
            return m_temporal_slices[index.temporal_slice].has_column(index.temporal_name());
        } else {
            return m_temporal_slices[index.temporal_slice].has_column(index.variable);
        }
    }

    template <typename Index, enable_if_index_t<Index, int> = 0>
    void raise_has_column(const Index& index) const {
        check_temporal_slice(index);

        if constexpr (util::is_stringable_v<typename Index::variable_type>) {
            m_temporal_slices[index.temporal_slice].raise_has_column(index.temporal_name());
        } else {
            m_temporal_slices[index.temporal_slice].raise_has_column(index.variable);
        }
    }

    template <typename Index, enable_if_index_t<Index, int> = 0>
    Field_ptr field(const Index& index) const {
        check_temporal_slice(index);
        if constexpr (util::is_stringable_v<typename Index::variable_type>) {
            return m_temporal_slices[index.temporal_slice].field(index.temporal_name());
        } else {
            return m_temporal_slices[index.temporal_slice].field(index.variable);
        }
    }

    template <typename Index, enable_if_index_t<Index, int> = 0>
    Array_ptr col(const Index& index) const {
        check_temporal_slice(index);
        if constexpr (util::is_stringable_v<typename Index::variable_type>) {
            return m_temporal_slices[index.temporal_slice].col(index.temporal_name());
        } else {
            return m_temporal_slices[index.temporal_slice].col(index.variable);
        }
    }

    DataFrame temporal_slice(int slice_index) const {
        if (slice_index < 0 || slice_index > m_markovian_order) {
            throw std::invalid_argument("slice_index must be an index between 0 and " +
                                        std::to_string(m_markovian_order));
        }

        return m_temporal_slices[slice_index];
    }

    template <typename T, util::enable_if_integral_container_t<T, int> = 0>
    DataFrame temporal_slice(const T& slices) const {
        return temporal_slice(slices.begin(), slices.end());
    }

    DataFrame temporal_slice(const std::initializer_list<int>& slices) const {
        return temporal_slice(slices.begin(), slices.end());
    }

    template <typename Iter, util::enable_if_integral_iterator_t<Iter, int> = 0>
    DataFrame temporal_slice(const Iter& begin, const Iter& end) const;
    template <typename... Args>
    DataFrame temporal_slice(const Args&... args) const;

    const DataFrame& origin_df() const { return m_origin; }

    const DataFrame& static_df() const { return m_static; }

    const DataFrame& transition_df() const { return m_transition; }

    std::shared_ptr<RecordBatch> operator->() const { return m_transition.record_batch(); }

private:
    DataFrame m_origin;
    std::vector<DataFrame> m_temporal_slices;
    DataFrame m_static;
    DataFrame m_transition;
    int m_markovian_order;
};

void append_slice(const std::vector<DataFrame>& slices,
                  Array_vector& columns,
                  Field_vector& fields,
                  int markovian_order,
                  int slice_index);

template <typename Iter, util::enable_if_integral_iterator_t<Iter, int> = 0>
void append_slice(const std::vector<DataFrame>& slices,
                  Array_vector& columns,
                  Field_vector& fields,
                  int markovian_order,
                  const Iter& slice_begin,
                  const Iter& slice_end) {
    for (auto it = slice_begin; it != slice_end; ++it) {
        append_slice(slices, columns, fields, markovian_order, *it);
    }
}

template <typename Iter, util::enable_if_integral_iterator_t<Iter, int> = 0>
void append_slice(const std::vector<DataFrame>& slices,
                  Array_vector& columns,
                  Field_vector& fields,
                  int markovian_order,
                  const std::pair<Iter, Iter>& slice_indices) {
    append_slice(slices, columns, fields, markovian_order, slice_indices.first, slice_indices.second);
}

template <typename T, util::enable_if_integral_container_t<T, int> = 0>
inline void append_slice(const std::vector<DataFrame>& slices,
                         Array_vector& columns,
                         Field_vector& fields,
                         int markovian_order,
                         const T& slice_indices) {
    append_slice(slices, columns, fields, markovian_order, slice_indices.begin(), slice_indices.end());
}

template <typename Iter, util::enable_if_integral_iterator_t<Iter, int>>
DataFrame DynamicDataFrame::temporal_slice(const Iter& begin, const Iter& end) const {
    auto num_slices = std::distance(begin, end);
    Array_vector columns;
    columns.reserve(num_slices * num_columns());
    Field_vector fields;
    fields.reserve(num_slices * num_columns());

    for (auto it = begin; it != end; ++it) {
        if (*it < 0 || *it > m_markovian_order) {
            throw std::invalid_argument("slice_index must be an index between 0 and " +
                                        std::to_string(m_markovian_order));
        }

        append_slice(m_temporal_slices[*it], columns, fields);
    }

    auto schema = arrow::schema(fields);

    return DataFrame(arrow::RecordBatch::Make(schema, num_rows(), columns));
}

template <typename... Args>
DataFrame DynamicDataFrame::temporal_slice(const Args&... args) const {
    auto total_cols = (size_argument(m_markovian_order + 1, args) + ...) * num_columns();

    Array_vector columns;
    columns.reserve(total_cols);
    Field_vector fields;
    fields.reserve(total_cols);

    (append_slice(m_temporal_slices, columns, fields, m_markovian_order, args), ...);

    auto schema = arrow::schema(fields);

    return DataFrame(arrow::RecordBatch::Make(schema, num_rows(), columns));
}

template <typename T>
class DynamicAdaptator {
public:
    template <typename... Args>
    DynamicAdaptator(DynamicDataFrame df, const Args&... args)
        : m_df(df), m_static(m_df.static_df(), args...), m_transition(m_df.transition_df(), args...) {}

    const DynamicDataFrame& dataframe() const { return m_df; }
    DynamicDataFrame& dataframe() { return m_df; }
    const T& static_element() const { return m_static; }
    T& static_element() { return m_static; }
    const T& transition_element() const { return m_transition; }
    T& transition_element() { return m_transition; }

    std::vector<std::string> variable_names() const { return m_df.origin_df().column_names(); }

    const std::string& name(int i) const { return m_df.origin_df().name(i); }

    bool has_variables(const std::string& name) const { return m_df.origin_df().has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const { return m_df.origin_df().has_columns(cols); }

    int num_variables() const { return m_df.num_variables(); }

    int markovian_order() const { return m_df.markovian_order(); }

private:
    DynamicDataFrame m_df;
    T m_static;
    T m_transition;
};

}  // namespace dataset

#endif  // PYBNESIAN_DATASET_DYNAMIC_DATASET_HPP