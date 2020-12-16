#ifndef PYBNESIAN_DATASET_DYNAMIC_DATASET_HPP
#define PYBNESIAN_DATASET_DYNAMIC_DATASET_HPP

#include <dataset/dataset.hpp>

using Field_ptr = std::shared_ptr<arrow::Field>;
using Field_vector = std::vector<Field_ptr>;
using Array_ptr = std::shared_ptr<arrow::Array>;
using Array_vector =  std::vector<Array_ptr>;

namespace dataset {

    DataFrame create_temporal_slice(const DataFrame& df, int slice_index, int markovian_order);
    std::vector<DataFrame> create_temporal_slices(const DataFrame& df, int markovian_order);
    DataFrame create_joint_df(std::vector<DataFrame>& v, int markovian_order);

    void append_slice(const DataFrame& slice, Array_vector& columns, Field_vector& fields);

    template<typename Index, util::enable_if_index_t<Index, int> = 0>
    using DynamicVariable = std::pair<Index, int>;

    class DynamicDataFrame {
    public:
        DynamicDataFrame(const DataFrame& df, int markovian_order) : m_origin(df),
                                                                     m_markovian_order(markovian_order) {
            if (markovian_order < 1) {
                throw std::invalid_argument("Markovian order must be at least 1.");
            }

            m_temporal_slices = create_temporal_slices(m_origin, m_markovian_order);
            m_joint = create_joint_df(m_temporal_slices, m_markovian_order);
        }

        DataFrame temporal_slice(int slice_index) const {
            if (slice_index < 0 || slice_index > m_markovian_order) {
                throw std::invalid_argument("slice_index must be an index between 0 and " + 
                                            std::to_string(m_markovian_order));
            }

            return m_temporal_slices[slice_index];
        }

        template<typename T, util::enable_if_integral_container_t<T, int> = 0>
        DataFrame temporal_slice(const T& slices) const {
            return temporal_slice(slices.begin(), slices.end());
        }

        DataFrame temporal_slice(const std::initializer_list<int>& slices) const { 
            return temporal_slice(slices.begin(), slices.end()); 
        }

        template<typename Iter, util::enable_if_integral_iterator_t<Iter, int> = 0>
        DataFrame temporal_slice(const Iter& begin, const Iter& end) const;

        template<typename ...Args>
        DataFrame temporal_slice(const Args&... args) const;

        template<typename Index>
        DataFrame loc(const DynamicVariable<Index>& v) const;

        template<typename T, util::enable_if_dynamic_index_container_t<T, int> = 0>
        DataFrame loc(const T& v) const;

        const DataFrame& joint() const {
            return m_joint;
        }

    private:
        DataFrame m_origin;
        std::vector<DataFrame> m_temporal_slices;
        DataFrame m_joint;
        int m_markovian_order;
    };

    void append_slice(const std::vector<DataFrame>& slices,
                      Array_vector& columns,
                      Field_vector& fields,
                      int markovian_order,
                      int slice_index);

    template<typename Iter, util::enable_if_integral_iterator_t<Iter, int> = 0>
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

    template<typename Iter, util::enable_if_integral_iterator_t<Iter, int> = 0>
    void append_slice(const std::vector<DataFrame>& slices,
                      Array_vector& columns,
                      Field_vector& fields,
                      int markovian_order,
                      const std::pair<Iter, Iter>& slice_indices) {
        append_slice(slices, columns, fields, markovian_order, slice_indices.first, slice_indices.second);
    }

    template<typename T, util::enable_if_integral_container_t<T, int> = 0>
    inline void append_slice(const std::vector<DataFrame>& slices,
                              Array_vector& columns,
                              Field_vector& fields,
                              int markovian_order,
                              const T& slice_indices) { 
        append_slice(slices, columns, fields, markovian_order, slice_indices.begin(), slice_indices.end());
    }

    template<typename Iter, util::enable_if_integral_iterator_t<Iter, int>>
    DataFrame DynamicDataFrame::temporal_slice(const Iter& begin, const Iter& end) const {
        auto num_slices = std::distance(begin, end);
        Array_vector columns;
        columns.reserve(num_slices * m_temporal_slices[0]->num_columns());
        Field_vector fields;
        fields.reserve(num_slices * m_temporal_slices[0]->num_columns());

        for (auto it = begin; it != end; ++it) {
            if (*it < 0 || *it > m_markovian_order) {
                throw std::invalid_argument("slice_index must be an index between 0 and " + 
                                        std::to_string(m_markovian_order));
            }

            append_slice(m_temporal_slices[*it], columns, fields);
        }

        auto schema = arrow::schema(fields);

        return DataFrame(arrow::RecordBatch::Make(schema, m_temporal_slices[0]->num_rows(), columns));
    }

    template<typename ...Args>
    DataFrame DynamicDataFrame::temporal_slice(const Args&... args) const {
        auto slice_num_columns = m_temporal_slices[0]->num_columns();
        auto num_columns = (size_argument(m_temporal_slices[0], args) + ...)*slice_num_columns;

        Array_vector columns;
        columns.reserve(num_columns);
        Field_vector fields;
        fields.reserve(num_columns);

        (append_slice(m_temporal_slices, columns, fields, m_markovian_order, args),...);

        auto schema = arrow::schema(fields);

        return DataFrame(arrow::RecordBatch::Make(schema, m_temporal_slices[0]->num_rows(), columns));
    }
}

#endif //PYBNESIAN_DATASET_DYNAMIC_DATASET_HPP