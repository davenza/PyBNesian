#ifndef PYBNESIAN_DATASET_DYNAMIC_DATASET_HPP
#define PYBNESIAN_DATASET_DYNAMIC_DATASET_HPP

#include <dataset/dataset.hpp>

using Field_ptr = std::shared_ptr<arrow::Field>;
using Field_vector = std::vector<Field_ptr>;
using Array_ptr = std::shared_ptr<arrow::Array>;
using Array_vector =  std::vector<Array_ptr>;

namespace dataset {

    std::string transform_temporal_name(const std::string& name, int slice_index);
    DataFrame create_temporal_slice(const DataFrame& df, int slice_index, int markovian_order);
    std::vector<DataFrame> create_temporal_slices(const DataFrame& df, int markovian_order);
    DataFrame create_joint_df(std::vector<DataFrame>& v, int markovian_order);

    void append_slice(const DataFrame& slice, Array_vector& columns, Field_vector& fields);

    template<typename Index, typename>
    struct DynamicVariable {
        using variable_type = Index;
        
        DynamicVariable(Index v, int s) : variable(v), temporal_slice(s) {}
        DynamicVariable(std::pair<Index, int> t) : DynamicVariable(t.first, t.second) {}

        Index variable;
        int temporal_slice;
    };

    class DynamicDataFrame : public DataFrameBase<DynamicDataFrame> {
    public:
        DynamicDataFrame(const DataFrame& df, int markovian_order) : m_origin(df),
                                                                     m_markovian_order(markovian_order) {
            if (markovian_order < 1) {
                throw std::invalid_argument("Markovian order must be at least 1.");
            }

            m_temporal_slices = create_temporal_slices(m_origin, m_markovian_order);
            m_joint = create_joint_df(m_temporal_slices, m_markovian_order);
        }

        int markovian_order() const {
            return m_markovian_order;
        }

        int num_columns() const {
            return m_joint->num_columns();
        }

        int num_rows() const {
            return m_temporal_slices[0]->num_rows();
        }

        Field_ptr field(DynamicVariable<int> index) const {
            if (index.temporal_slice < 0 || index.temporal_slice > m_markovian_order) {
                throw std::invalid_argument("slice_index must be an index between 0 and " + 
                                            std::to_string(m_markovian_order));
            }

            return m_temporal_slices[index.temporal_slice]->schema()->field(index.variable);
        }

        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Field_ptr field(const DynamicVariable<StringType>& index) const {
            if (index.temporal_slice < 0 || index.temporal_slice > m_markovian_order) {
                throw std::invalid_argument("slice_index must be an index between 0 and " + 
                                            std::to_string(m_markovian_order));
            }

            return m_temporal_slices[index.temporal_slice]->schema()->GetFieldByName(index.variable);
        }

        Array_ptr col(DynamicVariable<int> index) const { 
            if (index.temporal_slice < 0 || index.temporal_slice > m_markovian_order) {
                throw std::invalid_argument("slice_index must be an index between 0 and " + 
                                            std::to_string(m_markovian_order));
            }

            if (index.variable >= 0 && index.variable < m_temporal_slices[index.temporal_slice]->num_columns())
                return m_temporal_slices[index.temporal_slice].col(index.variable);
            else
                throw std::invalid_argument("Column index " + index_to_string(index) + " do not exist in DataFrame.");
        }

        template<typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
        Array_ptr col(const DynamicVariable<StringType>& index) const {
            if (index.temporal_slice < 0 || index.temporal_slice > m_markovian_order) {
                throw std::invalid_argument("slice_index must be an index between 0 and " + 
                                            std::to_string(m_markovian_order));
            }

            auto a = m_temporal_slices[index.temporal_slice].col(index.variable);
            if (a)
                return a;
            else
                throw std::invalid_argument("Column index " + index_to_string(index) + " do not exist in DataFrame.");
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

        const DataFrame& joint() const {
            return m_joint;
        }

    private:
        DataFrame m_origin;
        std::vector<DataFrame> m_temporal_slices;
        DataFrame m_joint;
        int m_markovian_order;
    };

    // class DynamicDataFrame {
    // public:
    //     DynamicDataFrame(const DataFrame& df, int markovian_order) : m_origin(df),
    //                                                                  m_markovian_order(markovian_order) {
    //         if (markovian_order < 1) {
    //             throw std::invalid_argument("Markovian order must be at least 1.");
    //         }

    //         m_temporal_slices = create_temporal_slices(m_origin, m_markovian_order);
    //         m_joint = create_joint_df(m_temporal_slices, m_markovian_order);
    //     }

    //     int markovian_order() const {
    //         return m_markovian_order;
    //     }

    //     int num_columns() const {
    //         return m_temporal_slices[0]->num_columns();
    //     }

    //     int num_rows() const {
    //         return m_temporal_slices[0]->num_rows();
    //     }

    //     DataFrame temporal_slice(int slice_index) const {
    //         if (slice_index < 0 || slice_index > m_markovian_order) {
    //             throw std::invalid_argument("slice_index must be an index between 0 and " + 
    //                                         std::to_string(m_markovian_order));
    //         }

    //         return m_temporal_slices[slice_index];
    //     }

    //     template<typename T, util::enable_if_integral_container_t<T, int> = 0>
    //     DataFrame temporal_slice(const T& slices) const {
    //         return temporal_slice(slices.begin(), slices.end());
    //     }

    //     DataFrame temporal_slice(const std::initializer_list<int>& slices) const { 
    //         return temporal_slice(slices.begin(), slices.end()); 
    //     }

    //     template<typename Iter, util::enable_if_integral_iterator_t<Iter, int> = 0>
    //     DataFrame temporal_slice(const Iter& begin, const Iter& end) const;
    //     template<typename ...Args>
    //     DataFrame temporal_slice(const Args&... args) const;

    //     template<typename Index>
    //     DataFrame loc(const DynamicVariable<Index>& v) const;
    //     template<typename T, util::enable_if_dynamic_index_container_t<T, int> = 0>
    //     DataFrame loc(const T& cols) const { return loc(cols.begin(), cols.end()); }
    //     template<typename Index>
    //     DataFrame loc(const std::initializer_list<DynamicVariable<Index>>& cols) { return loc(cols.begin(), cols.end()); }
    //     template<typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int> = 0>
    //     DataFrame loc(const IndexIter& begin, const IndexIter& end) const;
    //     template<typename ...Args>
    //     DataFrame loc(const Args&... args) const;

    //     const DataFrame& joint() const {
    //         return m_joint;
    //     }

    // private:
    //     Array_vector indices_to_columns() const {
    //         return m_joint->columns();
    //     }
    //     template<typename T, util::enable_if_index_container_t<T, int> = 0>
    //     Array_vector indices_to_columns(const T& cols) const {
    //         return indices_to_columns(cols.begin(), cols.end());
    //     }

    //     template<typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int> = 0>
    //     Array_vector indices_to_columns(const IndexIter& begin, const IndexIter& end) const;

    //     template<typename ...Args>
    //     Array_vector indices_to_columns(const Args&... args) const;

    //     DataFrame m_origin;
    //     std::vector<DataFrame> m_temporal_slices;
    //     DataFrame m_joint;
    //     int m_markovian_order;
    // };

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

    template<typename ...Args>
    DataFrame DynamicDataFrame::temporal_slice(const Args&... args) const {
        auto total_cols = (size_argument(m_markovian_order+1, args) + ...)*num_columns();

        Array_vector columns;
        columns.reserve(total_cols);
        Field_vector fields;
        fields.reserve(total_cols);

        (append_slice(m_temporal_slices, columns, fields, m_markovian_order, args),...);

        auto schema = arrow::schema(fields);

        return DataFrame(arrow::RecordBatch::Make(schema, num_rows(), columns));
    }

    // template<typename Index>
    // DataFrame DynamicDataFrame::loc(const DynamicVariable<Index>& v) const {
    //     if (v.temporal_slice < 0 || v.temporal_slice > m_markovian_order) {
    //         throw std::invalid_argument("slice_index must be an index between 0 and " + 
    //                                     std::to_string(m_markovian_order));
    //     }

    //     if constexpr (std::is_integral_v<Index>) {
    //         return m_temporal_slices[v.temporal_slice].loc(v.variable);
    //     } else if constexpr(std::is_convertible_v<Index, const std::string&>) {
    //         return m_temporal_slices[v.temporal_slice]
    //                     .loc(transform_temporal_name(v.variable, v.temporal_slice));
    //     }
    // }

    // template<typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int>>
    // DataFrame DynamicDataFrame::loc(const IndexIter& begin, const IndexIter& end) const {
    //     arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_APPEND);
    //     Array_vector new_cols;
    //     new_cols.reserve(std::distance(begin, end));

    //     for (auto it = begin; it != end; ++it) {
    //         if (it->temporal_slice < 0 || it->temporal_slice > m_markovian_order) {
    //             throw std::invalid_argument("slice_index must be an index between 0 and " + 
    //                                         std::to_string(m_markovian_order));
    //         }

    //         using Index = typename std::iterator_traits<IndexIter>::value_type::variable_type;

    //         if constexpr (std::is_convertible_v<Index, const std::string&>) {
    //             append_columns(m_temporal_slices[it->temporal_slice].record_batch(), new_cols, 
    //                             transform_temporal_name(it->variable, it->temporal_slice));
    //             append_schema(m_temporal_slices[it->temporal_slice].record_batch(), b, 
    //                             transform_temporal_name(it->variable, it->temporal_slice));
    //         } else {
    //             append_columns(m_temporal_slices[it->temporal_slice].record_batch(), new_cols, it->variable);
    //             append_schema(m_temporal_slices[it->temporal_slice].record_batch(), b, it->variable);
    //         }
    //     }

    //     auto r = b.Finish();
    //     if (!r.ok()) {
    //         throw std::domain_error("Schema could not be created for selected columns.");
    //     }
    //     return DataFrame(RecordBatch::Make(std::move(r).ValueOrDie(), num_rows(), new_cols));
    // }


    // template<typename Index>
    // inline void append_copy_dynamic_columns(const std::vector<DataFrame>& temporal_slices,
    //                                         Array_vector& arrays,
    //                                         int markovian_order,
    //                                         const DynamicVariable<Index>& v) {
    //     if (v.temporal_slice < 0 || v.temporal_slice > markovian_order) {
    //         throw std::invalid_argument("slice_index must be an index between 0 and " + 
    //                                         std::to_string(markovian_order));
    //     }

    //     if constexpr (std::is_convertible_v<Index, const std::string&>) {
    //         auto c = temporal_slices[v.temporal_slice].col(transform_temporal_name(v.variable, v.temporal_slice));
    //         arrays.push_back(copy_array(c));
    //     } else {
    //         auto c = temporal_slices[v.temporal_slice].col(v.variable);
    //         arrays.push_back(copy_array(c));
    //     }
    // }


    // template<typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int> = 0>
    // inline void append_copy_dynamic_columns(const std::vector<DataFrame>& temporal_slices,
    //                                         Array_vector& arrays,
    //                                         int markovian_order,
    //                                         const IndexIter& begin,
    //                                         const IndexIter& end) {
    //     for (auto it = begin; it != end; ++it) {
    //         append_copy_dynamic_columns(temporal_slices, arrays, markovian_order, *it);
    //     }
    // }

    // template<typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int> = 0>
    // inline void append_copy_dynamic_columns(const std::vector<DataFrame>& temporal_slices,
    //                                         Array_vector& arrays,
    //                                         int markovian_order,
    //                                         const std::pair<IndexIter, IndexIter>& it) {
    //     append_copy_dynamic_columns(temporal_slices, arrays, markovian_order, it.first, it.second);
    // }

    // template<typename T, util::enable_if_dynamic_index_container_t<T, int> = 0>
    // inline void append_copy_dynamic_columns(const std::vector<DataFrame>& temporal_slices,
    //                                         Array_vector& arrays,
    //                                         int markovian_order,
    //                                         const T& arg) { 
    //     append_copy_dynamic_columns(temporal_slices, arrays, markovian_order, arg.begin(), arg.end()); 
    // }


    // template<typename Index>
    // inline void append_dynamic_columns(const std::vector<DataFrame>& temporal_slices,
    //                                    Array_vector& arrays,
    //                                    int markovian_order,
    //                                    const DynamicVariable<Index>& v) {
    //     if (v.temporal_slice < 0 || v.temporal_slice > markovian_order) {
    //         throw std::invalid_argument("slice_index must be an index between 0 and " + 
    //                                         std::to_string(markovian_order));
    //     }

    //     if constexpr (std::is_convertible_v<Index, const std::string&>) {
    //         auto c = temporal_slices[v.temporal_slice].col(transform_temporal_name(v.variable, v.temporal_slice));
    //         arrays.push_back(c);
    //     } else {
    //         auto c = temporal_slices[v.temporal_slice].col(v.variable);
    //         arrays.push_back(c);
    //     }
    // }


    // template<typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int> = 0>
    // inline void append_dynamic_columns(const std::vector<DataFrame>& temporal_slices,
    //                                    Array_vector& arrays,
    //                                    int markovian_order,
    //                                    const IndexIter& begin,
    //                                    const IndexIter& end) {
    //     for (auto it = begin; it != end; ++it) {
    //         append_dynamic_columns(temporal_slices, arrays, markovian_order, *it);
    //     }
    // }

    // template<typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int> = 0>
    // inline void append_dynamic_columns(const std::vector<DataFrame>& temporal_slices,
    //                                    Array_vector& arrays,
    //                                    int markovian_order,
    //                                    const std::pair<IndexIter, IndexIter>& it) {
    //     append_dynamic_columns(temporal_slices, arrays, markovian_order, it.first, it.second);
    // }

    // template<typename T, util::enable_if_dynamic_index_container_t<T, int> = 0>
    // inline void append_dynamic_columns(const std::vector<DataFrame>& temporal_slices,
    //                                    Array_vector& arrays,
    //                                    int markovian_order,
    //                                    const T& arg) { 
    //     append_dynamic_columns(temporal_slices, arrays, markovian_order, arg.begin(), arg.end()); 
    // }

    // template<typename ...Args>
    // inline void append_dynamic_columns(const std::vector<DataFrame>& temporal_slices,
    //                                    Array_vector& arrays,
    //                                    int markovian_order,
    //                                    const CopyLOC<Args...>& cols) {
    //     if constexpr(std::tuple_size_v<std::remove_reference_t<decltype(cols.columns())>> == 0) {
    //         for (const auto& df : temporal_slices) {
    //             for (int i = 0; i < df->num_columns(); ++i) {
    //                 append_copy_columns(df.record_batch(), arrays, i);
    //             }
    //         }
    //     } else {
    //         std::apply([&temporal_slices, &arrays, markovian_order](const auto&...args) {
    //             (append_copy_dynamic_columns(temporal_slices, arrays, markovian_order, args),...);
    //         }, cols.columns());
    //     }
    // }

    // template<typename ...Args>
    // inline void append_dynamic_columns(const std::vector<DataFrame>& temporal_slices,
    //                            Array_vector& arrays,
    //                            int markovian_order,
    //                            const MoveLOC<Args...>& cols) {
    //     if constexpr(std::tuple_size_v<std::remove_reference_t<decltype(cols.columns())>> == 0) {
    //         for (const auto& df : temporal_slices) {
    //             for (int i = 0; i < df->num_columns(); ++i) {
    //                 append_columns(df.record_batch(), arrays, i);
    //             }
    //         }
    //     } else {
    //         std::apply([&temporal_slices, &arrays, markovian_order](const auto&...args) {
    //             (append_dynamic_columns(temporal_slices, arrays, markovian_order, args),...);
    //         }, cols.columns());
    //     }
    // }
    
    // template<typename Index>
    // inline void append_dynamic_schema(const std::vector<DataFrame>& temporal_slices,
    //                                   arrow::SchemaBuilder& b,
    //                                   const DynamicVariable<Index>& v) {
    //     if constexpr (std::is_convertible_v<Index, const std::string&>) {
    //         RAISE_STATUS_ERROR(b.AddField(temporal_slices[v.temporal_slice].field(
    //                             transform_temporal_name(v.variable, v.temporal_slice))));
    //     } else {
    //         RAISE_STATUS_ERROR(b.AddField(temporal_slices[v.temporal_slice].field(v.variable)));
    //     }
    // }

    // template<typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int> = 0>
    // inline void append_dynamic_schema(const std::vector<DataFrame>& temporal_slices,
    //                                   arrow::SchemaBuilder& b,
    //                                   const IndexIter& begin,
    //                                   const IndexIter& end) {
    //     for (auto it = begin; it != end; ++it) {
    //         append_dynamic_schema(temporal_slices, b, *it);
    //     }
    // }

    // template<typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int> = 0>
    // inline void append_dynamic_schema(const std::vector<DataFrame>& temporal_slices,
    //                                   arrow::SchemaBuilder& b,
    //                                   const std::pair<IndexIter, IndexIter>& it) {
    //     append_dynamic_schema(temporal_slices, b, it.first, it.second);
    // }

    // template<typename T, util::enable_if_dynamic_index_container_t<T, int> = 0>
    // inline void append_dynamic_schema(const std::vector<DataFrame>& temporal_slices,
    //                                   arrow::SchemaBuilder& b,
    //                                   const T& arg) { 
    //     append_dynamic_schema(temporal_slices, b, arg.begin(), arg.end()); 
    // }

    // template<bool copy, typename ...Args>
    // inline void append_schema(const std::vector<DataFrame>& temporal_slices,
    //                           arrow::SchemaBuilder& b,
    //                           const IndexLOC<copy, Args...>& cols) {
    //     if constexpr(std::tuple_size_v<std::remove_reference_t<decltype(cols.columns())>> == 0) {
    //         for (const auto& df : temporal_slices) {
    //             for (int i = 0; i < df->num_columns(); ++i) {
    //                 append_schema(df.record_batch(), b, i);
    //             }
    //         }
    //     } else {
    //         std::apply([&temporal_slices, &b](const auto&...args) {
    //             (append_dynamic_schema(temporal_slices, b, args),...);
    //         }, cols.columns());
    //     }
    // }

    // template<typename IndexIter, util::enable_if_dynamic_index_iterator_t<IndexIter, int>>
    // Array_vector DynamicDataFrame::indices_to_columns(const IndexIter& begin, const IndexIter& end) const {
    //     Array_vector v;
    //     v.reserve(std::distance(begin, end));
    //     append_dynamic_columns(m_temporal_slices, v, m_markovian_order, begin, end);
    //     return v;
    // }

    // template<typename ...Args>
    // Array_vector DynamicDataFrame::indices_to_columns(const Args&... args) const {
    //     Array_vector cols;

    //     int total_size = (size_argument(m_joint->num_columns(), args) + ...);
    //     cols.reserve(total_size);

    //     (append_dynamic_columns(m_temporal_slices, cols, m_markovian_order, args), ...);

    //     return cols;
    // }

    // template<typename ...Args>
    // DataFrame DynamicDataFrame::loc(const Args&... args) const {
    //     arrow::SchemaBuilder b(arrow::SchemaBuilder::ConflictPolicy::CONFLICT_APPEND);
    //     Array_vector new_cols;

    //     int total_size = (size_argument(m_joint->num_columns(), args) + ...);
    //     new_cols.reserve(total_size);

    //     (append_dynamic_columns(m_temporal_slices, new_cols, m_markovian_order, args),...);
    //     (append_dynamic_schema(m_temporal_slices, b, args),...);

    //     auto r = b.Finish();
    //     if (!r.ok()) {
    //         throw std::domain_error("Schema could not be created for selected columns.");
    //     }
    //     return DataFrame(RecordBatch::Make(std::move(r).ValueOrDie(), num_rows(), new_cols));
    // }
}

#endif //PYBNESIAN_DATASET_DYNAMIC_DATASET_HPP