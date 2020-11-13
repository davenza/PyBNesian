#ifndef PYBNESIAN_DATASET_VARIANT_IMPL_HPP
#define PYBNESIAN_DATASET_VARIANT_IMPL_HPP

#include <util/variant_util.hpp>

namespace dataset {
    template<bool append_ones, typename T, typename ArrowType, util::enable_if_index_container_t<T, int> = 0>
    DataFrame::EigenMatrixVariant DataFrame::to_eigen_variant_typed(T cols, Buffer_ptr bitmap) const {
//        TODO: Review static asserts.
//        static_assert(util::is_integral_container_v<T> || util::is_string_container_v<T>,
//                      "to_eigen_variant_typed() only accepts integral or string containers.");
        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, Dynamic>;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        if (bitmap) {
            auto rows = m_batch->num_rows();
            auto valid_rows = util::bit_util::non_null_count(bitmap, rows);

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

            auto bitmap_data = bitmap->data();

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
                    if (arrow::BitUtil::GetBit(bitmap_data, j))
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
    DataFrame::EigenMatrixVariant DataFrame::to_eigen_variant(T cols) const {
//        static_assert(!std::is_convertible_v<T,std::string> && (util::is_integral_container_v<T> || util::is_string_container_v<T>),
//                      "to_eigen() only accepts integral or string containers.");

        auto buffer_bitmap = combined_bitmap(cols);
        return to_eigen_variant<append_ones>(cols, buffer_bitmap);
    }

    template<bool append_ones, typename T, util::enable_if_index_container_t<T, int> = 0>
    DataFrame::EigenMatrixVariant DataFrame::to_eigen_variant(T cols, Buffer_ptr bitmap) const {
//        static_assert(!std::is_convertible_v<T,std::string> && (util::is_integral_container_v<T> || util::is_string_container_v<T>),
//                      "to_eigen() only accepts integral or string containers.");

        if (cols.size() == 0) {
//            TODO return empty matrix.
        }

        if (cols.size() == 1) {
            return util::variant_cast(to_eigen_variant<append_ones>(*cols.begin(), bitmap));
        }

        auto dt_id = [this, &cols]() {
            if constexpr(util::is_integral_container_v<T>) return m_batch->column(*cols.begin())->type_id();
            else if constexpr (util::is_string_container_v<T>) return m_batch->GetColumnByName(*cols.begin())->type_id();
        }();

        switch (dt_id) {
            case Type::DOUBLE:
                return to_eigen_variant_typed<append_ones, T, arrow::DoubleType>(cols, bitmap);
                break;
            case Type::FLOAT:
                return to_eigen_variant_typed<append_ones, T, arrow::FloatType>(cols, bitmap);
                break;
            default:
                throw pybind11::value_error("Only numeric data types can be transformed to Eigen matrix.");
        }
    }

    template<bool append_ones, typename ArrowType>
    DataFrame::EigenVectorVariant DataFrame::to_eigen_variant_typed(Array_ptr c, Buffer_ptr bitmap) const {
        using MatrixType = Matrix<typename ArrowType::c_type, Dynamic, 1+append_ones>;
        using MapType = Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>;
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        auto rows = c->length();

        if(bitmap) {
            auto valid_rows = util::bit_util::non_null_count(bitmap, rows);

            auto m = std::make_unique<MatrixType>(valid_rows, 1+append_ones);

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
    DataFrame::EigenVectorVariant DataFrame::to_eigen_variant(int i) const {
        auto buffer_bitmap = m_batch->column(i)->null_bitmap();
        return to_eigen_variant<append_ones>(i, buffer_bitmap);
    }

    template<bool append_ones, int = 0>
    DataFrame::EigenVectorVariant DataFrame::to_eigen_variant(int i, Buffer_ptr bitmap) const {
        auto col = m_batch->column(i);
        auto dt_id = col->type_id();
        switch (dt_id) {
            case Type::DOUBLE:
                return to_eigen_variant_typed<append_ones, DoubleType>(col, bitmap);
            case Type::FLOAT:
                return to_eigen_variant_typed<append_ones, FloatType>(col, bitmap);
            default:
                throw pybind11::value_error("Only numeric data types can be transformed to Eigen matrix.");
        }
    }

    template<bool append_ones, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    DataFrame::EigenVectorVariant DataFrame::to_eigen_variant(StringType name) const {
        //static_assert(std::is_convertible_v<StringType,std::string>, "to_eigen() only accepts integral or std::string.");
        auto buffer_bitmap = m_batch->GetColumnByName(name)->null_bitmap();
        return to_eigen_variant<append_ones>(name, buffer_bitmap);
    }

    template<bool append_ones, typename StringType, util::enable_if_stringable_t<StringType, int> = 0>
    DataFrame::EigenVectorVariant DataFrame::to_eigen_variant(StringType name, Buffer_ptr bitmap) const {
        auto col = m_batch->GetColumnByName(name);
        auto dt_id = col->type_id();
        switch (dt_id) {
            case Type::DOUBLE:
                return to_eigen_variant_typed<append_ones, DoubleType>(col, bitmap);
            case Type::FLOAT:
                return to_eigen_variant_typed<append_ones, FloatType>(col, bitmap);
            default:
                throw pybind11::value_error("Only numeric data types can be transformed to Eigen matrix.");
        }
    }
}

#endif //PYBNESIAN_DATASET_VARIANT_IMPL_HPP
