#include <algorithm>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/discrete/discrete_indices.hpp>
#include <learning/independences/hybrid/mutual_information.hpp>
#include <learning/parameters/mle_LinearGaussianCPD.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>

using factors::continuous::LinearGaussianCPD;
using learning::parameters::MLE;

#define SELECT_DISCRETE_CONTINUOUS_IMPL(contains_null, discrete_type, continuous_type, method, ...) \
    switch ((discrete_type)) {                                                                      \
        case Type::INT8:                                                                            \
            return ((continuous_type) == Type::DOUBLE)                                              \
                       ? method<contains_null, arrow::Int8Type, arrow::DoubleType>(__VA_ARGS__)     \
                       : method<contains_null, arrow::Int8Type, arrow::FloatType>(__VA_ARGS__);     \
        case Type::INT16:                                                                           \
            return ((continuous_type) == Type::DOUBLE)                                              \
                       ? method<contains_null, arrow::Int16Type, arrow::DoubleType>(__VA_ARGS__)    \
                       : method<contains_null, arrow::Int16Type, arrow::FloatType>(__VA_ARGS__);    \
        case Type::INT32:                                                                           \
            return ((continuous_type) == Type::DOUBLE)                                              \
                       ? method<contains_null, arrow::Int32Type, arrow::DoubleType>(__VA_ARGS__)    \
                       : method<contains_null, arrow::Int32Type, arrow::FloatType>(__VA_ARGS__);    \
        case Type::INT64:                                                                           \
            return ((continuous_type) == Type::DOUBLE)                                              \
                       ? method<contains_null, arrow::Int64Type, arrow::DoubleType>(__VA_ARGS__)    \
                       : method<contains_null, arrow::Int64Type, arrow::FloatType>(__VA_ARGS__);    \
        default:                                                                                    \
            throw std::runtime_error("Wrong index type! This code should be unreachable.");         \
    }

#define SELECT_CONTINUOUS_CONTINUOUS_IMPL(contains_null, continuous_type1, continuous_type2, method, ...) \
    switch ((continuous_type1)) {                                                                         \
        case Type::DOUBLE:                                                                                \
            ((continuous_type2) == Type::DOUBLE)                                                          \
                ? method<contains_null, arrow::DoubleType, arrow::DoubleType>(__VA_ARGS__)                \
                : method<contains_null, arrow::DoubleType, arrow::FloatType>(__VA_ARGS__);                \
            break;                                                                                        \
        case Type::FLOAT:                                                                                 \
            ((continuous_type2) == Type::DOUBLE)                                                          \
                ? method<contains_null, arrow::FloatType, arrow::DoubleType>(__VA_ARGS__)                 \
                : method<contains_null, arrow::FloatType, arrow::FloatType>(__VA_ARGS__);                 \
            break;                                                                                        \
        default:                                                                                          \
            throw std::runtime_error("Wrong index type! This code should be unreachable.");               \
    }

namespace learning::independences::hybrid {

struct ConditionalCovariance {
    std::vector<MatrixXd> cov_xyz;
    std::vector<MatrixXd> cov_xz;
    std::vector<MatrixXd> cov_yz;
    std::vector<MatrixXd> cov_z;
};

struct ConditionalMeans {
    std::vector<VectorXd> means_xyz;
    std::vector<VectorXd> means_xz;
    std::vector<VectorXd> means_yz;
    std::vector<VectorXd> means_z;
    VectorXi counts_xyz;
    VectorXi counts_xz;
    VectorXi counts_yz;
    VectorXi counts_z;
};

struct DiscreteConditions {
    bool x_is_discrete;
    bool y_is_discrete;
    bool has_discrete_z;
    VectorXi cardinality;
    VectorXi strides;
    VectorXi discrete_indices;
    int xyz_categories;
    int xz_categories;
    int yz_categories;
    int z_categories;
    int xyz_num_continuous;
    int xz_num_continuous;
    int yz_num_continuous;
    int z_num_continuous;
    int x_pos;
    int y_pos;
    int x_continuous_pos;
    int y_continuous_pos;
};

inline void xyz_marginal_indices(
    int index_xyz, const DiscreteConditions& dcond, int& index_xz, int& index_yz, int& index_z) {
    if (dcond.has_discrete_z) {
        if (dcond.x_is_discrete) {
            if (dcond.y_is_discrete) {
                auto x = index_xyz / dcond.strides(dcond.x_pos) % dcond.cardinality(dcond.x_pos);
                // Remove the X and Y from  the indices.
                index_z = index_xyz / dcond.strides(2);
                // Multiply again and divide by the cardianality of Y to remove Y. Then add the X component
                index_xz = (index_z * dcond.strides(2) / dcond.cardinality(dcond.y_pos)) + x;
                index_yz = index_xyz / dcond.cardinality(dcond.x_pos);
            } else {
                index_xz = index_xyz;
                index_z = index_xyz / dcond.cardinality(dcond.x_pos);
                index_yz = index_z;
            }
        } else {
            if (dcond.y_is_discrete) {
                index_yz = index_xyz;
                index_z = index_xyz / dcond.cardinality(dcond.y_pos);
                index_xz = index_z;
            } else {
                index_z = index_xz = index_yz = index_xyz;
            }
        }

    } else {
        index_z = 0;
        index_xz =
            (dcond.x_is_discrete) ? index_xyz / dcond.strides(dcond.x_pos) % dcond.cardinality(dcond.x_pos) : index_z;
        index_yz =
            (dcond.y_is_discrete) ? index_xyz / dcond.strides(dcond.y_pos) % dcond.cardinality(dcond.y_pos) : index_z;
    }
}

std::pair<int, int> xy_positions(const bool x_is_discrete, const bool y_is_discrete) {
    int x_pos = -1, y_pos = -1;

    if (x_is_discrete) {
        x_pos = 0;
        if (y_is_discrete) y_pos = 1;
    } else if (y_is_discrete) {
        y_pos = 0;
    }

    return std::make_pair(x_pos, y_pos);
}

std::pair<int, int> xy_continuous_positions(const bool x_is_discrete, const bool y_is_discrete) {
    int x_continuous_pos = -1, y_continuous_pos = -1;

    if (!x_is_discrete) {
        x_continuous_pos = 0;
        if (!y_is_discrete) y_continuous_pos = 1;
    } else if (!y_is_discrete) {
        y_continuous_pos = 0;
    }

    return std::make_pair(x_continuous_pos, y_continuous_pos);
}

template <bool contains_null, typename ArrowType>
void calculate_xcolumn_mean(const Array_ptr& column,
                            const uint8_t* bitmap_data,
                            const DiscreteConditions& dcond,
                            ConditionalMeans& cm) {
    const auto* column_data = column->data()->template GetValues<typename ArrowType::c_type>(1);

    for (int64_t i = 0, j = 0, i_end = column->length(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        int index_xz, index_yz, index_z;
        xyz_marginal_indices(dcond.discrete_indices(j), dcond, index_xz, index_yz, index_z);

        cm.means_xyz[dcond.discrete_indices(j)](dcond.x_continuous_pos) += column_data[i];
        cm.means_xz[index_xz](0) += column_data[i];
        ++j;
    }
}

template <bool contains_null, typename ArrowType>
void calculate_ycolumn_mean(const Array_ptr& column,
                            const uint8_t* bitmap_data,
                            const DiscreteConditions& dcond,
                            ConditionalMeans& cm) {
    const auto* column_data = column->data()->template GetValues<typename ArrowType::c_type>(1);

    for (int64_t i = 0, j = 0, i_end = column->length(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        int index_xz, index_yz, index_z;
        xyz_marginal_indices(dcond.discrete_indices(j), dcond, index_xz, index_yz, index_z);

        cm.means_xyz[dcond.discrete_indices(j)](dcond.y_continuous_pos) += column_data[i];
        cm.means_yz[index_yz](0) += column_data[i];
        ++j;
    }
}

template <bool contains_null, typename ArrowType>
void calculate_zcolumn_mean(const Array_ptr& column,
                            int column_index,
                            const uint8_t* bitmap_data,
                            const DiscreteConditions& dcond,
                            ConditionalMeans& cm) {
    const auto* column_data = column->data()->template GetValues<typename ArrowType::c_type>(1);

    auto xyz_col = !dcond.x_is_discrete + !dcond.y_is_discrete + column_index;
    auto xz_col = !dcond.x_is_discrete + column_index;
    auto yz_col = !dcond.y_is_discrete + column_index;

    for (int64_t i = 0, j = 0, i_end = column->length(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        int index_xz, index_yz, index_z;
        xyz_marginal_indices(dcond.discrete_indices(j), dcond, index_xz, index_yz, index_z);

        cm.means_xyz[dcond.discrete_indices(j)](xyz_col) += column_data[i];
        cm.means_xz[index_xz](xz_col) += column_data[i];
        cm.means_yz[index_yz](yz_col) += column_data[i];
        cm.means_z[index_z](column_index) += column_data[i];
        ++j;
    }
}

template <bool contains_null>
ConditionalMeans conditional_means_impl(const DataFrame& df,
                                        const uint8_t* bitmap_data,
                                        const std::vector<std::string>& continuous_z,
                                        const std::string& x,
                                        const std::string& y,
                                        const std::vector<std::string>& discrete_z,
                                        const DiscreteConditions& dcond) {
    ConditionalMeans cm{};

    cm.counts_xyz = VectorXi::Zero(dcond.xyz_categories);
    cm.counts_xz = VectorXi::Zero(dcond.xz_categories);
    cm.counts_yz = VectorXi::Zero(dcond.yz_categories);
    cm.counts_z = VectorXi::Zero(dcond.z_categories);

    for (int64_t i = 0, i_end = dcond.discrete_indices.rows(); i < i_end; ++i) {
        int index_xz, index_yz, index_z;
        xyz_marginal_indices(dcond.discrete_indices(i), dcond, index_xz, index_yz, index_z);

        ++cm.counts_xyz(dcond.discrete_indices(i));
        ++cm.counts_xz(index_xz);
        ++cm.counts_yz(index_yz);
        if (dcond.has_discrete_z) ++cm.counts_z(index_z);
    }

    if (!dcond.has_discrete_z) cm.counts_z(0) = df.valid_rows(continuous_z, x, y, discrete_z);

    cm.means_xyz.reserve(dcond.xyz_categories);
    cm.means_xz.reserve(dcond.xz_categories);
    cm.means_yz.reserve(dcond.yz_categories);
    cm.means_z.reserve(dcond.z_categories);

    for (auto i = 0; i < dcond.xyz_categories; ++i) {
        cm.means_xyz.push_back(VectorXd::Zero(dcond.xyz_num_continuous));
    }

    for (auto i = 0; i < dcond.xz_categories; ++i) {
        cm.means_xz.push_back(VectorXd::Zero(dcond.xz_num_continuous));
    }

    for (auto i = 0; i < dcond.yz_categories; ++i) {
        cm.means_yz.push_back(VectorXd::Zero(dcond.yz_num_continuous));
    }

    for (auto i = 0; i < dcond.z_categories; ++i) {
        cm.means_z.push_back(VectorXd::Zero(dcond.z_num_continuous));
    }

    if (!dcond.x_is_discrete) {
        auto col = df.col(x);
        switch (col->type_id()) {
            case Type::DOUBLE:
                calculate_xcolumn_mean<contains_null, arrow::DoubleType>(col, bitmap_data, dcond, cm);
                break;
            case Type::FLOAT:
                calculate_xcolumn_mean<contains_null, arrow::FloatType>(col, bitmap_data, dcond, cm);
                break;
            default:
                throw std::invalid_argument("Unreachable code!");
        }
    }

    if (!dcond.y_is_discrete) {
        auto col = df.col(y);
        switch (col->type_id()) {
            case Type::DOUBLE:
                calculate_ycolumn_mean<contains_null, arrow::DoubleType>(col, bitmap_data, dcond, cm);
                break;
            case Type::FLOAT:
                calculate_ycolumn_mean<contains_null, arrow::FloatType>(col, bitmap_data, dcond, cm);
                break;
            default:
                throw std::invalid_argument("Unreachable code!");
        }
    }

    for (int j = 0, j_end = static_cast<int>(continuous_z.size()); j < j_end; ++j) {
        auto col = df.col(continuous_z[j]);

        switch (col->type_id()) {
            case Type::DOUBLE:
                calculate_zcolumn_mean<contains_null, arrow::DoubleType>(col, j, bitmap_data, dcond, cm);
                break;
            case Type::FLOAT:
                calculate_zcolumn_mean<contains_null, arrow::FloatType>(col, j, bitmap_data, dcond, cm);
                break;
            default:
                throw std::invalid_argument("Unreachable code!");
        }
    }

    for (int i = 0; i < dcond.xyz_categories; ++i) {
        cm.means_xyz[i] /= cm.counts_xyz(i);
    }

    for (auto i = 0; i < dcond.xz_categories; ++i) {
        cm.means_xz[i] /= cm.counts_xz(i);
    }

    for (auto i = 0; i < dcond.yz_categories; ++i) {
        cm.means_yz[i] /= cm.counts_yz(i);
    }

    for (auto i = 0; i < dcond.z_categories; ++i) {
        cm.means_z[i] /= cm.counts_z(i);
    }

    return cm;
}

template <bool contains_null, typename ArrowType>
void calculate_xvariance(const Array_ptr& column,
                         const uint8_t* bitmap_data,
                         const DiscreteConditions& dcond,
                         const ConditionalMeans& cm,
                         ConditionalCovariance& cv) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

    auto dwn_col = std::static_pointer_cast<ArrayType>(column);
    auto* cont_data = dwn_col->raw_values();

    for (int64_t i = 0, j = 0, i_end = column->length(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        int index_xz, index_yz, index_z;
        xyz_marginal_indices(dcond.discrete_indices(j), dcond, index_xz, index_yz, index_z);

        auto d_xyz = cont_data[i] - cm.means_xyz[dcond.discrete_indices(j)](0);
        cv.cov_xyz[dcond.discrete_indices(j)](0, 0) += d_xyz * d_xyz;

        auto d_xz = cont_data[i] - cm.means_xz[index_xz](0);
        cv.cov_xz[index_xz](0, 0) += d_xz * d_xz;
        ++j;
    }
}

template <bool contains_null>
void calculate_xvariance(const Array_ptr& column,
                         const uint8_t* bitmap_data,
                         const DiscreteConditions& dcond,
                         const ConditionalMeans& cm,
                         ConditionalCovariance& cv) {
    switch (column->type_id()) {
        case Type::DOUBLE: {
            calculate_xvariance<contains_null, arrow::DoubleType>(column, bitmap_data, dcond, cm, cv);
            break;
        }
        case Type::FLOAT: {
            calculate_xvariance<contains_null, arrow::FloatType>(column, bitmap_data, dcond, cm, cv);
            break;
        }
        default:
            throw std::invalid_argument("Invalid continuous data type!");
    }
}

template <bool contains_null, typename ArrowType>
void calculate_yvariance(const Array_ptr& column,
                         const uint8_t* bitmap_data,
                         const DiscreteConditions& dcond,
                         const ConditionalMeans& cm,
                         ConditionalCovariance& cv) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

    auto dwn_col = std::static_pointer_cast<ArrayType>(column);
    auto* cont_data = dwn_col->raw_values();

    for (int64_t i = 0, j = 0, i_end = column->length(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        int index_xz, index_yz, index_z;
        xyz_marginal_indices(dcond.discrete_indices(j), dcond, index_xz, index_yz, index_z);

        auto d_xyz = cont_data[i] - cm.means_xyz[dcond.discrete_indices(j)](dcond.y_continuous_pos);
        cv.cov_xyz[dcond.discrete_indices(j)](dcond.y_continuous_pos, dcond.y_continuous_pos) += d_xyz * d_xyz;

        auto d_yz = cont_data[i] - cm.means_yz[index_yz](0);
        cv.cov_yz[index_yz](0, 0) += d_yz * d_yz;
        ++j;
    }
}

template <bool contains_null>
void calculate_yvariance(const Array_ptr& column,
                         const uint8_t* bitmap_data,
                         const DiscreteConditions& dcond,
                         const ConditionalMeans& cm,
                         ConditionalCovariance& cv) {
    switch (column->type_id()) {
        case Type::DOUBLE: {
            calculate_yvariance<contains_null, arrow::DoubleType>(column, bitmap_data, dcond, cm, cv);
            break;
        }
        case Type::FLOAT: {
            calculate_yvariance<contains_null, arrow::FloatType>(column, bitmap_data, dcond, cm, cv);
            break;
        }
        default:
            throw std::invalid_argument("Invalid continuous data type!");
    }
}

template <bool contains_null, typename ArrowType>
void calculate_zvariance(const Array_ptr& column,
                         int column_index,
                         const uint8_t* bitmap_data,
                         const DiscreteConditions& dcond,
                         const ConditionalMeans& cm,
                         ConditionalCovariance& cv) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

    auto dwn_col = std::static_pointer_cast<ArrayType>(column);
    auto* cont_data = dwn_col->raw_values();

    auto xyz_col = !dcond.x_is_discrete + !dcond.y_is_discrete + column_index;
    auto xz_col = !dcond.x_is_discrete + column_index;
    auto yz_col = !dcond.y_is_discrete + column_index;

    for (int64_t i = 0, j = 0, i_end = column->length(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        int index_xz, index_yz, index_z;
        xyz_marginal_indices(dcond.discrete_indices(j), dcond, index_xz, index_yz, index_z);

        auto d_xyz = cont_data[i] - cm.means_xyz[dcond.discrete_indices(j)](xyz_col);
        cv.cov_xyz[dcond.discrete_indices(j)](xyz_col, xyz_col) += d_xyz * d_xyz;

        auto d_xz = cont_data[i] - cm.means_xz[index_xz](xz_col);
        cv.cov_xz[index_xz](xz_col, xz_col) += d_xz * d_xz;

        double d_yz = cont_data[i] - cm.means_yz[index_yz](yz_col);
        cv.cov_yz[index_yz](yz_col, yz_col) += d_yz * d_yz;

        auto d_z = cont_data[i] - cm.means_z[index_z](column_index);
        cv.cov_z[index_z](column_index, column_index) += d_z * d_z;
        ++j;
    }
}

template <bool contains_null>
void calculate_zvariance(const Array_ptr& column,
                         int column_index,
                         const uint8_t* bitmap_data,
                         const DiscreteConditions& dcond,
                         const ConditionalMeans& cm,
                         ConditionalCovariance& cv) {
    switch (column->type_id()) {
        case Type::DOUBLE: {
            calculate_zvariance<contains_null, arrow::DoubleType>(column, column_index, bitmap_data, dcond, cm, cv);
            break;
        }
        case Type::FLOAT: {
            calculate_zvariance<contains_null, arrow::FloatType>(column, column_index, bitmap_data, dcond, cm, cv);
            break;
        }
        default:
            throw std::invalid_argument("Invalid continuous data type!");
    }
}

template <bool contains_null, typename ArrowTypeX, typename ArrowTypeY>
void calculate_xycovariance(const Array_ptr& x_column,
                            const Array_ptr& y_column,
                            const uint8_t* bitmap_data,
                            const DiscreteConditions& dcond,
                            const ConditionalMeans& cm,
                            ConditionalCovariance& cv) {
    using ArrayTypeX = typename arrow::TypeTraits<ArrowTypeX>::ArrayType;
    using ArrayTypeY = typename arrow::TypeTraits<ArrowTypeY>::ArrayType;

    auto dwn_colx = std::static_pointer_cast<ArrayTypeX>(x_column);
    auto* cont_datax = dwn_colx->raw_values();

    auto dwn_coly = std::static_pointer_cast<ArrayTypeY>(y_column);
    auto* cont_datay = dwn_coly->raw_values();

    for (int64_t i = 0, j = 0, i_end = x_column->length(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        auto dx_xyz = cont_datax[i] - cm.means_xyz[dcond.discrete_indices(j)](dcond.x_continuous_pos);
        auto dy_xyz = cont_datay[i] - cm.means_xyz[dcond.discrete_indices(j)](dcond.y_continuous_pos);
        cv.cov_xyz[dcond.discrete_indices(j)](dcond.x_continuous_pos, dcond.y_continuous_pos) += dx_xyz * dy_xyz;
        ++j;
    }

    for (auto i = 0; i < dcond.xyz_categories; ++i) {
        cv.cov_xyz[i](dcond.y_continuous_pos, dcond.x_continuous_pos) =
            cv.cov_xyz[i](dcond.x_continuous_pos, dcond.y_continuous_pos);
    }
}

template <bool contains_null>
void calculate_xycovariance(const Array_ptr& x_column,
                            const Array_ptr& y_column,
                            const uint8_t* bitmap_data,
                            const DiscreteConditions& dcond,
                            const ConditionalMeans& cm,
                            ConditionalCovariance& cv) {
    SELECT_CONTINUOUS_CONTINUOUS_IMPL(contains_null,
                                      x_column->type_id(),
                                      y_column->type_id(),
                                      calculate_xycovariance,
                                      x_column,
                                      y_column,
                                      bitmap_data,
                                      dcond,
                                      cm,
                                      cv)
}

template <bool contains_null, typename ArrowTypeX, typename ArrowTypeZ>
void calculate_xzcovariance(const Array_ptr& x_column,
                            const Array_ptr& z_column,
                            int zcolumn_index,
                            const uint8_t* bitmap_data,
                            const DiscreteConditions& dcond,
                            const ConditionalMeans& cm,
                            ConditionalCovariance& cv) {
    using ArrayTypeX = typename arrow::TypeTraits<ArrowTypeX>::ArrayType;
    using ArrayTypeZ = typename arrow::TypeTraits<ArrowTypeZ>::ArrayType;

    auto dwn_colx = std::static_pointer_cast<ArrayTypeX>(x_column);
    auto* cont_datax = dwn_colx->raw_values();

    auto dwn_colz = std::static_pointer_cast<ArrayTypeZ>(z_column);
    auto* cont_dataz = dwn_colz->raw_values();

    auto xyz_col = 1 + !dcond.y_is_discrete + zcolumn_index;
    auto xz_col = 1 + zcolumn_index;

    for (int64_t i = 0, j = 0, i_end = x_column->length(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        int index_xz, index_yz, index_z;
        xyz_marginal_indices(dcond.discrete_indices(j), dcond, index_xz, index_yz, index_z);

        auto dx_xyz = cont_datax[i] - cm.means_xyz[dcond.discrete_indices(j)](0);
        auto dz_xyz = cont_dataz[i] - cm.means_xyz[dcond.discrete_indices(j)](xyz_col);
        cv.cov_xyz[dcond.discrete_indices(j)](0, xyz_col) += dx_xyz * dz_xyz;

        auto dx_xz = cont_datax[i] - cm.means_xz[index_xz](0);
        auto dz_xz = cont_dataz[i] - cm.means_xz[index_xz](xz_col);
        cv.cov_xz[index_xz](0, xz_col) += dx_xz * dz_xz;
        ++j;
    }

    for (auto i = 0; i < dcond.xyz_categories; ++i) {
        cv.cov_xyz[i](xyz_col, 0) = cv.cov_xyz[i](0, xyz_col);
    }

    for (auto i = 0; i < dcond.xz_categories; ++i) {
        cv.cov_xz[i](xz_col, 0) = cv.cov_xz[i](0, xz_col);
    }
}

template <bool contains_null>
void calculate_xzcovariance(const Array_ptr& x_column,
                            const Array_ptr& z_column,
                            int zcolumn_index,
                            const uint8_t* bitmap_data,
                            const DiscreteConditions& dcond,
                            const ConditionalMeans& cm,
                            ConditionalCovariance& cv) {
    SELECT_CONTINUOUS_CONTINUOUS_IMPL(contains_null,
                                      x_column->type_id(),
                                      z_column->type_id(),
                                      calculate_xzcovariance,
                                      x_column,
                                      z_column,
                                      zcolumn_index,
                                      bitmap_data,
                                      dcond,
                                      cm,
                                      cv)
}

template <bool contains_null, typename ArrowTypeY, typename ArrowTypeZ>
void calculate_yzcovariance(const Array_ptr& y_column,
                            const Array_ptr& z_column,
                            int zcolumn_index,
                            const uint8_t* bitmap_data,
                            const DiscreteConditions& dcond,
                            const ConditionalMeans& cm,
                            ConditionalCovariance& cv) {
    using ArrayTypeY = typename arrow::TypeTraits<ArrowTypeY>::ArrayType;
    using ArrayTypeZ = typename arrow::TypeTraits<ArrowTypeZ>::ArrayType;

    auto dwn_coly = std::static_pointer_cast<ArrayTypeY>(y_column);
    auto* cont_datay = dwn_coly->raw_values();

    auto dwn_colz = std::static_pointer_cast<ArrayTypeZ>(z_column);
    auto* cont_dataz = dwn_colz->raw_values();

    auto xyz_col = !dcond.x_is_discrete + 1 + zcolumn_index;
    auto yz_col = 1 + zcolumn_index;

    for (int64_t i = 0, j = 0, i_end = y_column->length(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        int index_xz, index_yz, index_z;
        xyz_marginal_indices(dcond.discrete_indices(j), dcond, index_xz, index_yz, index_z);

        auto dy_xyz = cont_datay[i] - cm.means_xyz[dcond.discrete_indices(j)](dcond.y_continuous_pos);
        auto dz_xyz = cont_dataz[i] - cm.means_xyz[dcond.discrete_indices(j)](xyz_col);
        cv.cov_xyz[dcond.discrete_indices(j)](dcond.y_continuous_pos, xyz_col) += dy_xyz * dz_xyz;

        auto dy_yz = cont_datay[i] - cm.means_yz[index_yz](0);
        auto dz_yz = cont_dataz[i] - cm.means_yz[index_yz](yz_col);
        cv.cov_yz[index_yz](0, yz_col) += dy_yz * dz_yz;
        ++j;
    }

    for (auto i = 0; i < dcond.xyz_categories; ++i) {
        cv.cov_xyz[i](xyz_col, dcond.y_continuous_pos) = cv.cov_xyz[i](dcond.y_continuous_pos, xyz_col);
    }

    for (auto i = 0; i < dcond.yz_categories; ++i) {
        cv.cov_yz[i](yz_col, 0) = cv.cov_yz[i](0, yz_col);
    }
}

template <bool contains_null>
void calculate_yzcovariance(const Array_ptr& y_column,
                            const Array_ptr& z_column,
                            int zcolumn_index,
                            const uint8_t* bitmap_data,
                            const DiscreteConditions& dcond,
                            const ConditionalMeans& cm,
                            ConditionalCovariance& cv) {
    SELECT_CONTINUOUS_CONTINUOUS_IMPL(contains_null,
                                      y_column->type_id(),
                                      z_column->type_id(),
                                      calculate_yzcovariance,
                                      y_column,
                                      z_column,
                                      zcolumn_index,
                                      bitmap_data,
                                      dcond,
                                      cm,
                                      cv)
}

template <bool contains_null, typename ArrowType1, typename ArrowType2>
void calculate_zcovariance(const Array_ptr& column,
                           int column_index,
                           const Array_ptr& column2,
                           int column_index2,
                           const uint8_t* bitmap_data,
                           const DiscreteConditions& dcond,
                           const ConditionalMeans& cm,
                           ConditionalCovariance& cv) {
    using ArrayType1 = typename arrow::TypeTraits<ArrowType1>::ArrayType;
    using ArrayType2 = typename arrow::TypeTraits<ArrowType2>::ArrayType;

    auto xyz_col = !dcond.x_is_discrete + !dcond.y_is_discrete + column_index;
    auto xz_col = !dcond.x_is_discrete + column_index;
    auto yz_col = !dcond.y_is_discrete + column_index;

    auto xyz_col2 = !dcond.x_is_discrete + !dcond.y_is_discrete + column_index2;
    auto xz_col2 = !dcond.x_is_discrete + column_index2;
    auto yz_col2 = !dcond.y_is_discrete + column_index2;

    auto dwn_col = std::static_pointer_cast<ArrayType1>(column);
    auto* cont_data = dwn_col->raw_values();

    auto dwn_col2 = std::static_pointer_cast<ArrayType2>(column2);
    auto* cont_data2 = dwn_col2->raw_values();

    for (int64_t i = 0, j = 0, i_end = column->length(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        int index_xz, index_yz, index_z;
        xyz_marginal_indices(dcond.discrete_indices(j), dcond, index_xz, index_yz, index_z);

        auto d_xyz = cont_data[i] - cm.means_xyz[dcond.discrete_indices(j)](xyz_col);
        auto d_xyz2 = cont_data2[i] - cm.means_xyz[dcond.discrete_indices(j)](xyz_col2);
        cv.cov_xyz[dcond.discrete_indices(j)](xyz_col, xyz_col2) += d_xyz * d_xyz2;

        auto d_xz = cont_data[i] - cm.means_xz[index_xz](xz_col);
        auto d_xz2 = cont_data2[i] - cm.means_xz[index_xz](xz_col2);
        cv.cov_xz[index_xz](xz_col, xz_col2) += d_xz * d_xz2;

        double d_yz = cont_data[i] - cm.means_yz[index_yz](yz_col);
        double d_yz2 = cont_data2[i] - cm.means_yz[index_yz](yz_col2);
        cv.cov_yz[index_yz](yz_col, yz_col2) += d_yz * d_yz2;

        auto d_z = cont_data[i] - cm.means_z[index_z](column_index);
        auto d_z2 = cont_data2[i] - cm.means_z[index_z](column_index2);
        cv.cov_z[index_z](column_index, column_index2) += d_z * d_z2;
        ++j;
    }

    for (auto i = 0; i < dcond.xyz_categories; ++i) {
        cv.cov_xyz[i](xyz_col2, xyz_col) = cv.cov_xyz[i](xyz_col, xyz_col2);
    }

    for (auto i = 0; i < dcond.xz_categories; ++i) {
        cv.cov_xz[i](xz_col2, xz_col) = cv.cov_xz[i](xz_col, xz_col2);
    }

    for (auto i = 0; i < dcond.yz_categories; ++i) {
        cv.cov_yz[i](yz_col2, yz_col) = cv.cov_yz[i](yz_col, yz_col2);
    }

    for (auto i = 0; i < dcond.z_categories; ++i) {
        cv.cov_z[i](column_index2, column_index) = cv.cov_z[i](column_index, column_index2);
    }
}

template <bool contains_null>
void calculate_zcovariance(const Array_ptr& column,
                           int column_index,
                           const Array_ptr& column2,
                           int column_index2,
                           const uint8_t* bitmap_data,
                           const DiscreteConditions& dcond,
                           const ConditionalMeans& cm,
                           ConditionalCovariance& cv) {
    SELECT_CONTINUOUS_CONTINUOUS_IMPL(contains_null,
                                      column->type_id(),
                                      column2->type_id(),
                                      calculate_zcovariance,
                                      column,
                                      column_index,
                                      column2,
                                      column_index2,
                                      bitmap_data,
                                      dcond,
                                      cm,
                                      cv)
}

template <bool contains_null>
ConditionalCovariance conditional_covariance_impl(const DataFrame& df,
                                                  const std::vector<std::string>& continuous_z,
                                                  const std::string& x,
                                                  const std::string& y,
                                                  const std::vector<std::string>& discrete_z,
                                                  const DiscreteConditions& dcond) {
    Buffer_ptr combined_bitmap;
    const uint8_t* bitmap_data = nullptr;

    if constexpr (contains_null) {
        combined_bitmap = df.combined_bitmap(continuous_z, x, y, discrete_z);
        bitmap_data = combined_bitmap->data();
    }

    auto means = conditional_means_impl<contains_null>(df, bitmap_data, continuous_z, x, y, discrete_z, dcond);

    ConditionalCovariance cv;
    cv.cov_xyz.reserve(dcond.xyz_categories);
    cv.cov_xz.reserve(dcond.xz_categories);
    cv.cov_yz.reserve(dcond.yz_categories);
    cv.cov_z.reserve(dcond.z_categories);

    for (auto i = 0; i < dcond.xyz_categories; ++i) {
        cv.cov_xyz.push_back(MatrixXd::Zero(dcond.xyz_num_continuous, dcond.xyz_num_continuous));
    }

    for (auto i = 0; i < dcond.xz_categories; ++i) {
        cv.cov_xz.push_back(MatrixXd::Zero(dcond.xz_num_continuous, dcond.xz_num_continuous));
    }

    for (auto i = 0; i < dcond.yz_categories; ++i) {
        cv.cov_yz.push_back(MatrixXd::Zero(dcond.yz_num_continuous, dcond.yz_num_continuous));
    }

    for (auto i = 0; i < dcond.z_categories; ++i) {
        cv.cov_z.push_back(MatrixXd::Zero(dcond.z_num_continuous, dcond.z_num_continuous));
    }

    if (!dcond.x_is_discrete) {
        auto col = df.col(x);
        calculate_xvariance<contains_null>(col, bitmap_data, dcond, means, cv);

        if (!dcond.y_is_discrete) {
            auto coly = df.col(y);
            calculate_xycovariance<contains_null>(col, coly, bitmap_data, dcond, means, cv);
        }

        for (int i = 0, i_end = static_cast<int>(continuous_z.size()); i < i_end; ++i) {
            auto colz = df.col(continuous_z[i]);
            calculate_xzcovariance<contains_null>(col, colz, i, bitmap_data, dcond, means, cv);
        }
    }

    if (!dcond.y_is_discrete) {
        auto col = df.col(y);
        calculate_yvariance<contains_null>(col, bitmap_data, dcond, means, cv);

        for (int i = 0, i_end = static_cast<int>(continuous_z.size()); i < i_end; ++i) {
            auto colz = df.col(continuous_z[i]);
            calculate_yzcovariance<contains_null>(col, colz, i, bitmap_data, dcond, means, cv);
        }
    }

    for (int i = 0, i_end = static_cast<int>(continuous_z.size()); i < i_end; ++i) {
        auto c = df.col(continuous_z[i]);
        calculate_zvariance<contains_null>(c, i, bitmap_data, dcond, means, cv);
        for (int j = i + 1; j < i_end; ++j) {
            auto c2 = df.col(continuous_z[j]);
            calculate_zcovariance<contains_null>(c, i, c2, j, bitmap_data, dcond, means, cv);
        }
    }

    for (auto i = 0; i < dcond.xyz_categories; ++i) {
        cv.cov_xyz[i] /= means.counts_xyz(i) - 1;
    }

    for (auto i = 0; i < dcond.xz_categories; ++i) {
        cv.cov_xz[i] /= means.counts_xz(i) - 1;
    }

    for (auto i = 0; i < dcond.yz_categories; ++i) {
        cv.cov_yz[i] /= means.counts_yz(i) - 1;
    }

    for (auto i = 0; i < dcond.z_categories; ++i) {
        cv.cov_z[i] /= means.counts_z(i) - 1;
    }

    return cv;
}

std::pair<ConditionalCovariance, DiscreteConditions> conditional_covariance(
    const DataFrame& df,
    const std::vector<std::string>& continuous_z,
    const std::string& x,
    const std::string& y,
    const std::vector<std::string>& discrete_z) {
    bool x_is_discrete = df.is_discrete(x);
    bool y_is_discrete = df.is_discrete(y);
    auto [x_pos, y_pos] = xy_positions(x_is_discrete, y_is_discrete);
    auto [x_continuous_pos, y_continuous_pos] = xy_continuous_positions(x_is_discrete, y_is_discrete);

    std::vector<std::string> discrete_vars;
    discrete_vars.reserve(2 + discrete_z.size());

    if (x_is_discrete) discrete_vars.push_back(x);
    if (y_is_discrete) discrete_vars.push_back(y);

    discrete_vars.insert(discrete_vars.end(), discrete_z.begin(), discrete_z.end());

    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(df, discrete_vars);
    auto discrete_indices = factors::discrete::discrete_indices(df, discrete_vars, strides);

    auto xyz_categories =
        (!discrete_vars.empty()) ? strides(discrete_vars.size() - 1) * cardinality(discrete_vars.size() - 1) : 1;
    auto xz_categories = (y_is_discrete) ? xyz_categories / cardinality(y_pos) : xyz_categories;
    auto yz_categories = (x_is_discrete) ? xyz_categories / cardinality(x_pos) : xyz_categories;
    auto z_categories = (x_is_discrete) ? xz_categories / cardinality(x_pos) : xyz_categories;

    int xyz_num_continuous = !x_is_discrete + !y_is_discrete + continuous_z.size();
    int xz_num_continuous = !x_is_discrete + continuous_z.size();
    int yz_num_continuous = !y_is_discrete + continuous_z.size();
    int z_num_continuous = continuous_z.size();

    DiscreteConditions dcond{
        /*.x_is_discrete = */ x_is_discrete,
        /*.y_is_discrete = */ y_is_discrete,
        /*.has_discrete_z = */ !discrete_z.empty(),
        /*.cardinality = */ cardinality,
        /*.strides = */ strides,
        /*.discrete_indices = */ discrete_indices,
        /*.xyz_categories = */ xyz_categories,
        /*.xz_categories = */ xz_categories,
        /*.yz_categories = */ yz_categories,
        /*.z_categories = */ z_categories,
        /*.xyz_num_continuous = */ xyz_num_continuous,
        /*.xz_num_continuous = */ xz_num_continuous,
        /*.yz_num_continuous = */ yz_num_continuous,
        /*.z_num_continuous = */ z_num_continuous,
        /*.x_pos = */ x_pos,
        /*.y_pos = */ y_pos,
        /*.x_continuous_pos = */ x_continuous_pos,
        /*.y_continuous_pos = */ y_continuous_pos,
    };

    if (df.null_count(continuous_z, x, y, discrete_z) > 0) {
        return std::make_pair(conditional_covariance_impl<true>(df, continuous_z, x, y, discrete_z, dcond), dcond);
    } else {
        return std::make_pair(conditional_covariance_impl<false>(df, continuous_z, x, y, discrete_z, dcond), dcond);
    }
}

double entropy_mvn(int dimensionality, double cov_det) {
    auto d = static_cast<double>(dimensionality);
    return 0.5 * d + 0.5 * d * std::log(2 * util::pi<double>) + 0.5 * std::log(cov_det);
}

double MutualInformation::mi_discrete(const std::string& x, const std::string& y) const {
    std::vector<std::string> dummy_y{y};
    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, x, dummy_y);
    auto joint_counts = factors::discrete::joint_counts(m_df, x, dummy_y, cardinality, strides);

    auto x_marg = factors::discrete::marginal_counts(joint_counts, 0, cardinality, strides);
    auto y_marg = factors::discrete::marginal_counts(joint_counts, 1, cardinality, strides);

    double mi = 0;

    auto N = static_cast<double>(x_marg.sum());
    for (int i = 0; i < cardinality(0); ++i) {
        auto pi = static_cast<double>(x_marg(i)) / N;

        for (int j = 0; j < cardinality(1); ++j) {
            auto index = i + j * strides(1);
            auto Nij = static_cast<double>(joint_counts(index));

            if (Nij > 0) {
                // Divide first to avoid overflow if there is a large sample size.
                auto pij = Nij / N;
                auto pj = static_cast<double>(y_marg(j)) / N;

                mi += pij * std::log(pij / (pi * pj));
            }
        }
    }

    return mi;
}

template <bool contains_null, typename IndicesArrowType, typename ContinuousArrowType>
double MutualInformation::mi_mixed_impl(const std::string& discrete, const std::string& continuous) const {
    using IndicesArrayType = typename arrow::TypeTraits<IndicesArrowType>::ArrayType;
    Buffer_ptr combined_bitmap;
    const uint8_t* bitmap_data;

    if constexpr (contains_null) {
        combined_bitmap = m_df.combined_bitmap(discrete, continuous);
        bitmap_data = combined_bitmap->data();
    }

    auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(discrete));
    auto indices_array = std::static_pointer_cast<IndicesArrayType>(dict_array->indices());

    auto num_categories = dict_array->dictionary()->length();

    auto* discrete_data = indices_array->raw_values();
    auto* cont_data = m_df.data<ContinuousArrowType>(continuous);

    auto counts = VectorXd::Zero(num_categories).eval();
    auto mean = VectorXd::Zero(num_categories).eval();
    // Estimate the mean and variance of each discrete configuration
    for (int64_t i = 0, i_end = m_df->num_rows(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        mean(discrete_data[i]) += cont_data[i];
        ++counts(discrete_data[i]);
    }

    int total_counts;
    if constexpr (contains_null) {
        total_counts = counts.sum();
    } else {
        total_counts = m_df->num_rows();
    }

    auto total_mean = mean.sum() / total_counts;
    for (auto j = 0; j < num_categories; ++j) {
        mean(j) /= counts(j);
    }

    auto variance = VectorXd::Zero(num_categories).eval();
    double total_variance = 0;
    for (int64_t i = 0, i_end = m_df->num_rows(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        auto d = cont_data[i] - mean(discrete_data[i]);
        variance(discrete_data[i]) += d * d;

        auto d_total = cont_data[i] - total_mean;
        total_variance += d_total * d_total;
    }

    total_variance /= total_counts - 1;
    for (auto j = 0; j < num_categories; ++j) {
        variance(j) /= counts(j) - 1;
    }

    // MI(X_D; Y_C) = H(Y_C) - H(Y_C | X_D)

    // Add H(Y_C)
    double mi = 0.5 + 0.5 * std::log(2 * util::pi<double> * total_variance);
    for (auto j = 0; j < num_categories; ++j) {
        if (counts(j) > 0) {
            auto pj = static_cast<double>(counts(j)) / total_counts;
            auto h_yx = 0.5 + 0.5 * std::log(2 * util::pi<double> * variance(j));
            // Substract H(Y_C | X_D)
            mi -= pj * h_yx;
        }
    }

    return std::max(mi, 0.);
}

double MutualInformation::mi_mixed(const std::string& discrete, const std::string& continuous) const {
    auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(discrete));

    if (m_df.null_count(discrete, continuous) > 0) {
        SELECT_DISCRETE_CONTINUOUS_IMPL(true,
                                        dict_array->indices()->type_id(),
                                        m_df.col(continuous)->type_id(),
                                        mi_mixed_impl,
                                        discrete,
                                        continuous);
    } else {
        SELECT_DISCRETE_CONTINUOUS_IMPL(false,
                                        dict_array->indices()->type_id(),
                                        m_df.col(continuous)->type_id(),
                                        mi_mixed_impl,
                                        discrete,
                                        continuous);
    }
}

template <typename ArrowType>
double MutualInformation::mi_continuous_impl(const std::string& x, const std::string& y) const {
    auto pcov = m_df.cov<ArrowType>(x, y);
    auto& cov = *pcov;

    auto cor = cov(0, 1) / sqrt(cov(0, 0) * cov(1, 1));
    return -0.5 * std::log(1 - cor * cor);
}

double MutualInformation::mi_continuous(const std::string& x, const std::string& y) const {
    auto tt = m_df.same_type(x, y);

    switch (tt->id()) {
        case Type::DOUBLE:
            return mi_continuous_impl<arrow::DoubleType>(x, y);
        case Type::FLOAT:
            return mi_continuous_impl<arrow::FloatType>(x, y);
        default:
            throw std::runtime_error("Wrong data type! This code should be unreachable.");
    }
}

double MutualInformation::mi(const std::string& x, const std::string& y) const {
    if (m_df.is_discrete(x)) {
        if (m_df.is_discrete(y)) {
            return mi_discrete(x, y);
        } else {
            return mi_mixed(x, y);
        }
    } else {
        if (m_df.is_discrete(y)) {
            return mi_mixed(y, x);
        } else {
            return mi_continuous(x, y);
        }
    }
}

double MutualInformation::discrete_df(const std::string& x, const std::string& y) const {
    auto llx = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(x))->dictionary()->length();
    auto lly = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(y))->dictionary()->length();
    return (llx - 1) * (lly - 1);
}

double MutualInformation::mixed_df(const std::string& discrete, const std::string&) const {
    auto ll_discrete = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(discrete))->dictionary()->length();

    if (m_asymptotic_df) {
        return (ll_discrete - 1) * 2;
    } else {
        return ll_discrete - 1;
    }
}

double MutualInformation::calculate_df(const std::string& x, const std::string& y) const {
    if (m_df.is_discrete(x)) {
        if (m_df.is_discrete(y)) {
            return discrete_df(x, y);
        } else {
            return mixed_df(x, y);
        }
    } else {
        if (m_df.is_discrete(y)) {
            return mixed_df(y, x);
        } else {
            return 1;
        }
    }
}

double MutualInformation::pvalue(const std::string& x, const std::string& y) const {
    auto mi_value = mi(x, y);
    // Multiply by 2*N to obtain 2*N*MI(X; Y). This follows a X^2 distribution.
    mi_value *= 2 * m_df.valid_rows(x, y);
    auto df = calculate_df(x, y);

    boost::math::chi_squared_distribution chidist(static_cast<double>(df));
    return cdf(complement(chidist, mi_value));
}

/************************************************************
 *  SINGLE CONDITIONAL MI
 * **********************************************************/

template <bool contains_null, typename IndicesArrowType, typename ContinuousArrowType>
double MutualInformation::cmi_discrete_continuous_impl(const std::string& x,
                                                       const std::string& y,
                                                       const std::string& z) const {
    using IndicesArrayType = typename arrow::TypeTraits<IndicesArrowType>::ArrayType;

    Buffer_ptr combined_bitmap;
    const uint8_t* bitmap_data;

    if constexpr (contains_null) {
        combined_bitmap = m_df.combined_bitmap(x, y, z);
        bitmap_data = combined_bitmap->data();
    }

    std::vector<std::string> dummy_vars;
    dummy_vars.reserve(1);
    dummy_vars.push_back(y);

    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, x, dummy_vars);
    auto discrete_indices = factors::discrete::discrete_indices<contains_null>(m_df, x, dummy_vars, strides);

    auto x_array = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(x));
    auto x_indices = std::static_pointer_cast<IndicesArrayType>(x_array->indices());
    auto* x_data = x_indices->raw_values();

    auto y_array = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(y));
    auto y_indices = std::static_pointer_cast<IndicesArrayType>(y_array->indices());
    auto* y_data = y_indices->raw_values();

    auto* cont_data = m_df.data<ContinuousArrowType>(z);

    auto vars_configurations = cardinality(0) * cardinality(1);

    VectorXd means_xy = VectorXd::Zero(vars_configurations).eval();
    VectorXd means_x = VectorXd::Zero(cardinality(0)).eval();
    VectorXd means_y = VectorXd::Zero(cardinality(1)).eval();
    double total_mean = 0;

    VectorXi counts_xy = VectorXi::Zero(vars_configurations).eval();
    VectorXi counts_x = VectorXi::Zero(cardinality(0)).eval();
    VectorXi counts_y = VectorXi::Zero(cardinality(1)).eval();

    // Estimate the mean and variance of each discrete configuration
    for (int64_t i = 0, i_end = m_df->num_rows(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        means_xy(discrete_indices(i)) += cont_data[i];
        means_x(x_data[i]) += cont_data[i];
        means_y(y_data[i]) += cont_data[i];
        total_mean += cont_data[i];

        ++counts_xy(discrete_indices(i));
        ++counts_x(x_data[i]);
        ++counts_y(y_data[i]);
    }

    int total_counts;
    if constexpr (contains_null) {
        total_counts = discrete_indices.rows();
    } else {
        total_counts = m_df->num_rows();
    }

    means_xy = (means_xy.array() / counts_xy.template cast<double>().array()).matrix();
    means_x = (means_x.array() / counts_x.template cast<double>().array()).matrix();
    means_y = (means_y.array() / counts_y.template cast<double>().array()).matrix();
    total_mean /= total_counts;

    VectorXd variance_xy = VectorXd::Zero(vars_configurations).eval();
    VectorXd variance_x = VectorXd::Zero(cardinality(0)).eval();
    VectorXd variance_y = VectorXd::Zero(cardinality(1)).eval();
    double total_variance = 0;

    for (int64_t i = 0, i_end = m_df->num_rows(); i < i_end; ++i) {
        if constexpr (contains_null) {
            if (!util::bit_util::GetBit(bitmap_data, i)) continue;
        }

        auto d_xy = cont_data[i] - means_xy(discrete_indices(i));
        variance_xy(discrete_indices(i)) += d_xy * d_xy;

        auto d_x = cont_data[i] - means_x(x_data[i]);
        variance_x(x_data[i]) += d_x * d_x;

        auto d_y = cont_data[i] - means_y(y_data[i]);
        variance_y(y_data[i]) += d_y * d_y;

        auto d_total = cont_data[i] - total_mean;
        total_variance += d_total * d_total;
    }

    for (auto i = 0; i < vars_configurations; ++i) variance_xy(i) /= counts_xy(i) - 1;
    for (auto i = 0; i < cardinality(0); ++i) variance_x(i) /= counts_x(i) - 1;
    for (auto i = 0; i < cardinality(1); ++i) variance_y(i) /= counts_y(i) - 1;
    total_variance /= total_counts - 1;

    // MI(X; Y | Z) = MI(X; Y) + H(Z|X) + H(Z|Y) - H(Z|X, Y) - H(Z)
    double mi = 0;

    // Sum MI(X, Y) - H(Z|X, Y)
    for (int i = 0; i < cardinality(0); ++i) {
        for (int j = 0; j < cardinality(1); ++j) {
            auto k = i + j * strides(1);
            if (counts_xy(k) > 0) {
                auto nij = counts_xy(k);
                auto ni = counts_x(i);
                auto nj = counts_y(j);

                double pij = static_cast<double>(nij) / total_counts;
                double pi = static_cast<double>(ni) / total_counts;
                double pj = static_cast<double>(nj) / total_counts;

                auto h_xy = 0.5 + 0.5 * std::log(2 * util::pi<double> * variance_xy(k));
                mi += pij * (-h_xy + std::log(pij / (pi * pj)));
            }
        }
    }

    // Sum H(Z|X)
    for (int i = 0; i < cardinality(0); ++i) {
        if (counts_x(i) > 0) {
            auto h_x = 0.5 + 0.5 * std::log(2 * util::pi<double> * variance_x(i));
            double pi = static_cast<double>(counts_x(i)) / total_counts;
            mi += pi * h_x;
        }
    }

    // Sum H(Z|Y)
    for (int j = 0; j < cardinality(1); ++j) {
        if (counts_y(j) > 0) {
            auto h_y = 0.5 + 0.5 * std::log(2 * util::pi<double> * variance_y(j));
            double pj = static_cast<double>(counts_y(j)) / total_counts;
            mi += pj * h_y;
        }
    }

    // Sum - H(Z)
    mi -= 0.5 + 0.5 * std::log(2 * util::pi<double> * total_variance);
    return std::max(mi, 0.);
}

double MutualInformation::cmi_discrete_continuous(const std::string& x,
                                                  const std::string& y,
                                                  const std::string& z) const {
    auto discrete_type = std::static_pointer_cast<arrow::DictionaryType>(m_df.same_type(x, y));
    auto continuous_type = m_df.same_type(z);

    if (m_df.null_count(x, y, z) > 0) {
        SELECT_DISCRETE_CONTINUOUS_IMPL(
            true, discrete_type->index_type()->id(), continuous_type->id(), cmi_discrete_continuous_impl, x, y, z);
    } else {
        SELECT_DISCRETE_CONTINUOUS_IMPL(
            false, discrete_type->index_type()->id(), continuous_type->id(), cmi_discrete_continuous_impl, x, y, z);
    }
}

double MutualInformation::mi(const std::string& x, const std::string& y, const std::string& z) const {
    if (m_df.is_discrete(x)) {
        if (m_df.is_discrete(y)) {
            return (m_df.is_discrete(z)) ? cmi_discrete_discrete(x, y, {z}) : cmi_discrete_continuous(x, y, z);
        } else {
            return (m_df.is_discrete(z)) ? cmi_general_mixed(x, y, {z}, {}) : cmi_general_mixed(x, y, {}, {z});
        }
    } else {
        if (m_df.is_discrete(y)) {
            return (m_df.is_discrete(z)) ? cmi_general_mixed(y, x, {z}, {}) : cmi_general_mixed(y, x, {}, {z});
        } else {
            return (m_df.is_discrete(z)) ? cmi_general_both_continuous(y, x, {z}, {})
                                         : cmi_general_both_continuous(y, x, {}, {z});
        }
    }
}

double MutualInformation::discrete_df(const std::string& x, const std::string& y, const std::string& z) const {
    auto llx = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(x))->dictionary()->length();
    auto lly = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(y))->dictionary()->length();

    if (m_df.is_discrete(z)) {
        auto llz = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(z))->dictionary()->length();
        return (llx - 1) * (lly - 1) * llz;
    } else {
        if (m_asymptotic_df) {
            return (llx - 1) * (lly - 1) * 3;
        } else {
            // By simulation:
            return (llx - 1) * (lly - 1) * 2;
        }
    }
}

double MutualInformation::mixed_df(const std::string& discrete, const std::string&, const std::string& z) const {
    auto llx = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(discrete))->dictionary()->length();

    if (m_df.is_discrete(z)) {
        auto llz = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(z))->dictionary()->length();
        if (m_asymptotic_df) {
            return (llx - 1) * llz * 2;
        } else {
            // By simulation:
            return (llx - 1) * llz;
        }
    } else {
        if (m_asymptotic_df) {
            return (llx - 1) * 3;
        } else {
            // By simulation:
            return (llx - 1) * 2;
        }
    }
}

double MutualInformation::continuous_df(const std::string&, const std::string&, const std::string& z) const {
    if (m_df.is_discrete(z)) {
        auto llz = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(z))->dictionary()->length();
        return llz;
    } else {
        return 1;
    }
}

double MutualInformation::calculate_df(const std::string& x, const std::string& y, const std::string& z) const {
    if (m_df.is_discrete(x)) {
        if (m_df.is_discrete(y)) {
            return discrete_df(x, y, z);
        } else {
            return mixed_df(x, y, z);
        }
    } else {
        if (m_df.is_discrete(y)) {
            return mixed_df(y, x, z);
        } else {
            return continuous_df(x, y, z);
        }
    }
}

double MutualInformation::pvalue(const std::string& x, const std::string& y, const std::string& z) const {
    auto mi_value = mi(x, y, z);
    // Multiply by 2*N to obtain 2*N*MI(X; Y). This follows a X^2 distribution.
    mi_value *= 2 * m_df.valid_rows(x, y, z);
    auto df = calculate_df(x, y, z);

    boost::math::chi_squared_distribution chidist(static_cast<double>(df));
    return cdf(complement(chidist, mi_value));
}

/************************************************************
 *  CONDITIONAL MI
 * **********************************************************/

double MutualInformation::cmi_discrete_discrete(const std::string& x,
                                                const std::string& y,
                                                const std::vector<std::string>& discrete_z) const {
    if (discrete_z.empty()) return mi_discrete(x, y);

    std::vector<std::string> dummy_vars;
    dummy_vars.reserve(1 + discrete_z.size());
    dummy_vars.push_back(y);
    dummy_vars.insert(dummy_vars.end(), discrete_z.begin(), discrete_z.end());

    auto [cardinality, strides] = factors::discrete::create_cardinality_strides(m_df, x, dummy_vars);
    auto joint_counts = factors::discrete::joint_counts(m_df, x, dummy_vars, cardinality, strides);

    auto vars_configurations = strides(2);
    auto evidence_configurations =
        (cardinality(cardinality.rows() - 1) * strides(strides.rows() - 1)) / vars_configurations;

    double mi = 0;

    auto N = static_cast<double>(joint_counts.sum());
    for (auto k = 0; k < evidence_configurations; ++k) {
        auto offset = k * vars_configurations;

        int Nz = 0;
        VectorXi Nxz = VectorXi::Zero(cardinality(0));
        VectorXi Nyz = VectorXi::Zero(cardinality(1));

        for (auto i = 0; i < cardinality(0); ++i) {
            for (auto j = 0; j < cardinality(1); ++j) {
                auto Nxyz = joint_counts(offset + i + j * strides(1));
                Nxz(i) += Nxyz;
                Nyz(j) += Nxyz;
                Nz += Nxyz;
            }
        }

        if (Nz == 0) continue;

        auto pz = static_cast<double>(Nz) / N;

        for (auto i = 0; i < cardinality(0); ++i) {
            auto pxz = static_cast<double>(Nxz(i)) / N;
            for (auto j = 0; j < cardinality(1); ++j) {
                auto Nxyz = joint_counts(offset + i + j * strides(1));
                if (Nxyz > 0) {
                    auto pxyz = static_cast<double>(Nxyz) / N;
                    auto pyz = static_cast<double>(Nyz(j)) / N;
                    mi += pxyz * std::log(static_cast<double>(pz * pxyz) / (pxz * pyz));
                }
            }
        }
    }

    // mi contains N*MI(X; Y).
    return mi;
}

double MutualInformation::cmi_general_both_discrete(const std::string& x,
                                                    const std::string& y,
                                                    const std::vector<std::string>& discrete_z,
                                                    const std::vector<std::string>& continuous_z) const {
    if (continuous_z.empty()) return cmi_discrete_discrete(x, y, discrete_z);

    auto [cv, dcond] = conditional_covariance(m_df, continuous_z, x, y, discrete_z);

    double N = m_df.valid_rows(continuous_z, x, y, discrete_z);
    auto vars_configurations = dcond.cardinality(dcond.x_pos) * dcond.cardinality(dcond.y_pos);

    VectorXd joint_counts = VectorXd::Zero(dcond.xyz_categories);
    // Compute counts
    for (auto i = 0; i < dcond.discrete_indices.rows(); ++i) {
        ++joint_counts(dcond.discrete_indices(i));
    }

    double mi = 0;
    for (auto k = 0; k < dcond.z_categories; ++k) {
        auto offset = k * vars_configurations;

        int Nz = 0;
        VectorXi Nxz = VectorXi::Zero(dcond.cardinality(dcond.x_pos));
        VectorXi Nyz = VectorXi::Zero(dcond.cardinality(dcond.y_pos));

        for (auto i = 0; i < dcond.cardinality(dcond.x_pos); ++i) {
            for (auto j = 0; j < dcond.cardinality(dcond.y_pos); ++j) {
                auto index_xyz = offset + i + j * dcond.strides(dcond.y_pos);
                auto Nxyz = joint_counts(index_xyz);

                Nxz(i) += Nxyz;
                Nyz(j) += Nxyz;
                Nz += Nxyz;
            }
        }

        if (Nz == 0) continue;

        auto pz = static_cast<double>(Nz) / N;
        for (auto i = 0; i < dcond.cardinality(dcond.x_pos); ++i) {
            auto pxz = static_cast<double>(Nxz(i)) / N;
            for (auto j = 0; j < dcond.cardinality(dcond.y_pos); ++j) {
                auto index_xyz = offset + i + j * dcond.strides(dcond.y_pos);
                auto Nxyz = joint_counts(index_xyz);
                if (Nxyz == 0) continue;

                auto pyz = static_cast<double>(Nyz(j)) / N;
                auto pxyz = static_cast<double>(Nxyz) / N;

                auto h_xyz = entropy_mvn(continuous_z.size(), cv.cov_xyz[index_xyz].determinant());
                // Add the MI(X; Y | Z_D) - H(Z_C | X, Y, Z_D)
                mi += pxyz * (std::log((pz * pxyz) / (pxz * pyz)) - h_xyz);
            }
        }

        auto offset_xz = k * dcond.cardinality(dcond.x_pos);
        for (auto i = 0; i < dcond.cardinality(dcond.x_pos); ++i) {
            if (Nxz(i) == 0) continue;

            auto index_xz = offset_xz + i;
            auto pxz = static_cast<double>(Nxz(i)) / N;
            auto h_xz = entropy_mvn(continuous_z.size(), cv.cov_xz[index_xz].determinant());
            // Add H(Z_C | X, Z_D)
            mi += pxz * h_xz;
        }

        auto offset_yz = k * dcond.cardinality(dcond.y_pos);
        for (auto j = 0; j < dcond.cardinality(dcond.y_pos); ++j) {
            if (Nyz(j) == 0) continue;

            auto index_yz = offset_yz + j;
            auto pyz = static_cast<double>(Nyz(j)) / N;
            auto h_yz = entropy_mvn(continuous_z.size(), cv.cov_yz[index_yz].determinant());
            // Add H(Z_C | Y, Z_D)
            mi += pyz * h_yz;
        }

        // Substract H(Z_C | Z_D)
        auto h_z = entropy_mvn(continuous_z.size(), cv.cov_z[k].determinant());
        mi -= pz * h_z;
    }

    return std::max(mi, 0.);
}

double MutualInformation::cmi_general_mixed(const std::string& x_discrete,
                                            const std::string& y_continuous,
                                            const std::vector<std::string>& discrete_z,
                                            const std::vector<std::string>& continuous_z) const {
    auto [cv, dcond] = conditional_covariance(m_df, continuous_z, x_discrete, y_continuous, discrete_z);

    double N = m_df.valid_rows(continuous_z, x_discrete, y_continuous, discrete_z);
    auto vars_configurations = dcond.cardinality(dcond.x_pos);

    VectorXd joint_counts = VectorXd::Zero(dcond.xyz_categories);
    // Compute counts
    for (auto i = 0; i < dcond.discrete_indices.rows(); ++i) {
        ++joint_counts(dcond.discrete_indices(i));
    }

    double mi = 0;
    for (auto k = 0; k < dcond.z_categories; ++k) {
        auto offset = k * vars_configurations;

        int Nz = 0;
        for (auto i = 0; i < dcond.cardinality(dcond.x_pos); ++i) {
            Nz += joint_counts(offset + i);
        }

        if (Nz == 0) continue;

        auto pz = static_cast<double>(Nz) / N;
        for (auto i = 0; i < dcond.cardinality(dcond.x_pos); ++i) {
            auto Nxz = joint_counts(offset + i);
            if (Nxz == 0) continue;

            auto pxz = static_cast<double>(Nxz) / N;
            auto h_xyz = entropy_mvn(continuous_z.size() + 1, cv.cov_xyz[offset + i].determinant());
            // Substract H(Z_C, Y | X, Z_D)
            mi -= pxz * h_xyz;

            if (!continuous_z.empty()) {
                auto h_xz = entropy_mvn(continuous_z.size(), cv.cov_xz[offset + i].determinant());
                // Add H(Z_C | X, Z_D)
                mi += pxz * h_xz;
            }
        }

        auto h_yz = entropy_mvn(continuous_z.size() + 1, cv.cov_yz[k].determinant());
        // Add H(Z_C, Y | Z_D)
        mi += pz * h_yz;

        if (!continuous_z.empty()) {
            auto h_z = entropy_mvn(continuous_z.size(), cv.cov_z[k].determinant());
            // Substract H(Z_C | Z_D)
            mi -= pz * h_z;
        }
    }

    return std::max(mi, 0.);
}

double MutualInformation::cmi_general_both_continuous(const std::string& x,
                                                      const std::string& y,
                                                      const std::vector<std::string>& discrete_z,
                                                      const std::vector<std::string>& continuous_z) const {
    auto [cv, dcond] = conditional_covariance(m_df, continuous_z, x, y, discrete_z);

    double N = m_df.valid_rows(continuous_z, x, y, discrete_z);

    VectorXd joint_counts = VectorXd::Zero(dcond.xyz_categories);
    // Compute counts
    for (auto i = 0; i < dcond.discrete_indices.rows(); ++i) {
        ++joint_counts(dcond.discrete_indices(i));
    }

    double mi = 0;
    for (auto k = 0; k < dcond.z_categories; ++k) {
        auto Nz = joint_counts(k);
        if (Nz == 0) continue;

        auto pz = static_cast<double>(Nz) / N;

        auto h_xyz = entropy_mvn(continuous_z.size() + 2, cv.cov_xyz[k].determinant());
        auto h_xz = entropy_mvn(continuous_z.size() + 1, cv.cov_xz[k].determinant());
        auto h_yz = entropy_mvn(continuous_z.size() + 1, cv.cov_yz[k].determinant());

        // Add H(X, Z_C | Z_D) + H(Y, Z_C | Z_D) - H(X, Y, Z_C | Z_D)
        mi += pz * (h_xz + h_yz - h_xyz);
        if (!continuous_z.empty()) {
            auto h_z = entropy_mvn(continuous_z.size(), cv.cov_z[k].determinant());
            // Substract H(Z_C | Z_D)
            mi -= pz * h_z;
        }
    }

    return std::max(mi, 0.);
}

double MutualInformation::cmi_general(const std::string& x,
                                      const std::string& y,
                                      const std::vector<std::string>& discrete_z,
                                      const std::vector<std::string>& continuous_z) const {
    if (m_df.is_discrete(x)) {
        if (m_df.is_discrete(y)) {
            return cmi_general_both_discrete(x, y, discrete_z, continuous_z);
        } else {
            return cmi_general_mixed(x, y, discrete_z, continuous_z);
        }
    } else {
        if (m_df.is_discrete(y)) {
            return cmi_general_mixed(y, x, discrete_z, continuous_z);
        } else {
            return cmi_general_both_continuous(y, x, discrete_z, continuous_z);
        }
    }
}

double MutualInformation::mi(const std::string& x, const std::string& y, const std::vector<std::string>& z) const {
    std::vector<std::string> discrete_z;
    std::vector<std::string> continuous_z;

    for (const auto& e : z) {
        if (m_df.is_discrete(e))
            discrete_z.push_back(e);
        else
            continuous_z.push_back(e);
    }

    return cmi_general(x, y, discrete_z, continuous_z);
}

double MutualInformation::discrete_df(const std::string& x,
                                      const std::string& y,
                                      const std::vector<std::string>& discrete_z,
                                      const std::vector<std::string>& continuous_z) const {
    auto llx = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(x))->dictionary()->length();
    auto lly = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(y))->dictionary()->length();

    auto llz = 1;
    for (const auto& dz : discrete_z) {
        llz *= std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(dz))->dictionary()->length();
    }

    auto zc = continuous_z.size();

    if (m_asymptotic_df) {
        return (llx - 1) * (lly - 1) * llz * (1 + 0.5 * (zc * (zc + 3)));
    } else {
        return (llx - 1) * (lly - 1) * llz * (1 + 0.5 * (zc * (zc + 1)));
    }
}

double MutualInformation::mixed_df(const std::string& discrete,
                                   const std::string&,
                                   const std::vector<std::string>& discrete_z,
                                   const std::vector<std::string>& continuous_z) const {
    auto llx = std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(discrete))->dictionary()->length();

    auto llz = 1;
    for (const auto& dz : discrete_z) {
        llz *= std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(dz))->dictionary()->length();
    }

    auto zc = continuous_z.size();

    if (m_asymptotic_df) {
        return (llx - 1) * llz * (zc + 2);
    } else {
        // By simulation:
        return (llx - 1) * llz * (zc + 1);
    }
}

double MutualInformation::continuous_df(const std::string&,
                                        const std::string&,
                                        const std::vector<std::string>& discrete_z,
                                        const std::vector<std::string>&) const {
    auto llz = 1;
    for (const auto& dz : discrete_z) {
        llz *= std::static_pointer_cast<arrow::DictionaryArray>(m_df.col(dz))->dictionary()->length();
    }

    return llz;
}

double MutualInformation::calculate_df(const std::string& x,
                                       const std::string& y,
                                       const std::vector<std::string>& discrete_z,
                                       const std::vector<std::string>& continuous_z) const {
    if (m_df.is_discrete(x)) {
        if (m_df.is_discrete(y)) {
            return discrete_df(x, y, discrete_z, continuous_z);
        } else {
            return mixed_df(x, y, discrete_z, continuous_z);
        }
    } else {
        if (m_df.is_discrete(y)) {
            return mixed_df(y, x, discrete_z, continuous_z);
        } else {
            return continuous_df(x, y, discrete_z, continuous_z);
        }
    }
}

double MutualInformation::pvalue(const std::string& x, const std::string& y, const std::vector<std::string>& z) const {
    std::vector<std::string> discrete_z;
    std::vector<std::string> continuous_z;

    for (const auto& e : z) {
        if (m_df.is_discrete(e))
            discrete_z.push_back(e);
        else
            continuous_z.push_back(e);
    }

    auto mi_value = cmi_general(x, y, discrete_z, continuous_z);
    // Multiply by 2*N to obtain 2*N*MI(X; Y). This follows a X^2 distribution.
    mi_value *= 2 * m_df.valid_rows(x, y, z);
    auto df = calculate_df(x, y, discrete_z, continuous_z);

    boost::math::chi_squared_distribution chidist(static_cast<double>(df));
    return cdf(complement(chidist, mi_value));
}

}  // namespace learning::independences::hybrid