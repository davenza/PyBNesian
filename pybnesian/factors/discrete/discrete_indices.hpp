#ifndef PYBNESIAN_FACTORS_DISCRETE_DISCRETE_INDICES_HPP
#define PYBNESIAN_FACTORS_DISCRETE_DISCRETE_INDICES_HPP

#include <arrow/compute/api.h>
#include <Eigen/Dense>
#include <dataset/dataset.hpp>
#include <factors/assignment.hpp>
#include <util/hash_utils.hpp>

using dataset::DataFrame;
using Eigen::VectorXi;

namespace factors::discrete {

void check_is_string_dictionary(const std::shared_ptr<arrow::DictionaryArray>& dict, const std::string& variable);

template <typename ArrowType>
void sum_to_discrete_indices_null(VectorXi& accum_indices,
                                  Array_ptr& indices,
                                  int stride,
                                  Buffer_ptr& combined_bitmap) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto dwn_indices = std::static_pointer_cast<ArrayType>(indices);

    auto raw_combined_bitmap = combined_bitmap->data();
    for (auto i = 0, j = 0; i < indices->length(); ++i) {
        if (util::bit_util::GetBit(raw_combined_bitmap, i)) {
            accum_indices(j++) += dwn_indices->Value(i) * stride;
        }
    }
}

void sum_to_discrete_indices_null(VectorXi& accum_indices, Array_ptr& indices, int stride, Buffer_ptr& combined_bitmap);

template <typename ArrowType>
void sum_to_discrete_indices(VectorXi& accum_indices, Array_ptr& indices, int stride) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using EigenMap = Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>;
    auto dwn_indices = std::static_pointer_cast<ArrayType>(indices);
    auto* raw_values = dwn_indices->raw_values();
    const EigenMap map_eigen(raw_values, indices->length());
    accum_indices += map_eigen.template cast<int>() * stride;
}

void sum_to_discrete_indices(VectorXi& accum_indices, Array_ptr& indices, int stride);

template <bool contains_null>
VectorXi discrete_indices(const DataFrame& df,
                          const std::string& variable,
                          const std::vector<std::string>& evidence,
                          const VectorXi& strides) {
    if constexpr (contains_null) {
        auto combined_bitmap = df.combined_bitmap(variable, evidence);

        auto valid_rows = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

        VectorXi indices = VectorXi::Zero(valid_rows);

        auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));
        auto variable_indices = dict_variable->indices();

        sum_to_discrete_indices_null(indices, variable_indices, strides(0), combined_bitmap);

        int i = 1;
        for (auto it = evidence.begin(), end = evidence.end(); it != end; ++it, ++i) {
            auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
            auto evidence_indices = dict_evidence->indices();
            sum_to_discrete_indices_null(indices, evidence_indices, strides(i), combined_bitmap);
        }

        return indices;
    } else {
        VectorXi indices = VectorXi::Zero(df->num_rows());

        auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));
        auto variable_indices = dict_variable->indices();

        sum_to_discrete_indices(indices, variable_indices, strides(0));

        int i = 1;
        for (auto it = evidence.begin(), end = evidence.end(); it != end; ++it, ++i) {
            auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
            auto evidence_indices = dict_evidence->indices();
            sum_to_discrete_indices(indices, evidence_indices, strides(i));
        }

        return indices;
    }
}

VectorXi discrete_indices(const DataFrame& df,
                          const std::string& variable,
                          const std::vector<std::string>& evidence,
                          const VectorXi& strides);

template <bool contains_null>
VectorXi discrete_indices(const DataFrame& df, const std::vector<std::string>& variables, const VectorXi& strides) {
    if constexpr (contains_null) {
        auto combined_bitmap = df.combined_bitmap(variables);

        auto valid_rows = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

        VectorXi indices = VectorXi::Zero(valid_rows);

        int i = 0;
        for (auto it = variables.begin(), end = variables.end(); it != end; ++it, ++i) {
            auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
            auto variable_indices = dict_variable->indices();
            sum_to_discrete_indices_null(indices, variable_indices, strides(i), combined_bitmap);
        }

        return indices;
    } else {
        VectorXi indices = VectorXi::Zero(df->num_rows());

        int i = 0;
        for (auto it = variables.begin(), end = variables.end(); it != end; ++it, ++i) {
            auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
            auto variable_indices = dict_variable->indices();
            sum_to_discrete_indices(indices, variable_indices, strides(i));
        }

        return indices;
    }
}

VectorXi discrete_indices(const DataFrame& df, const std::vector<std::string>& variables, const VectorXi& strides);

std::pair<VectorXi, VectorXi> create_cardinality_strides(const DataFrame& df,
                                                         const std::string& variable,
                                                         const std::vector<std::string>& evidence);
std::pair<VectorXi, VectorXi> create_cardinality_strides(const DataFrame& df,
                                                         const std::vector<std::string>& variables);

VectorXi joint_counts(const DataFrame& df,
                      const std::string& variable,
                      const std::vector<std::string>& evidence,
                      const VectorXi& cardinality,
                      const VectorXi& strides);

VectorXi marginal_counts(const VectorXi& joint_counts, int index, const VectorXi& cardinality, const VectorXi& strides);

std::vector<Array_ptr> discrete_slice_indices(const DataFrame& df,
                                              const std::vector<std::string>& discrete_vars,
                                              const VectorXi& indices,
                                              int num_factors);

void check_domain_variable(const DataFrame& df,
                           const std::string& variable,
                           const std::vector<std::string>& variable_values);

}  // namespace factors::discrete

#endif  // PYBNESIAN_FACTORS_DISCRETE_DISCRETE_INDICES_HPP