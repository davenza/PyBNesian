#include <factors/discrete/discrete_indices.hpp>

namespace factors::discrete {

void check_is_string_dictionary(const std::shared_ptr<arrow::DictionaryArray>& dict, const std::string& variable) {
    if (dict->dictionary()->type_id() != Type::STRING) {
        throw std::invalid_argument(
            "The categories of the data must be of type string. The categories of the variable " + variable +
            " are of type " + dict->dictionary()->type()->ToString() + ".");
    }
}

void sum_to_discrete_indices_null(VectorXi& accum_indices,
                                  Array_ptr& indices,
                                  int stride,
                                  Buffer_ptr& combined_bitmap) {
    switch (indices->type_id()) {
        case Type::INT8:
            sum_to_discrete_indices_null<arrow::Int8Type>(accum_indices, indices, stride, combined_bitmap);
            break;
        case Type::INT16:
            sum_to_discrete_indices_null<arrow::Int16Type>(accum_indices, indices, stride, combined_bitmap);
            break;
        case Type::INT32:
            sum_to_discrete_indices_null<arrow::Int32Type>(accum_indices, indices, stride, combined_bitmap);
            break;
        case Type::INT64:
            sum_to_discrete_indices_null<arrow::Int64Type>(accum_indices, indices, stride, combined_bitmap);
            break;
        default:
            throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
    }
}

void sum_to_discrete_indices(VectorXi& accum_indices, Array_ptr& indices, int stride) {
    switch (indices->type_id()) {
        case Type::INT8:
            sum_to_discrete_indices<arrow::Int8Type>(accum_indices, indices, stride);
            break;
        case Type::INT16:
            sum_to_discrete_indices<arrow::Int16Type>(accum_indices, indices, stride);
            break;
        case Type::INT32:
            sum_to_discrete_indices<arrow::Int32Type>(accum_indices, indices, stride);
            break;
        case Type::INT64:
            sum_to_discrete_indices<arrow::Int64Type>(accum_indices, indices, stride);
            break;
        default:
            throw std::invalid_argument("Wrong indices array type of DictionaryArray.");
    }
}

VectorXi discrete_indices(const DataFrame& df,
                          const std::string& variable,
                          const std::vector<std::string>& evidence,
                          const VectorXi& strides) {
    if (df.null_count(variable, evidence) == 0)
        return discrete_indices<false>(df, variable, evidence, strides);
    else
        return discrete_indices<true>(df, variable, evidence, strides);
}

VectorXi discrete_indices(const DataFrame& df, const std::vector<std::string>& variables, const VectorXi& strides) {
    if (df.null_count(variables) == 0)
        return discrete_indices<false>(df, variables, strides);
    else
        return discrete_indices<true>(df, variables, strides);
}

std::pair<VectorXi, VectorXi> create_strides(const DataFrame& df,
                                             const std::string& variable,
                                             const std::vector<std::string>& evidence) {
    auto num_variables = evidence.size() + 1;

    VectorXi cardinality(num_variables);
    VectorXi strides(num_variables);

    auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));

    cardinality(0) = dict_variable->dictionary()->length();
    strides(0) = 1;

    for (size_t i = 1; i < num_variables; ++i) {
        auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(evidence[i - 1]));
        cardinality(i) = dict_evidence->dictionary()->length();
        strides(i) = strides(i - 1) * cardinality(i - 1);
    }

    return std::make_pair(cardinality, strides);
}

std::pair<VectorXi, VectorXi> create_cardinality_strides(const DataFrame& df,
                                                         const std::string& variable,
                                                         const std::vector<std::string>& evidence) {
    VectorXi cardinality(1 + evidence.size());
    VectorXi strides(1 + evidence.size());

    auto first_array = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));

    cardinality(0) = first_array->dictionary()->length();
    strides(0) = 1;

    for (size_t i = 0, i_end = evidence.size(); i < i_end; ++i) {
        auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(df.col(evidence[i]));
        cardinality(i + 1) = dict_array->dictionary()->length();
        strides(i + 1) = strides(i) * cardinality(i);
    }

    return std::make_pair(cardinality, strides);
}

std::pair<VectorXi, VectorXi> create_cardinality_strides(const DataFrame& df,
                                                         const std::vector<std::string>& variables) {
    if (variables.empty()) return std::make_pair(VectorXi(), VectorXi());

    VectorXi cardinality(variables.size());
    VectorXi strides(variables.size());

    auto first_array = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variables[0]));

    cardinality(0) = first_array->dictionary()->length();
    strides(0) = 1;

    for (size_t i = 1, i_end = variables.size(); i < i_end; ++i) {
        auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variables[i]));
        cardinality(i) = dict_array->dictionary()->length();
        strides(i) = strides(i - 1) * cardinality(i - 1);
    }

    return std::make_pair(cardinality, strides);
}

VectorXi joint_counts(const DataFrame& df,
                      const std::string& variable,
                      const std::vector<std::string>& evidence,
                      const VectorXi& cardinality,
                      const VectorXi& strides) {
    auto joint_values = cardinality.prod();

    VectorXi counts = VectorXi::Zero(joint_values);
    VectorXi indices = discrete_indices(df, variable, evidence, strides);

    // Compute counts
    for (auto i = 0; i < indices.rows(); ++i) {
        ++counts(indices(i));
    }

    return counts;
}

VectorXi marginal_counts(const VectorXi& joint_counts,
                         int index,
                         const VectorXi& cardinality,
                         const VectorXi& strides) {
    auto result = VectorXi::Zero(cardinality(index)).eval();

    auto stride = strides(index);
    auto card = cardinality(index);

    for (auto i = 0; i < joint_counts.rows(); ++i) {
        auto vindex = (i / stride) % card;
        result(vindex) += joint_counts(i);
    }

    return result;
}

std::vector<Array_ptr> discrete_slice_indices(const DataFrame& df,
                                              const std::vector<std::string>& discrete_vars,
                                              const VectorXi& strides,
                                              int num_factors) {
    arrow::NumericBuilder<arrow::Int32Type> builder;
    std::vector<Array_ptr> slices(num_factors);
    std::vector<std::vector<int>> slice_indices(num_factors);

    auto indices = discrete_indices(df, discrete_vars, strides);

    if (df.null_count(discrete_vars) == 0) {
        for (auto i = 0; i < indices.rows(); ++i) {
            slice_indices[indices(i)].push_back(i);
        }
    } else {
        auto bitmap = df.combined_bitmap(discrete_vars);
        auto bitmap_data = bitmap->data();

        for (auto i = 0, j = 0; i < df->num_rows(); ++i) {
            if (util::bit_util::GetBit(bitmap_data, i)) {
                slice_indices[indices(j++)].push_back(i);
            }
        }
    }

    for (auto i = 0; i < num_factors; ++i) {
        if (slice_indices[i].empty())
            slices[i] = nullptr;
        else {
            RAISE_STATUS_ERROR(builder.AppendValues(slice_indices[i].data(), slice_indices[i].size()));
            RAISE_STATUS_ERROR(builder.Finish(&slices[i]));
        }
    }

    return slices;
}

void check_domain_variable(const DataFrame& df,
                           const std::string& variable,
                           const std::vector<std::string>& variable_values) {
    auto var_array = df.col(variable);
    if (var_array->type_id() != arrow::Type::DICTIONARY)
        throw std::invalid_argument("Variable " + variable + " is not categorical.");

    auto var_dictionary = std::static_pointer_cast<arrow::DictionaryArray>(var_array);

    factors::discrete::check_is_string_dictionary(var_dictionary, variable);
    auto var_names = std::static_pointer_cast<arrow::StringArray>(var_dictionary->dictionary());

    if (variable_values.size() != static_cast<size_t>(var_names->length()))
        throw std::invalid_argument("Variable " + variable + " does not contain the same categories.");

    for (auto j = 0; j < var_names->length(); ++j) {
        if (variable_values[j] != var_names->GetString(j))
            throw std::invalid_argument("Category at index " + std::to_string(j) + " is different for variable " +
                                        variable);
    }
}

}  // namespace factors::discrete