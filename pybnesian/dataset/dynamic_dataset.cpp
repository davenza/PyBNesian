#include <dataset/dynamic_dataset.hpp>
#include <util/util_types.hpp>

using util::ArcStringVector;

namespace dataset {

std::string index_to_string(const DynamicVariable<int>& index) {
    return "(" + std::to_string(index.variable) + ", " + std::to_string(index.temporal_slice) + ")";
}

std::string index_to_string(const DynamicVariable<std::string>& index) {
    return "(" + index.variable + ", " + std::to_string(index.temporal_slice) + ")";
}

DataFrame create_temporal_slice(const DataFrame& df, int slice_index, int slice_offset, int markovian_order) {
    int new_length = df->num_rows() - markovian_order;
    int offset = markovian_order - slice_index;
    auto slice_df = df->Slice(offset, new_length);

    auto columns = slice_df->columns();

    Field_vector new_fields;
    new_fields.reserve(slice_df->num_columns());

    for (auto field : df->schema()->fields()) {
        new_fields.push_back(field->WithName(util::temporal_name(field->name(), slice_index + slice_offset)));
    }

    auto new_schema = arrow::schema(new_fields);

    return DataFrame(arrow::RecordBatch::Make(new_schema, new_length, columns));
}

std::vector<DataFrame> create_temporal_slices(const DataFrame& df, int markovian_order) {
    std::vector<DataFrame> temporal_slices;

    for (auto i = 0; i <= markovian_order; ++i) {
        temporal_slices.push_back(create_temporal_slice(df, i, 0, markovian_order));
    }

    return temporal_slices;
}

DataFrame create_static_df(const DataFrame& df, int markovian_order) {
    if (markovian_order == 1) {
        Field_vector new_fields;
        new_fields.reserve(df->num_columns());

        for (auto field : df->schema()->fields()) {
            new_fields.push_back(field->WithName(util::temporal_name(field->name(), 1)));
        }

        auto new_schema = arrow::schema(new_fields);

        return DataFrame(arrow::RecordBatch::Make(new_schema, df->num_rows(), df->columns()));
    }

    Array_vector columns;
    Field_vector new_fields;
    columns.reserve(df->num_columns() * markovian_order);
    new_fields.reserve(df->num_columns() * markovian_order);

    for (auto i = 0; i < markovian_order; ++i) {
        auto new_slice = create_temporal_slice(df, i, 1, markovian_order - 1);
        append_slice(new_slice, columns, new_fields);
    }

    auto new_schema = arrow::schema(new_fields);
    return DataFrame(arrow::RecordBatch::Make(new_schema, columns[0]->length(), columns));
}

DataFrame create_transition_df(std::vector<DataFrame>& v, int markovian_order) {
    Array_vector columns;
    Field_vector new_fields;
    columns.reserve(v[0]->num_columns() * (markovian_order + 1));
    new_fields.reserve(v[0]->num_columns() * (markovian_order + 1));

    for (const auto& s : v) {
        append_slice(s, columns, new_fields);
    }

    auto new_schema = arrow::schema(new_fields);
    return DataFrame(arrow::RecordBatch::Make(new_schema, v[0]->num_rows(), columns));
}

void append_slice(const DataFrame& slice, Array_vector& columns, Field_vector& fields) {
    for (auto field : slice->schema()->fields()) {
        fields.push_back(field);
    }

    for (auto column : slice->columns()) {
        columns.push_back(column);
    }
}

void append_slice(const std::vector<DataFrame>& slices,
                  Array_vector& columns,
                  Field_vector& fields,
                  int markovian_order,
                  int slice_index) {
    if (slice_index < 0 || slice_index > markovian_order) {
        throw std::invalid_argument("slice_index must be an index between 0 and " + std::to_string(markovian_order));
    }

    append_slice(slices[slice_index], columns, fields);
}

}  // namespace dataset