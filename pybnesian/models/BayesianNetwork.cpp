#include <models/BayesianNetwork.hpp>

namespace models {

void requires_continuous_data(const DataFrame& df) {
    auto schema = df->schema();

    if (schema->num_fields() == 0) {
        throw std::invalid_argument("Provided dataset does not contain columns.");
    }

    auto dtid = schema->field(0)->type()->id();

    if (dtid != Type::DOUBLE && dtid != Type::FLOAT) {
        throw std::invalid_argument(
            "Continuous data (double or float) is needed to learn Gaussian networks. "
            "Column \"" +
            schema->field(0)->name() + "\" (DataType: " + schema->field(0)->type()->ToString() + ").");
    }

    for (auto i = 1; i < schema->num_fields(); ++i) {
        auto new_dtid = schema->field(i)->type()->id();
        if (dtid != new_dtid)
            throw std::invalid_argument(
                "All the columns should have the same data type. "
                "Column \"" +
                schema->field(0)->name() + "\" (DataType: " + schema->field(0)->type()->ToString() +
                "). "
                "Column \"" +
                schema->field(i)->name() + "\" (DataType: " + schema->field(i)->type()->ToString() + ").");
    }
}

void requires_discrete_data(const DataFrame& df) {
    auto schema = df->schema();

    if (schema->num_fields() == 0) {
        throw std::invalid_argument("Provided dataset does not contain columns.");
    }

    for (auto i = 0; i < schema->num_fields(); ++i) {
        auto dtid = schema->field(i)->type()->id();
        if (dtid != Type::DICTIONARY)
            throw std::invalid_argument(
                "Categorical data is needed to learn discrete Bayesian networks. "
                "Column \"" +
                schema->field(i)->name() + "\" (DataType: " + schema->field(i)->type()->ToString() + ").");
    }
}

DataFrame ConditionalBayesianNetwork::sample(const DataFrame& evidence,
                                             unsigned int seed,
                                             bool concat_evidence,
                                             bool ordered) const {
    this->check_fitted();
    evidence.raise_has_columns(interface_nodes());

    DataFrame parents(evidence);

    auto top_sort = this->g.topological_sort();
    for (size_t i = 0; i < top_sort.size(); ++i) {
        auto idx = this->index(top_sort[i]);
        auto array = this->m_cpds[idx]->sample(evidence->num_rows(), parents, seed + i);

        auto res = parents->AddColumn(evidence->num_columns() + i, top_sort[i], array);
        parents = DataFrame(std::move(res).ValueOrDie());
    }

    std::vector<Field_ptr> fields;
    std::vector<Array_ptr> columns;

    auto schema = parents->schema();
    if (ordered) {
        for (const auto& name : this->nodes()) {
            fields.push_back(schema->GetFieldByName(name));
            columns.push_back(parents.col(name));
        }
    } else {
        for (auto i = evidence->num_columns(); i < parents->num_columns(); ++i) {
            fields.push_back(schema->field(i));
            columns.push_back(parents.col(i));
        }
    }

    if (concat_evidence) {
        auto evidence_schema = evidence->schema();
        for (auto i = 0; i < evidence->num_columns(); ++i) {
            fields.push_back(evidence_schema->field(i));
            columns.push_back(evidence.col(i));
        }
    }

    auto new_schema = std::make_shared<arrow::Schema>(fields);
    auto new_rb = arrow::RecordBatch::Make(new_schema, evidence->num_rows(), columns);
    return DataFrame(new_rb);
}

}  // namespace models