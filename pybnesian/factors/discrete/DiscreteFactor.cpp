#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <factors/discrete/DiscreteFactor.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/parameters/mle_base.hpp>
#include <util/math_constants.hpp>
#include <fort.hpp>

using dataset::DataFrame;
using learning::parameters::MLE;

using DataType_ptr = std::shared_ptr<arrow::DataType>;

using models::BayesianNetworkBase, models::ConditionalBayesianNetworkBase;

namespace factors::discrete {

std::shared_ptr<Factor> DiscreteFactorType::new_factor(const BayesianNetworkBase&,
                                                       const std::string& variable,
                                                       const std::vector<std::string>& evidence,
                                                       py::args args,
                                                       py::kwargs kwargs) const {
    return generic_new_factor<DiscreteFactor>(variable, evidence, args, kwargs);
}

std::shared_ptr<Factor> DiscreteFactorType::new_factor(const ConditionalBayesianNetworkBase&,
                                                       const std::string& variable,
                                                       const std::vector<std::string>& evidence,
                                                       py::args args,
                                                       py::kwargs kwargs) const {
    return generic_new_factor<DiscreteFactor>(variable, evidence, args, kwargs);
}

void DiscreteFactor::fit(const DataFrame& df) {
    MLE<DiscreteFactor> mle;

    auto params = mle.estimate(df, variable(), evidence());

    m_logprob = params.logprob;
    m_cardinality = params.cardinality;
    m_strides = VectorXi(m_cardinality.rows());
    m_strides(0) = 1;
    for (size_t i = 1, i_end = static_cast<size_t>(m_strides.rows()); i < i_end; ++i) {
        m_strides(i) = m_strides(i - 1) * m_cardinality(i - 1);
    }

    auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable()));

    factors::discrete::check_is_string_dictionary(dict_variable, variable());
    auto dict_variable_values = std::static_pointer_cast<arrow::StringArray>(dict_variable->dictionary());

    m_variable_values.clear();
    m_variable_values.reserve(dict_variable_values->length());
    for (auto i = 0; i < dict_variable_values->length(); ++i) {
        m_variable_values.push_back(dict_variable_values->GetString(i));
    }

    m_evidence_values.clear();
    m_evidence_values.reserve(evidence().size());
    for (auto it = evidence().begin(), end = evidence().end(); it != end; ++it) {
        auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));

        factors::discrete::check_is_string_dictionary(dict_evidence, *it);
        auto dict_evidence_values = std::static_pointer_cast<arrow::StringArray>(dict_evidence->dictionary());

        std::vector<std::string> ev;
        ev.reserve(dict_evidence_values->length());
        for (auto j = 0; j < dict_evidence_values->length(); ++j) {
            ev.push_back(dict_evidence_values->GetString(j));
        }

        m_evidence_values.push_back(ev);
    }

    m_fitted = true;
}

void DiscreteFactor::check_equal_domain(const DataFrame& df, bool check_variable) const {
    if (check_variable) {
        df.raise_has_column(variable());
        check_domain_variable(df, variable(), m_variable_values);
    }

    df.raise_has_columns(evidence());
    int i = 0;
    for (auto it = evidence().begin(); it != evidence().end(); ++it, ++i) {
        check_domain_variable(df, *it, m_evidence_values[i]);
    }
}

VectorXd DiscreteFactor::_logl_null(const DataFrame& df) const {
    VectorXi indices = discrete_indices<true>(df);

    auto evidence_pair = std::make_pair(evidence().begin(), evidence().end());
    auto combined_bitmap = df.combined_bitmap(variable(), evidence_pair);
    auto* bitmap_data = combined_bitmap->data();

    VectorXd res(df->num_rows());
    for (auto i = 0, j = 0; i < indices.rows(); ++i) {
        if (util::bit_util::GetBit(bitmap_data, i))
            res(i) = m_logprob(indices(j++));
        else
            res(i) = util::nan<double>;
    }

    return res;
}

VectorXd DiscreteFactor::_logl(const DataFrame& df) const {
    VectorXi indices = discrete_indices<false>(df);

    VectorXd res(df->num_rows());

    for (auto i = 0; i < indices.rows(); ++i) {
        res(i) = m_logprob(indices(i));
    }

    return res;
}

VectorXd DiscreteFactor::logl(const DataFrame& df) const {
    run_checks(df, true);
    auto evidence_pair = std::make_pair(evidence().begin(), evidence().end());
    bool contains_null = df.null_count(variable(), evidence_pair) > 0;

    if (!contains_null) {
        return _logl(df);
    } else {
        return _logl_null(df);
    }
}

double DiscreteFactor::_slogl_null(const DataFrame& df) const {
    VectorXi indices = discrete_indices<true>(df);

    auto evidence_pair = std::make_pair(evidence().begin(), evidence().end());
    auto combined_bitmap = df.combined_bitmap(variable(), evidence_pair);
    auto* bitmap_data = combined_bitmap->data();

    double res = 0;
    for (auto i = 0, j = 0; i < indices.rows(); ++i) {
        if (util::bit_util::GetBit(bitmap_data, i)) res += m_logprob(indices(j++));
    }

    return res;
}

double DiscreteFactor::_slogl(const DataFrame& df) const {
    VectorXi indices = discrete_indices<false>(df);

    double res = 0;

    for (auto i = 0; i < indices.rows(); ++i) {
        res += m_logprob(indices(i));
    }

    return res;
}

double DiscreteFactor::slogl(const DataFrame& df) const {
    run_checks(df, true);

    auto evidence_pair = std::make_pair(evidence().begin(), evidence().end());
    bool contains_null = df.null_count(variable(), evidence_pair) > 0;

    if (!contains_null) {
        return _slogl(df);
    } else {
        return _slogl_null(df);
    }
}

Array_ptr DiscreteFactor::sample(int n, const DataFrame& evidence_values, unsigned int seed) const {
    if (n < 0) {
        throw std::invalid_argument("n should be a non-negative number");
    }

    run_checks(evidence_values, false);

    arrow::StringBuilder dict_builder;
    RAISE_STATUS_ERROR(dict_builder.AppendValues(m_variable_values));

    std::shared_ptr<arrow::StringArray> dictionary;
    RAISE_STATUS_ERROR(dict_builder.Finish(&dictionary));

    Array_ptr indices;
    DataType_ptr type = data_type();
    auto dwn_type = std::static_pointer_cast<arrow::DictionaryType>(type);

    switch (dwn_type->index_type()->id()) {
        case Type::INT8:
            indices = sample_indices<arrow::Int8Type>(n, evidence_values, seed);
            break;
        case Type::INT16:
            indices = sample_indices<arrow::Int16Type>(n, evidence_values, seed);
            break;
        case Type::INT32:
            indices = sample_indices<arrow::Int32Type>(n, evidence_values, seed);
            break;
        case Type::INT64:
            indices = sample_indices<arrow::Int64Type>(n, evidence_values, seed);
            break;
        default:
            throw std::runtime_error("Wrong index type! This code should be unreachable.");
    }

    return std::make_shared<arrow::DictionaryArray>(type, indices, dictionary);
}

std::string DiscreteFactor::ToString() const {
    std::stringstream stream;
    stream << std::setprecision(3);
    if (!evidence().empty()) {
        const auto& e = evidence();
        stream << "[DiscreteFactor] P(" << variable() << " | " << e[0];

        for (size_t i = 1; i < e.size(); ++i) {
            stream << ", " << e[i];
        }
        stream << ")";

        if (m_fitted) {
            stream << std::endl;

            fort::char_table table;
            table.set_cell_text_align(fort::text_align::center);

            table << std::setprecision(3) << fort::header;
            table[0][0].set_cell_span(e.size());
            table[0][e.size()] = variable();
            table[0][e.size()].set_cell_span(m_cardinality(0));
            table << fort::endr << fort::header;
            table.range_write(e.begin(), e.end());
            table.range_write_ln(m_variable_values.begin(), m_variable_values.end());

            auto parent_configurations = m_cardinality.bottomRows(e.size()).prod();

            for (auto k = 0; k < parent_configurations; ++k) {
                double index = k * m_cardinality(0);
                for (size_t j = 0; j < e.size(); ++j) {
                    auto assignment_index =
                        static_cast<int>(std::floor(index / m_strides(j + 1))) % m_cardinality(j + 1);
                    table << m_evidence_values[j][assignment_index];
                }

                for (auto i = 0; i < m_cardinality(0); ++i) {
                    table << std::exp(m_logprob(index + i));
                }
                table << fort::endr;
            }

            stream << table.to_string();
        } else {
            stream << " not fitted.";
        }
    } else {
        stream << "[DiscreteFactor] P(" << variable() << ")";

        if (m_fitted) {
            stream << std::endl;
            fort::char_table table;
            table << std::setprecision(3) << fort::header << variable() << fort::endr << fort::header;
            table[0][0].set_cell_span(m_cardinality(0));
            table[0][0].set_cell_text_align(fort::text_align::center);
            table.range_write_ln(m_variable_values.begin(), m_variable_values.end());
            table.row(1).set_cell_text_align(fort::text_align::center);

            for (auto i = 0; i < m_cardinality(0); ++i) {
                table << std::exp(m_logprob(i));
            }
            table << fort::endr;
            stream << table.to_string();
        } else {
            stream << " not fitted.";
        }
    }

    return stream.str();
}

py::tuple DiscreteFactor::__getstate__() const {
    std::vector<std::string> variable_values;
    std::vector<std::vector<std::string>> evidence_values;
    VectorXd logprob;

    if (m_fitted) {
        variable_values = m_variable_values;
        evidence_values = m_evidence_values;
        logprob = m_logprob;
    }

    return py::make_tuple(variable(), evidence(), m_fitted, variable_values, evidence_values, logprob);
}

DiscreteFactor DiscreteFactor::__setstate__(py::tuple& t) {
    if (t.size() != 6) throw std::runtime_error("Not valid DiscreteFactor.");

    DiscreteFactor dist(t[0].cast<std::string>(), t[1].cast<std::vector<std::string>>());

    dist.m_fitted = t[2].cast<bool>();

    if (dist.m_fitted) {
        dist.m_variable_values = t[3].cast<std::vector<std::string>>();
        dist.m_evidence_values = t[4].cast<std::vector<std::vector<std::string>>>();
        dist.m_logprob = t[5].cast<VectorXd>();

        VectorXi cardinality(dist.evidence().size() + 1);
        VectorXi strides(dist.evidence().size() + 1);

        cardinality(0) = dist.m_variable_values.size();
        strides(0) = 1;

        int i = 1;
        for (auto it = dist.m_evidence_values.begin(), end = dist.m_evidence_values.end(); it != end; ++it, ++i) {
            cardinality(i) = it->size();
            strides(i) = strides(i - 1) * cardinality(i - 1);
        }

        dist.m_cardinality = std::move(cardinality);
        dist.m_strides = std::move(strides);
    }

    return dist;
}

}  // namespace factors::discrete