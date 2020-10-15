#include <factors/discrete/DiscreteFactor.hpp>
#include <learning/parameters/mle.hpp>
#include <util/math_constants.hpp>
#include <fort.hpp>

using dataset::DataFrame;
using learning::parameters::MLE;

using DataType_ptr = std::shared_ptr<arrow::DataType>;

namespace factors::discrete {

    void sum_to_discrete_indices_null(VectorXi& accum_indices, Array_ptr& indices, int stride, Buffer_ptr& combined_bitmap) {
        switch(indices->type_id()) {
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
        switch(indices->type_id()) {
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

    void DiscreteFactor::fit(const DataFrame& df) {
        
        MLE<DiscreteFactor> mle;

        auto params = mle.estimate(df, m_variable, m_evidence.begin(), m_evidence.end());

        m_logprob = params.logprob;
        m_cardinality = params.cardinality;
        m_strides = params.strides;

        auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(m_variable));
        auto dict_variable_values = std::static_pointer_cast<arrow::StringArray>(dict_variable->dictionary());

        m_variable_values.reserve(dict_variable_values->length());
        for (auto i = 0; i < dict_variable_values->length(); ++i) {
            m_variable_values.push_back(dict_variable_values->GetString(i));
        }

        m_evidence_values.reserve(m_evidence.size());

        for (auto it = m_evidence.begin(), end = m_evidence.end(); it != end; ++it) {
            auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
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

    void check_domain_variable(const DataFrame& df, 
                               const std::string& variable, 
                               const std::vector<std::string>& variable_values) {
        auto var_array = df.col(variable);
        if (var_array->type_id() != arrow::Type::DICTIONARY)
            throw std::invalid_argument("Variable " + variable + " is not categorical.");

        auto var_dictionary = std::static_pointer_cast<arrow::DictionaryArray>(var_array)->dictionary();
        auto var_names = std::static_pointer_cast<arrow::StringArray>(var_dictionary);

        if (variable_values.size() != static_cast<size_t>(var_names->length())) 
            throw std::invalid_argument("Variable " + variable + " does not contain the same categories.");

        for (auto j = 0; j < var_names->length(); ++j) {
            if (variable_values[j] != var_names->GetString(j))
                throw std::invalid_argument("Category at index " + std::to_string(j) + " is different for variable " + variable);
        }
    }

    void DiscreteFactor::check_equal_domain(const DataFrame& df) const {
        check_domain_variable(df, m_variable, m_variable_values);
        int i = 0;
        for (auto it = m_evidence.begin(); it != m_evidence.end(); ++it, ++i) {
            check_domain_variable(df, *it, m_evidence_values[i]);
        }
    }

    VectorXd DiscreteFactor::_logl_null(const DataFrame& df) const {
        VectorXi indices = discrete_indices<true>(df);

        auto evidence_pair = std::make_pair(m_evidence.begin(), m_evidence.end());
        auto combined_bitmap = df.combined_bitmap(m_variable, evidence_pair);
        auto* bitmap_data = combined_bitmap->data();

        VectorXd res(df->num_rows());
        for (auto i = 0, j = 0; i < indices.rows(); ++i) {
            if (arrow::BitUtil::GetBit(bitmap_data, i))
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


    VectorXd DiscreteFactor::logl(const DataFrame& df, bool check_domain) const {
        if (check_domain)
            check_equal_domain(df);

        auto evidence_pair = std::make_pair(m_evidence.begin(), m_evidence.end());
        bool contains_null = df.null_count(m_variable, evidence_pair) > 0;

        if (!contains_null) {
            return _logl(df);
        } else {
            return _logl_null(df);
        }
    }


    double DiscreteFactor::_slogl_null(const DataFrame& df) const {
        VectorXi indices = discrete_indices<true>(df);

        auto evidence_pair = std::make_pair(m_evidence.begin(), m_evidence.end());
        auto combined_bitmap = df.combined_bitmap(m_variable, evidence_pair);
        auto* bitmap_data = combined_bitmap->data();

        double res = 0;
        for (auto i = 0, j = 0; i < indices.rows(); ++i) {
            if (arrow::BitUtil::GetBit(bitmap_data, i))
                res += m_logprob(indices(j++));
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

    double DiscreteFactor::slogl(const DataFrame& df, bool check_domain) const {
        if (check_domain)
            check_equal_domain(df);

        auto evidence_pair = std::make_pair(m_evidence.begin(), m_evidence.end());
        bool contains_null = df.null_count(m_variable, evidence_pair) > 0;

        if (!contains_null) {
            return _slogl(df);
        } else {
            return _slogl_null(df);
        }
    }

    Array_ptr DiscreteFactor::sample(int n, const DataFrame& evidence_values, 
                                     long unsigned int seed) const {
        arrow::StringBuilder dict_builder;
        RAISE_STATUS_ERROR(dict_builder.AppendValues(m_variable_values));


        std::shared_ptr<arrow::StringArray> dictionary;
        RAISE_STATUS_ERROR(dict_builder.Finish(&dictionary));

        Array_ptr indices;
        DataType_ptr type;
        if (m_variable_values.size() <= std::numeric_limits<typename arrow::Int8Type::c_type>::max()) {
            type = arrow::dictionary(arrow::int8(), arrow::utf8());
            indices = sample_indices<arrow::Int8Type>(n, evidence_values, seed);
        } else if (m_variable_values.size() <= std::numeric_limits<typename arrow::Int16Type::c_type>::max()) {
            type = arrow::dictionary(arrow::int16(), arrow::utf8());
            indices = sample_indices<arrow::Int16Type>(n, evidence_values, seed);
        } else if (m_variable_values.size() <= std::numeric_limits<typename arrow::Int32Type::c_type>::max()) {
            type = arrow::dictionary(arrow::int32(), arrow::utf8());
            indices = sample_indices<arrow::Int32Type>(n, evidence_values, seed);
        } else {
            type = arrow::dictionary(arrow::int64(), arrow::utf8());
            indices = sample_indices<arrow::Int64Type>(n, evidence_values, seed);
        }

        return std::make_shared<arrow::DictionaryArray>(type, indices, dictionary);
    }

    std::string DiscreteFactor::ToString() const {
        std::stringstream stream;
        stream << std::setprecision(3);
        if (!m_evidence.empty()) {
            stream << "[DiscreteFactor] P(" << m_variable << " | " << m_evidence[0];
            for (size_t i = 1; i < m_evidence.size(); ++i) {
                stream << ", " << m_evidence[i];
            }
        
            stream << ")" << std::endl;

            fort::char_table table;                                                
            table.set_cell_text_align(fort::text_align::center);

            table << std::setprecision(3) << fort::header;
            table[0][0].set_cell_span(m_evidence.size());
            table[0][2] = m_variable;
            table[0][2].set_cell_span(m_cardinality(0));
            table << fort::endr << fort::header;
            table.range_write(m_evidence.begin(), m_evidence.end());
            table.range_write_ln(m_variable_values.begin(), m_variable_values.end());

            auto parent_configurations = m_cardinality.bottomRows(m_evidence.size()).prod();

            for (auto k = 0; k < parent_configurations; ++k) {
                double index = k*m_cardinality(0);
                for (size_t j = 0; j < m_evidence.size(); ++j) {
                    auto assignment_index = static_cast<int>(std::floor(index / m_strides(j+1))) 
                                                                % m_cardinality(j+1);
                    table << m_evidence_values[j][assignment_index];
                }

                for (auto i = 0; i < m_cardinality(0); ++i) {
                    table << std::exp(m_logprob(index + i));
                }
                table << fort::endr;
            }

            stream << table.to_string();
        } else {
            stream << "[DiscreteFactor] P(" << m_variable << ")" << std::endl;

            fort::char_table table;                                                
            table << std::setprecision(3) << fort::header                                             
                << m_variable << fort::endr
                << fort::header;
            table[0][0].set_cell_span(m_cardinality(0));
            table[0][0].set_cell_text_align(fort::text_align::center);
            table.range_write_ln(m_variable_values.begin(), m_variable_values.end());
            table.row(1).set_cell_text_align(fort::text_align::center);

            for (auto i = 0; i < m_cardinality(0); ++i) {
                table << std::exp(m_logprob(i));
            }
            table << fort::endr;

            stream << table.to_string();
        }

        return stream.str();
    }
}