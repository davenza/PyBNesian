#include <iostream>
#include <factors/discrete/DiscreteFactor.hpp>
#include <learning/parameters/mle.hpp>

using dataset::DataFrame;
using learning::parameters::MLE;

namespace factors::discrete {

    void process_dictionary(std::shared_ptr<arrow::Array>& d) {
        auto dwn = std::static_pointer_cast<arrow::StringArray>(d);

        auto num_values = dwn->length();

        for (auto i = 0; i < num_values; ++i) {
            std::cout << "Dictionary value: " << dwn->GetString(i) << std::endl;
        }
    }

    void process_indices(std::shared_ptr<arrow::Array>& a, int dictionary_length) {
        auto dwn = std::static_pointer_cast<arrow::Int8Array>(a);


        std::cout << a->type()->ToString() << std::endl;
        std::cout << a->type_id() << std::endl;

        std::vector<int> counts(dictionary_length);

        auto raw_values = dwn->raw_values();
        for (auto i = 0; i < dwn->length(); ++i) {
            ++counts[raw_values[i]];
        }

        std::cout << "Count values: ";
        for (auto i = 0; i < dictionary_length; ++i) {
            std::cout << counts[i] << " ";
        }
        std::cout << std::endl;
    }

    void DiscreteFactor::fit(const DataFrame& df) {
        
        MLE<DiscreteFactor> mle;

        auto params = mle.estimate(df, m_variable, m_evidence.begin(), m_evidence.end());

        m_prob = params.prob;
        m_cardinality = params.cardinality;
        m_strides = params.strides;

        auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(m_variable));
        auto dict_variable_values = std::static_pointer_cast<arrow::StringArray>(dict_variable->dictionary());

        std::vector<std::string> variable_values;
        variable_values.reserve(dict_variable_values->length());

        for (auto i = 0; i < dict_variable_values->length(); ++i) {
            variable_values.push_back(dict_variable_values->GetString(i));
        }

        m_variable_values = variable_values;

        std::vector<std::vector<std::string>> evidence_values;
        evidence_values.reserve(m_evidence.size());

        for (auto it = m_evidence.begin(); it != m_evidence.end(); ++it) {
            auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
            auto dict_evidence_values = std::static_pointer_cast<arrow::StringArray>(dict_evidence->dictionary());

            std::vector<std::string> ev;
            ev.reserve(dict_evidence->length());
            for (auto j = 0; j < dict_evidence_values->length(); ++j) {
                ev.push_back(dict_evidence_values->GetString(j));
            }

            evidence_values.push_back(ev);
        }

        m_evidence_values = evidence_values;
    }

}