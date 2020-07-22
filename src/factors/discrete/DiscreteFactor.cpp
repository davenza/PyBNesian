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

        m_values = params.values;
        m_cardinality = params.cardinality;
        m_strides = params.strides;
    }

}