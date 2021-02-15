#include <models/DynamicBayesianNetwork.hpp>

namespace models {
    
    template<typename ArrowType>
    Array_ptr new_numeric_array(int length) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        arrow::NumericBuilder<ArrowType> builder;
        RAISE_STATUS_ERROR(builder.Resize(length));

        RAISE_STATUS_ERROR(builder.AppendEmptyValues(length));

        std::shared_ptr<ArrayType> out;
        RAISE_STATUS_ERROR(builder.Finish(&out));
        return out;
    }

    Array_ptr new_numeric_array(arrow::Type::type t, int length) {
        switch (t) {
            case Type::DOUBLE:
                return new_numeric_array<arrow::DoubleType>(length);
            case Type::FLOAT:
                return new_numeric_array<arrow::FloatType>(length);
            default:
                throw std::invalid_argument("ArrowType " + std::to_string(t) + " not valid for numeric array.");
        }
    }

    Array_ptr new_discrete_array(const std::vector<std::string>& values, int length) {
        arrow::StringBuilder dict_builder;
        RAISE_STATUS_ERROR(dict_builder.AppendValues(values));

        std::shared_ptr<arrow::StringArray> dictionary;
        RAISE_STATUS_ERROR(dict_builder.Finish(&dictionary));

        arrow::DictionaryBuilder<arrow::StringType> builder(dictionary);
        RAISE_STATUS_ERROR(builder.Resize(length));

        RAISE_STATUS_ERROR(builder.AppendEmptyValues(length));

        std::shared_ptr<arrow::DictionaryArray> out;
        RAISE_STATUS_ERROR(builder.Finish(&out));

        return out;
    }

    std::vector<std::string> conditional_topological_sort(const ConditionalDag& dag) {
        auto top_sort = dag.topological_sort();

        std::vector<std::string> filtered_sort;
        filtered_sort.reserve(dag.num_nodes());

        for (const auto& v : top_sort) {
            if (dag.contains_node(v)) {
                filtered_sort.push_back(v);
            }
        }
        
        return filtered_sort;
    }

}