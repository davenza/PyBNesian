#include <models/DynamicBayesianNetwork.hpp>

namespace models {
    
    template<typename ArrowType>
    Array_ptr new_numeric_array(int length) {
        arrow::NumericBuilder<ArrowType> builder;
        RAISE_STATUS_ERROR(builder.Resize(length));

        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

        std::shared_ptr<ArrayType> out;
        RAISE_STATUS_ERROR(builder.Finish(&out));

        return out;
    }


    Array_ptr new_array(const LinearGaussianCPD& cpd, int length) {
        return new_numeric_array<arrow::DoubleType>(length);
    }

    Array_ptr new_array(const CKDE& cpd, int length) {
        switch (cpd.data_type()) {
            case Type::DOUBLE:
                return new_numeric_array<arrow::DoubleType>(length);
            case Type::FLOAT:
                return new_numeric_array<arrow::FloatType>(length);
            default:
                throw py::value_error("Wrong data type for CKDE.");
        }
    }

    Array_ptr new_array(const SemiparametricCPD& cpd, int length) {
        switch (cpd.factor_type()) {
            case FactorType::LinearGaussianCPD:
                return new_array(cpd.as_lg(), length);
            case FactorType::CKDE:
                return new_array(cpd.as_ckde(), length);
            default:
                throw py::value_error("Wrong factor type for SemiparametricCPD.");
        }

    }

    Array_ptr new_array(const DiscreteFactor& cpd, int length) {
        arrow::StringBuilder dict_builder;
        RAISE_STATUS_ERROR(dict_builder.AppendValues(cpd.variable_values()));

        std::shared_ptr<arrow::StringArray> dictionary;
        RAISE_STATUS_ERROR(dict_builder.Finish(&dictionary));

    }
}