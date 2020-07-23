#ifndef PGM_DATASET_MLE_DISCRETEFACTOR_HPP
#define PGM_DATASET_MLE_DISCRETEFACTOR_HPP

#include <factors/discrete/DiscreteFactor.hpp>
#include <learning/parameters/mle.hpp>
#include <util/bit_util.hpp>

using arrow::Array;
using factors::discrete::DiscreteFactor;

using Array_ptr = std::shared_ptr<arrow::Array>;
using Buffer_ptr = std::shared_ptr<arrow::Buffer>;

// using learning::parameters::MLE;


namespace learning::parameters {

    template<typename ArrowType>
    void sum_indices_null(VectorXi& accum_indices, Array_ptr& indices, int stride, Buffer_ptr& combined_bitmap) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        auto dwn_indices = std::static_pointer_cast<ArrayType>(indices);

        auto raw_combined_bitmap = combined_bitmap->data();
        for (auto i = 0, j = 0; i < indices->length(); ++i) {
            if (arrow::BitUtil::GetBit(raw_combined_bitmap, i)) {
                accum_indices(j++) += dwn_indices->Value(i) * stride;
            }
        }
    }

    void sum_indices_null(VectorXi& accum_indices, Array_ptr& indices, int stride, Buffer_ptr& combined_bitmap);

    template<typename VarType, typename EvidenceIter>
    VectorXd _joint_counts_null(const DataFrame& df, 
                                const VarType& variable, 
                                EvidenceIter evidence_begin, 
                                EvidenceIter evidence_end,
                                VectorXi& cardinality,
                                VectorXi& strides) {
        auto joint_values = cardinality.prod();
        VectorXd counts = VectorXd::Zero(joint_values);

        auto combined_bitmap = df.combined_bitmap(variable, std::make_pair(evidence_begin, evidence_end));

        auto valid_rows = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

        VectorXi indices = VectorXi::Zero(valid_rows);

        auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));
        auto variable_indices = dict_variable->indices();

        sum_indices_null(indices, variable_indices, strides(0), combined_bitmap);

        int i = 1;
        for (auto it = evidence_begin; it != evidence_end; ++it, ++i) {
            auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
            auto evidence_indices = dict_evidence->indices();
            sum_indices_null(indices, evidence_indices, strides(i), combined_bitmap);
        }

        // Compute counts
        for (auto i = 0; i < indices.rows(); ++i) {
            ++counts(indices(i));
        }

        return counts;
    }

    template<typename ArrowType>
    void sum_indices(VectorXi& accum_indices, Array_ptr& indices, int stride) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using EigenMap = Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>;
        auto dwn_indices = std::static_pointer_cast<ArrayType>(indices);
        auto* raw_values = dwn_indices->raw_values();
        const EigenMap map_eigen(raw_values, indices->length());
        accum_indices += (map_eigen * stride).template cast<int>();
    }

    void sum_indices(VectorXi& accum_indices, Array_ptr& indices, int stride);

    template<typename VarType, typename EvidenceIter>
    VectorXd _joint_counts(const DataFrame& df, 
                                const VarType& variable, 
                                EvidenceIter evidence_begin, 
                                EvidenceIter evidence_end,
                                VectorXi& cardinality,
                                VectorXi& strides) {
        auto joint_values = cardinality.prod();

        VectorXd counts = VectorXd::Zero(joint_values);

        VectorXi indices = VectorXi::Zero(df->num_rows());

        auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));
        auto variable_indices = dict_variable->indices();

        sum_indices(indices, variable_indices, strides(0));

        int i = 1;
        for (auto it = evidence_begin; it != evidence_end; ++it, ++i) {
            auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
            auto evidence_indices = dict_evidence->indices();
            sum_indices(indices, evidence_indices, strides(i));
        }

        // Compute counts
        for (auto i = 0; i < indices.rows(); ++i) {
            ++counts(indices(i));
        }

        return counts;
    }
    

    template<bool contains_null, typename VarType, typename EvidenceIter>
    typename DiscreteFactor::ParamsClass _fit(const DataFrame& df, 
                                                const VarType& variable, 
                                                EvidenceIter evidence_begin, 
                                                EvidenceIter evidence_end) {
        
        auto num_variables = std::distance(evidence_begin, evidence_end) + 1;

        VectorXi cardinality(num_variables);
        VectorXi strides(num_variables);

        auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));
        
        cardinality(0) = dict_variable->dictionary()->length();
        strides(0) = 1;

        int i = 1;
        for(auto it = evidence_begin; it != evidence_end; ++it, ++i) {
            auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
            cardinality(i) = dict_evidence->dictionary()->length();
            strides(i) = strides(i-1)*cardinality(i-1);            
        }

        auto prob = [&df, &variable, evidence_begin, evidence_end, &cardinality, &strides]() {
            if constexpr (contains_null) {
                return _joint_counts_null(df, variable, evidence_begin, evidence_end, cardinality, strides);
            } else {
                return _joint_counts(df, variable, evidence_begin, evidence_end, cardinality, strides);
            }
        }();

        // Normalize the CPD.
        auto parent_configurations = cardinality.bottomRows(num_variables-1).prod();

        for (auto k = 0; k < parent_configurations; ++k) {
            auto offset = k*cardinality(0);

            double sum_configuration = 0;
            for(auto i = 0; i < cardinality(0); ++i) {
                sum_configuration += prob(offset + i);
            }

            if (sum_configuration == 0) {
                auto uniform = 1. / cardinality(0);
                for(auto i = 0; i < cardinality(0); ++i) {
                    prob(offset + i) = uniform;
                }       
            } else {
                for(auto i = 0; i < cardinality(0); ++i) {
                    prob(offset + i) /= sum_configuration;
                }       
            }
        }

        return typename DiscreteFactor::ParamsClass {
            .prob = prob,
            .cardinality = cardinality,
            .strides = strides
        };
    }


    template<>
    template<typename VarType, typename EvidenceIter>
    typename DiscreteFactor::ParamsClass MLE<DiscreteFactor>::estimate(const DataFrame& df, 
                                                                        const VarType& variable, 
                                                                        EvidenceIter evidence_begin, 
                                                                        EvidenceIter evidence_end) {
        
        auto evidence_pair = std::make_pair(evidence_begin, evidence_end);
        auto type_id = df.same_type(variable, evidence_pair);

        bool contains_null = df.null_count(variable, evidence_pair) > 0;

        if (type_id != Type::DICTIONARY) {
            throw py::value_error("Wrong data type to fit DiscreteFactor. Ccategorical data is expected.");
        }

        if (contains_null)
            return _fit<true>(df, variable, evidence_begin, evidence_end);
        else
            return _fit<false>(df, variable, evidence_begin, evidence_end);

    }
}

#endif //PGM_DATASET_MLE_DISCRETEFACTOR_HPP
