#ifndef PGM_DATASET_DISCRETE_FACTOR_HPP
#define PGM_DATASET_DISCRETE_FACTOR_HPP


#include <iostream>
#include <dataset/dataset.hpp>
#include <Eigen/Dense>

using dataset::DataFrame;
using Eigen::VectorXd, Eigen::VectorXi;

namespace factors::discrete {


    template<typename ArrowType>
    void sum_to_discrete_indices_null(VectorXi& accum_indices, Array_ptr& indices, int stride, Buffer_ptr& combined_bitmap) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        auto dwn_indices = std::static_pointer_cast<ArrayType>(indices);

        auto raw_combined_bitmap = combined_bitmap->data();
        for (auto i = 0, j = 0; i < indices->length(); ++i) {
            if (arrow::BitUtil::GetBit(raw_combined_bitmap, i)) {
                accum_indices(j++) += dwn_indices->Value(i) * stride;
            }
        }
    }

    void sum_to_discrete_indices_null(VectorXi& accum_indices, Array_ptr& indices, int stride, Buffer_ptr& combined_bitmap);

    template<typename ArrowType>
    void sum_to_discrete_indices(VectorXi& accum_indices, Array_ptr& indices, int stride) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        using EigenMap = Map<const Matrix<typename ArrowType::c_type, Dynamic, 1>>;
        auto dwn_indices = std::static_pointer_cast<ArrayType>(indices);
        auto* raw_values = dwn_indices->raw_values();
        const EigenMap map_eigen(raw_values, indices->length());
        accum_indices += (map_eigen * stride).template cast<int>();
    }

    void sum_to_discrete_indices(VectorXi& accum_indices, Array_ptr& indices, int stride);

    template<bool contains_null, typename VarType, typename EvidenceIter>
    VectorXi discrete_indices(const DataFrame& df, 
                                const VarType& variable, 
                                EvidenceIter evidence_begin,
                                EvidenceIter evidence_end,
                                const VectorXi& strides) {
        
        if constexpr(contains_null) {
            auto evidence_pair = std::make_pair(evidence_begin, evidence_end);
            auto combined_bitmap = df.combined_bitmap(variable, evidence_pair);

            auto valid_rows = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

            VectorXi indices = VectorXi::Zero(valid_rows);

            auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));
            auto variable_indices = dict_variable->indices();

            sum_to_discrete_indices_null(indices, variable_indices, strides(0), combined_bitmap);

            int i = 1;
            for (auto it = evidence_begin; it != evidence_end; ++it, ++i) {
                auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
                auto evidence_indices = dict_evidence->indices();
                sum_to_discrete_indices_null(indices, evidence_indices, strides(i), combined_bitmap);
            }

            return indices;
        } else {
            VectorXi indices = VectorXi::Zero(df->num_rows());

            auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));
            auto variable_indices = dict_variable->indices();

            sum_to_discrete_indices(indices, variable_indices, strides(0));

            int i = 1;
            for (auto it = evidence_begin; it != evidence_end; ++it, ++i) {
                auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
                auto evidence_indices = dict_evidence->indices();
                sum_to_discrete_indices(indices, evidence_indices, strides(i));
            }

            return indices;
        }
    }

    template<typename VarType, typename EvidenceIter>
    VectorXi discrete_indices(const DataFrame& df, 
                                const VarType& variable, 
                                EvidenceIter evidence_begin,
                                EvidenceIter evidence_end,
                                const VectorXi& strides) {
        auto evidence_pair = std::make_pair(evidence_begin, evidence_end);
        if (df.null_count(variable, evidence_pair) == 0)
            return discrete_indices<false>(df, variable, evidence_begin, evidence_end, strides);
        else
            return discrete_indices<true>(df, variable, evidence_begin, evidence_end, strides);
    }


    struct DiscreteFactor_Params {
        VectorXd logprob;
        VectorXi cardinality;
        VectorXi strides;
    };

    class DiscreteFactor {
    public:
        using ParamsClass = DiscreteFactor_Params;

        DiscreteFactor(std::string variable, std::vector<std::string> evidence) : m_variable(variable),
                                                                                  m_evidence(evidence),
                                                                                  m_variable_values(),
                                                                                  m_evidence_values(),
                                                                                  m_logprob(),
                                                                                  m_cardinality(),
                                                                                  m_strides(),
                                                                                  m_fitted(false) {}


        const std::string& variable() const { return m_variable; }
        const std::vector<std::string>& evidence() const { return m_evidence; }
        bool fitted() const { return m_fitted; }
        void fit(const DataFrame& df);
        VectorXd logl(const DataFrame& df, bool check_domain=true) const;
        double slogl(const DataFrame& df, bool check_domain=true) const;

        std::string ToString() const;

        VectorXi discrete_indices(const DataFrame& df) const {
            return factors::discrete::discrete_indices(df, m_variable, 
                                        m_evidence.begin(), m_evidence.end(), m_strides);
        }

        template<bool contains_null>
        VectorXi discrete_indices(const DataFrame& df) const {
            return factors::discrete::discrete_indices<contains_null>(df, m_variable, 
                                        m_evidence.begin(), m_evidence.end(), m_strides);
        }

    private:
        void check_equal_domain(const DataFrame& df) const;
        VectorXd _logl(const DataFrame& df) const;
        VectorXd _logl_null(const DataFrame& df) const;
        double _slogl(const DataFrame& df) const;
        double _slogl_null(const DataFrame& df) const;

        std::string m_variable;
        std::vector<std::string> m_evidence;
        std::vector<std::string> m_variable_values;
        std::vector<std::vector<std::string>> m_evidence_values;
        VectorXd m_logprob;
        VectorXi m_cardinality;
        VectorXi m_strides;
        bool m_fitted;
    };

}

#endif //PGM_DATASET_DISCRETE_FACTOR_HPP