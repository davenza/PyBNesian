#ifndef PYBNESIAN_FACTORS_DISCRETE_DISCRETEFACTOR_HPP
#define PYBNESIAN_FACTORS_DISCRETE_DISCRETEFACTOR_HPP

#include <random>
#include <dataset/dataset.hpp>

using dataset::DataFrame;
using Eigen::VectorXd, Eigen::VectorXi;

using Array_ptr = std::shared_ptr<arrow::Array>;

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

    template<bool contains_null>
    VectorXi discrete_indices(const DataFrame& df, 
                              const std::string& variable, 
                              const std::vector<std::string>& evidence,
                              const VectorXi& strides) {
        
        if constexpr(contains_null) {
            auto combined_bitmap = df.combined_bitmap(variable, evidence);

            auto valid_rows = util::bit_util::non_null_count(combined_bitmap, df->num_rows());

            VectorXi indices = VectorXi::Zero(valid_rows);

            auto dict_variable = std::static_pointer_cast<arrow::DictionaryArray>(df.col(variable));
            auto variable_indices = dict_variable->indices();

            sum_to_discrete_indices_null(indices, variable_indices, strides(0), combined_bitmap);

            int i = 1;
            for (auto it = evidence.begin(), end = evidence.end(); it != end; ++it, ++i) {
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
            for (auto it = evidence.begin(), end = evidence.end(); it != end; ++it, ++i) {
                auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));
                auto evidence_indices = dict_evidence->indices();
                sum_to_discrete_indices(indices, evidence_indices, strides(i));
            }

            return indices;
        }
    }

    VectorXi discrete_indices(const DataFrame& df, 
                              const std::string& variable, 
                              const std::vector<std::string>& evidence,
                              const VectorXi& strides);

    struct DiscreteFactor_Params {
        VectorXd logprob;
        VectorXi cardinality;
        VectorXi strides;
    };

    class DiscreteFactor {
    public:
        using ParamsClass = DiscreteFactor_Params;

        DiscreteFactor() = default;
        DiscreteFactor(std::string variable, std::vector<std::string> evidence) : m_variable(variable),
                                                                                  m_evidence(evidence),
                                                                                  m_variable_values(),
                                                                                  m_evidence_values(),
                                                                                  m_logprob(),
                                                                                  m_cardinality(),
                                                                                  m_strides(),
                                                                                  m_fitted(false) {}


        const std::string& variable() const { return m_variable; }
        const std::vector<std::string>& variable_values() const { return m_variable_values; }
        const std::vector<std::string>& evidence() const { return m_evidence; }
        const std::vector<std::vector<std::string>>& evidence_values() const { return m_evidence_values; }
        bool fitted() const { return m_fitted; }
        void fit(const DataFrame& df);
        VectorXd logl(const DataFrame& df, bool check_domain=true) const;
        double slogl(const DataFrame& df, bool check_domain=true) const;

        std::string ToString() const;

        VectorXi discrete_indices(const DataFrame& df) const {
            return factors::discrete::discrete_indices(df, m_variable, m_evidence, m_strides);
        }

        template<bool contains_null>
        VectorXi discrete_indices(const DataFrame& df) const {
            return factors::discrete::discrete_indices<contains_null>(df, m_variable, m_evidence, m_strides);
        }

        Array_ptr sample(int n, const DataFrame& evidence_values, 
                         unsigned int seed = std::random_device{}()) const;

        py::tuple __getstate__() const;
        static DiscreteFactor __setstate__(py::tuple& t);
        static DiscreteFactor __setstate__(py::tuple&& t) {
            return __setstate__(t);
        }
    private:
        void check_equal_domain(const DataFrame& df) const;
        VectorXd _logl(const DataFrame& df) const;
        VectorXd _logl_null(const DataFrame& df) const;
        double _slogl(const DataFrame& df) const;
        double _slogl_null(const DataFrame& df) const;

        template<typename ArrowType>
        Array_ptr sample_indices(int n, const DataFrame& evidence_values,
                                 unsigned int seed) const;

        std::string m_variable;
        std::vector<std::string> m_evidence;
        std::vector<std::string> m_variable_values;
        std::vector<std::vector<std::string>> m_evidence_values;
        VectorXd m_logprob;
        VectorXi m_cardinality;
        VectorXi m_strides;
        bool m_fitted;
    };

    template<typename ArrowType>
    Array_ptr DiscreteFactor::sample_indices(int n, 
                                             const DataFrame& evidence_values, 
                                             unsigned int seed) const {

        int parent_configurations = m_logprob.rows() / m_variable_values.size();
        VectorXd accum_prob(m_logprob.rows());

        for (auto i = 0; i < parent_configurations; ++i) {
            auto offset = i*m_variable_values.size();

            accum_prob(offset) = std::exp(m_logprob(offset));
            for (size_t j = 1, end = m_variable_values.size()-1; j < end; ++j) {
                accum_prob(offset + j) = accum_prob(offset + j - 1) + std::exp(m_logprob(offset + j));
            }
        }

        std::mt19937 rng{seed};
        std::uniform_int_distribution<> uniform(0, 1);

        using CType = typename ArrowType::c_type;
        arrow::NumericBuilder<ArrowType> builder;
        RAISE_STATUS_ERROR(builder.Resize(n));

        if (!m_evidence.empty()) {
            if (!evidence_values.has_columns(m_evidence))
                throw std::domain_error("Evidence values not present for sampling.");

            VectorXi parent_offset = VectorXi::Zero(n);
            for (size_t i = 0; i < m_evidence.size(); ++i) {
                auto array = evidence_values->GetColumnByName(m_evidence[i]);

                auto dwn_array = std::static_pointer_cast<arrow::DictionaryArray>(array);
                auto array_indices = dwn_array->indices();

                sum_to_discrete_indices(parent_offset, array_indices, m_strides(i+1));
            }

            for (auto i = 0; i < n; ++i) {
                double random_number = uniform(rng);

                CType index = m_variable_values.size()-1;
                for (size_t j = 0, end = m_variable_values.size()-1; j < end; ++j) {
                    if (random_number < accum_prob(parent_offset(i) + j)) {
                        index = j;
                        break;
                    }
                }
                builder.UnsafeAppend(index);
            }
        } else {
            for (auto i = 0; i < n; ++i) {
                double random_number = uniform(rng);

                CType index = m_variable_values.size()-1;
                for (size_t j = 0, end = m_variable_values.size()-1; j < end; ++j) {
                    if (random_number < accum_prob(j)) {
                        index = j;
                        break;
                    }
                }
                builder.UnsafeAppend(index);
            }
        }
        
        std::shared_ptr<arrow::Array> out;
        RAISE_STATUS_ERROR(builder.Finish(&out));
        
        return out;
    }

}

#endif //PYBNESIAN_FACTORS_DISCRETE_DISCRETEFACTOR_HPP