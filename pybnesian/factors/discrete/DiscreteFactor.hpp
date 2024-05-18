#ifndef PYBNESIAN_FACTORS_DISCRETE_DISCRETEFACTOR_HPP
#define PYBNESIAN_FACTORS_DISCRETE_DISCRETEFACTOR_HPP

#include <random>
#include <dataset/dataset.hpp>
#include <factors/factors.hpp>
#include <factors/discrete/discrete_indices.hpp>

using dataset::DataFrame;
using Eigen::VectorXd, Eigen::VectorXi;
using factors::FactorType;

using Array_ptr = std::shared_ptr<arrow::Array>;

namespace factors::discrete {

class DiscreteFactorType : public FactorType {
public:
    DiscreteFactorType(const DiscreteFactorType&) = delete;
    void operator=(const DiscreteFactorType&) = delete;

    static std::shared_ptr<DiscreteFactorType> get() {
        static std::shared_ptr<DiscreteFactorType> singleton =
            std::shared_ptr<DiscreteFactorType>(new DiscreteFactorType);
        return singleton;
    }

    static DiscreteFactorType& get_ref() {
        static DiscreteFactorType& ref = *DiscreteFactorType::get();
        return ref;
    }

    std::shared_ptr<Factor> new_factor(const BayesianNetworkBase&,
                                       const std::string&,
                                       const std::vector<std::string>&,
                                       py::args = py::args{},
                                       py::kwargs = py::kwargs{}) const override;
    std::shared_ptr<Factor> new_factor(const ConditionalBayesianNetworkBase&,
                                       const std::string&,
                                       const std::vector<std::string>&,
                                       py::args = py::args{},
                                       py::kwargs = py::kwargs{}) const override;

    std::string ToString() const override { return "DiscreteFactor"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<DiscreteFactorType> __setstate__(py::tuple&) { return DiscreteFactorType::get(); }

private:
    DiscreteFactorType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

struct DiscreteFactor_Params {
    VectorXd logprob;
    VectorXi cardinality;
};

class DiscreteFactor : public Factor {
public:
    using ParamsClass = DiscreteFactor_Params;

    DiscreteFactor() = default;
    DiscreteFactor(std::string variable, std::vector<std::string> evidence)
        : Factor(variable, evidence),
          m_variable_values(),
          m_evidence_values(),
          m_logprob(),
          m_cardinality(),
          m_strides(),
          m_fitted(false) {}

    std::shared_ptr<FactorType> type() const override { return DiscreteFactorType::get(); }

    FactorType& type_ref() const override { return DiscreteFactorType::get_ref(); }

    std::shared_ptr<arrow::DataType> data_type() const override {
        check_fitted();
        if (m_variable_values.size() - 1 <= std::numeric_limits<typename arrow::Int8Type::c_type>::max()) {
            return arrow::dictionary(arrow::int8(), arrow::utf8());
        } else if (m_variable_values.size() - 1 <= std::numeric_limits<typename arrow::Int16Type::c_type>::max()) {
            return arrow::dictionary(arrow::int16(), arrow::utf8());
        } else if (m_variable_values.size() - 1 <= std::numeric_limits<typename arrow::Int32Type::c_type>::max()) {
            return arrow::dictionary(arrow::int32(), arrow::utf8());
        } else {
            return arrow::dictionary(arrow::int64(), arrow::utf8());
        }
    }

    const std::vector<std::string>& variable_values() const {
        check_fitted();
        return m_variable_values;
    }
    const std::vector<std::vector<std::string>>& evidence_values() const {
        check_fitted();
        return m_evidence_values;
    }
    bool fitted() const override { return m_fitted; }
    void fit(const DataFrame& df) override;
    VectorXd logl(const DataFrame& df) const override;
    double slogl(const DataFrame& df) const override;

    std::string ToString() const override;

    template <bool contains_null>
    VectorXi discrete_indices(const DataFrame& df) const {
        return factors::discrete::discrete_indices<contains_null>(df, variable(), evidence(), m_strides);
    }

    Array_ptr sample(int n,
                     const DataFrame& evidence_values,
                     unsigned int seed = std::random_device{}()) const override;

    py::tuple __getstate__() const override;
    static DiscreteFactor __setstate__(py::tuple& t);
    static DiscreteFactor __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    void check_fitted() const {
        if (!fitted()) throw std::invalid_argument("DiscreteFactor factor not fitted.");
    }
    void check_equal_domain(const DataFrame& df, bool check_variable) const;
    void run_checks(const DataFrame& df, bool check_variable) const {
        check_fitted();
        check_equal_domain(df, check_variable);
    }

    VectorXd _logl(const DataFrame& df) const;
    VectorXd _logl_null(const DataFrame& df) const;
    double _slogl(const DataFrame& df) const;
    double _slogl_null(const DataFrame& df) const;

    template <typename ArrowType>
    Array_ptr sample_indices(int n, const DataFrame& evidence_values, unsigned int seed) const;

    std::vector<std::string> m_variable_values;
    std::vector<std::vector<std::string>> m_evidence_values;
    VectorXd m_logprob;
    VectorXi m_cardinality;
    VectorXi m_strides;
    bool m_fitted;
};

template <typename ArrowType>
Array_ptr DiscreteFactor::sample_indices(int n, const DataFrame& evidence_values, unsigned int seed) const {
    int parent_configurations = m_logprob.rows() / m_variable_values.size();
    VectorXd accum_prob(m_logprob.rows());

    for (auto i = 0; i < parent_configurations; ++i) {
        auto offset = i * m_variable_values.size();

        accum_prob(offset) = std::exp(m_logprob(offset));
        for (size_t j = 1, end = m_variable_values.size() - 1; j < end; ++j) {
            accum_prob(offset + j) = accum_prob(offset + j - 1) + std::exp(m_logprob(offset + j));
        }
    }

    std::mt19937 rng{seed};
    std::uniform_real_distribution<> uniform(0, 1);

    using CType = typename ArrowType::c_type;
    arrow::NumericBuilder<ArrowType> builder;
    RAISE_STATUS_ERROR(builder.Resize(n));

    if (!evidence().empty()) {
        if (!evidence_values.has_columns(evidence()))
            throw std::domain_error("Evidence values not present for sampling.");
        if (evidence_values->num_rows() != n)
            throw std::domain_error("Evidence values do not have " + std::to_string(n) + " rows to sample.");
        if (evidence_values.null_count(evidence()) > 0)
            throw std::domain_error("Evidence values contain null rows in the evidence variables.");

        auto parent_offset =
            factors::discrete::discrete_indices(evidence_values, evidence(), m_strides.tail(evidence().size()));

        for (auto i = 0; i < n; ++i) {
            double random_number = uniform(rng);

            CType index = m_variable_values.size() - 1;
            for (size_t j = 0, end = m_variable_values.size() - 1; j < end; ++j) {
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

            CType index = m_variable_values.size() - 1;
            for (size_t j = 0, end = m_variable_values.size() - 1; j < end; ++j) {
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

}  // namespace factors::discrete

#endif  // PYBNESIAN_FACTORS_DISCRETE_DISCRETEFACTOR_HPP