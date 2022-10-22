#ifndef PYBNESIAN_FACTORS_DISCRETE_DISCRETEADAPTATOR_HPP
#define PYBNESIAN_FACTORS_DISCRETE_DISCRETEADAPTATOR_HPP

#include <factors/factors.hpp>
#include <factors/discrete/discrete_indices.hpp>
#include <util/math_constants.hpp>
#include <fort.hpp>

using Eigen::VectorXi;

namespace factors::discrete {

class BaseFactorParameters {
public:
    virtual ~BaseFactorParameters() {}
    virtual std::shared_ptr<Factor> initialize(const std::string& variable,
                                               const std::vector<std::string>& evidence,
                                               const Assignment& discrete_assignment) const = 0;

    virtual py::tuple __getstate__() const = 0;
};

template <typename BaseFactor, typename... Args>
class BaseFactorParametersImpl : public BaseFactorParameters {
public:
    BaseFactorParametersImpl(Args... args) : m_args(args...) {
        static_assert(std::is_constructible_v<BaseFactor, const std::string&, const std::vector<std::string>&, Args...>,
                      "BaseFactor cannot be constructed with provided Args...");
    }

    std::shared_ptr<Factor> initialize(const std::string& variable,
                                       const std::vector<std::string>& evidence,
                                       const Assignment&) const override {
        if constexpr (std::tuple_size_v<decltype(m_args)> == 0) {
            return std::make_shared<BaseFactor>(variable, evidence);
        } else {
            return std::apply(
                [&variable, &evidence](const auto&... args) {
                    return std::make_shared<BaseFactor>(variable, evidence, args...);
                },
                m_args);
        }
    }

    py::tuple __getstate__() const override {
        return py::make_tuple(false, py::module_::import("pickle").attr("dumps")(m_args));
    }

private:
    std::tuple<Args...> m_args;
};

template <typename BaseFactor, typename... Args>
class SpecificBaseFactorParameters : public BaseFactorParameters {
public:
    SpecificBaseFactorParameters(std::unordered_map<Assignment, std::tuple<Args...>, AssignmentHash>& args)
        : m_args(args) {
        static_assert(std::is_constructible_v<BaseFactor, const std::string&, const std::vector<std::string>&, Args...>,
                      "BaseFactor cannot be constructed with provided Args...");
    }

    std::shared_ptr<Factor> initialize(const std::string& variable,
                                       const std::vector<std::string>& evidence,
                                       const Assignment& discrete_assignment) const override {
        if constexpr (std::tuple_size_v<typename decltype(m_args)::mapped_type> == 0) {
            return std::make_shared<BaseFactor>(variable, evidence);
        } else {
            auto f = m_args.find(discrete_assignment);

            if (f == m_args.end()) {
                return std::make_shared<BaseFactor>(variable, evidence);
            }

            return std::apply(
                [&variable, &evidence](const auto&... args) {
                    return std::make_shared<BaseFactor>(variable, evidence, args...);
                },
                f->second);
        }
    }

    py::tuple __getstate__() const override {
        return py::make_tuple(true, py::module_::import("pickle").attr("dumps")(m_args));
    }

private:
    std::unordered_map<Assignment, std::tuple<Args...>, AssignmentHash> m_args;
};

template <typename BaseFactor, typename BaseFitter, typename FactorName>
class DiscreteAdaptator : public Factor {
public:
    template <typename... CArgs>
    DiscreteAdaptator(const std::string& variable, const std::vector<std::string>& evidence, CArgs... args)
        : Factor(variable, evidence),
          m_args(std::make_unique<BaseFactorParametersImpl<BaseFactor, CArgs...>>(args...)),
          m_fitted(false),
          m_discrete_evidence(),
          m_discrete_values(),
          m_continuous_evidence(),
          m_cardinality(),
          m_strides(),
          m_factors() {}

    template <typename... CArgs>
    DiscreteAdaptator(const std::string& variable,
                      const std::vector<std::string>& evidence,
                      std::unordered_map<Assignment, std::tuple<CArgs...>, AssignmentHash> args)
        : Factor(variable, evidence),
          m_args(std::make_unique<SpecificBaseFactorParameters<BaseFactor, CArgs...>>(args)),
          m_fitted(false),
          m_discrete_evidence(),
          m_discrete_values(),
          m_continuous_evidence(),
          m_cardinality(),
          m_strides(),
          m_factors() {}

    std::shared_ptr<arrow::DataType> data_type() const override {
        check_fitted();
        return m_factors[0]->data_type();
    }

    std::shared_ptr<FactorType> type() const override { return BaseFactor::FactorTypeClass::get(); }
    FactorType& type_ref() const override { return BaseFactor::FactorTypeClass::get_ref(); }

    bool fitted() const override { return m_fitted; }

    void fit(const DataFrame& df) override;
    VectorXd logl(const DataFrame& df) const override;
    double slogl(const DataFrame& df) const override;

    std::shared_ptr<Factor> conditional_factor(Assignment& assignment) const;

    std::string ToString() const override;

    Array_ptr sample(int n,
                     const DataFrame& evidence_values,
                     unsigned int seed = std::random_device{}()) const override;

    py::tuple __getstate__() const override;

    static DiscreteAdaptator<BaseFactor, BaseFitter, FactorName> __setstate__(py::tuple& t);
    static DiscreteAdaptator<BaseFactor, BaseFitter, FactorName> __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    void check_fitted() const;
    void check_equal_domain(const DataFrame& df, bool check_variable) const;
    void run_checks(const DataFrame& df, bool check_variable) const {
        check_fitted();
        check_equal_domain(df, check_variable);
    }

    std::unique_ptr<BaseFactorParameters> m_args;
    bool m_fitted;
    std::vector<std::string> m_discrete_evidence;
    std::vector<std::vector<std::string>> m_discrete_values;
    std::vector<std::string> m_continuous_evidence;
    VectorXi m_cardinality;
    VectorXi m_strides;
    std::vector<std::shared_ptr<Factor>> m_factors;
};

template <typename BaseFactor, typename BaseFitter, typename FactorName>
void DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>::check_fitted() const {
    if (!m_fitted) throw std::invalid_argument("Factor " + ToString() + " not fitted.");
}

template <typename BaseFactor, typename BaseFitter, typename FactorName>
void DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>::check_equal_domain(const DataFrame& df,
                                                                               bool check_variable) const {
    if (check_variable) {
        df.raise_has_column(variable());

        switch (df.col(variable())->type_id()) {
            case Type::DOUBLE:
            case Type::FLOAT:
                break;
            default:
                throw std::invalid_argument("Variable " + variable() + " must have \"double\" or \"float\" data type.");
        }
    }

    df.raise_has_columns(evidence());

    for (const auto& e : m_continuous_evidence) {
        switch (df.col(e)->type_id()) {
            case Type::DOUBLE:
            case Type::FLOAT:
                break;
            default:
                throw std::invalid_argument("Variable " + e + " must have \"double\" or \"float\" data type.");
        }
    }

    for (size_t i = 0, i_end = m_discrete_evidence.size(); i < i_end; ++i) {
        check_domain_variable(df, m_discrete_evidence[i], m_discrete_values[i]);
    }
}

template <typename BaseFactor, typename BaseFitter, typename FactorName>
void DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>::fit(const DataFrame& df) {
    std::vector<std::string> discrete_evidence, continuous_evidence;

    for (const auto& e : evidence()) {
        switch (df.col(e)->type_id()) {
            case Type::DICTIONARY:
                discrete_evidence.push_back(e);
                break;
            case Type::DOUBLE:
            case Type::FLOAT:
                continuous_evidence.push_back(e);
                break;
            default:
                throw std::invalid_argument("Non valid data type for variable " + e +
                                            ". Only \"dictionary\", "
                                            "\"double\" and \"float\" data types are allowed.");
        }
    }

    m_discrete_evidence = discrete_evidence;
    m_discrete_values.clear();
    m_continuous_evidence = continuous_evidence;
    m_factors.clear();

    if (m_discrete_evidence.empty()) {
        auto factor = m_args->initialize(variable(), m_continuous_evidence, Assignment());
        m_factors.push_back(std::move(factor));
        m_factors.back()->fit(df);
    } else {
        std::tie(m_cardinality, m_strides) = factors::discrete::create_cardinality_strides(df, m_discrete_evidence);

        m_discrete_values.reserve(m_discrete_evidence.size());
        for (auto it = m_discrete_evidence.begin(), end = m_discrete_evidence.end(); it != end; ++it) {
            auto dict_evidence = std::static_pointer_cast<arrow::DictionaryArray>(df.col(*it));

            factors::discrete::check_is_string_dictionary(dict_evidence, *it);
            auto dict_evidence_values = std::static_pointer_cast<arrow::StringArray>(dict_evidence->dictionary());

            std::vector<std::string> ev;
            ev.reserve(dict_evidence_values->length());
            for (auto j = 0; j < dict_evidence_values->length(); ++j) {
                ev.push_back(dict_evidence_values->GetString(j));
            }

            m_discrete_values.push_back(ev);
        }

        auto num_factors = m_cardinality.prod();
        m_factors.reserve(num_factors);

        auto slices = discrete_slice_indices(df, m_discrete_evidence, m_strides, num_factors);

        for (auto i = 0; i < num_factors; ++i) {
            if (slices[i]) {
                auto assignment =
                    Assignment::from_index(i, m_discrete_evidence, m_discrete_values, m_cardinality, m_strides);

                auto factor = m_args->initialize(variable(), m_continuous_evidence, assignment);
                m_factors.push_back(std::move(factor));

                if (!m_factors.back()->fitted()) {
                    auto df_filtered = df.take(slices[i]);

                    if (!BaseFitter::fit(m_factors.back(), df_filtered)) {
                        m_factors.back() = nullptr;
                    }
                }
            } else {
                m_factors.push_back(nullptr);
            }
        }
    }

    m_fitted = true;
}

template <typename ArrowType>
void logl_impl(const std::shared_ptr<Factor>& f, const DataFrame& df, const Array_ptr& indices, VectorXd& res) {
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    auto dwn_indices = std::static_pointer_cast<ArrayType>(indices);
    auto raw_indices = dwn_indices->raw_values();

    if (f) {
        auto df_filtered = df.take(indices);
        auto ll = f->logl(df_filtered);

        for (auto i = 0; i < dwn_indices->length(); ++i) {
            res(raw_indices[i]) = ll(i);
        }
    } else {
        for (auto i = 0; i < dwn_indices->length(); ++i) {
            res(raw_indices[i]) = util::nan<double>;
        }
    }
}

template <typename BaseFactor, typename BaseFitter, typename FactorName>
VectorXd DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>::logl(const DataFrame& df) const {
    run_checks(df, true);

    if (m_discrete_evidence.empty()) {
        return m_factors[0]->logl(df);
    } else {
        auto num_factors = m_factors.size();
        auto slices = discrete_slice_indices(df, m_discrete_evidence, m_strides, num_factors);

        VectorXd res(df->num_rows());

        if (auto combined_bitmap = df.combined_bitmap(m_discrete_evidence)) {
            auto bitmap_data = combined_bitmap->data();
            for (auto i = 0; i < df->num_rows(); ++i) {
                if (!util::bit_util::GetBit(bitmap_data, i)) res(i) = util::nan<double>;
            }
        }

        for (size_t i = 0; i < num_factors; ++i) {
            if (slices[i]) {
                logl_impl<arrow::Int32Type>(m_factors[i], df, slices[i], res);
            }
        }

        return res;
    }
}

template <typename BaseFactor, typename BaseFitter, typename FactorName>
double DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>::slogl(const DataFrame& df) const {
    run_checks(df, true);

    if (m_discrete_evidence.empty()) {
        return m_factors[0]->slogl(df);
    } else {
        auto num_factors = m_factors.size();
        auto slices = discrete_slice_indices(df, m_discrete_evidence, m_strides, num_factors);

        double res = 0;

        for (size_t i = 0; i < num_factors; ++i) {
            if (slices[i] && m_factors[i]) {
                auto df_filtered = df.take(slices[i]);
                res += m_factors[i]->slogl(df_filtered);
            }
        }

        return res;
    }
}

template <typename BaseFactor, typename BaseFitter, typename FactorName>
std::shared_ptr<Factor> DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>::conditional_factor(
    Assignment& assignment) const {
    check_fitted();
    auto index = assignment.index(m_discrete_evidence, m_discrete_values, m_strides);
    return m_factors[index];
}

template <typename BaseFactor, typename BaseFitter, typename FactorName>
std::string DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>::ToString() const {
    std::stringstream ss;

    if (!this->evidence().empty()) {
        const auto& e = evidence();
        ss << "[" << FactorName::str << "] P(" << this->variable() << " | " << e[0];
        for (size_t i = 1; i < e.size(); ++i) {
            ss << ", " << e[i];
        }

        ss << ")";
    } else {
        ss << "[" << FactorName::str << "] P(" << variable() << ")";
    }

    if (!m_discrete_evidence.empty()) {
        if (m_fitted) {
            ss << std::endl;

            fort::char_table table;
            table.set_cell_text_align(fort::text_align::center);

            table[0][0].set_cell_span(m_discrete_evidence.size());

            std::stringstream varname;
            varname << variable();
            if (!m_continuous_evidence.empty()) {
                varname << " | " << m_continuous_evidence[0];
                for (size_t i = 1; i < m_continuous_evidence.size(); ++i) {
                    varname << ", " << m_continuous_evidence[i];
                }
            }

            table[0][m_discrete_evidence.size()] = varname.str();
            table << fort::endr << fort::header;
            table.range_write_ln(m_discrete_evidence.begin(), m_discrete_evidence.end());

            for (size_t k = 0, num_factors = m_cardinality.prod(); k < num_factors; ++k) {
                auto ass = Assignment::from_index(k, m_discrete_evidence, m_discrete_values, m_cardinality, m_strides);

                for (const auto& discrete_evidence : m_discrete_evidence) {
                    table << static_cast<std::string>(ass.value(discrete_evidence));
                }

                if (m_factors[k])
                    table << m_factors[k]->ToString();
                else
                    table << "not fitted";

                table << fort::endr;
            }

            ss << table.to_string();
        } else {
            ss << " not fitted.";
        }
    } else {
        if (m_fitted) {
            ss << " = " << m_factors[0]->ToString();
        } else {
            ss << " not fitted.";
        }
    }

    return ss.str();
}

template <typename IndicesArrowType, typename ResultArrowType>
void sample_factor_impl(const std::shared_ptr<Factor>& f,
                        int n,
                        const DataFrame& evidence_values,
                        unsigned int seed,
                        const Array_ptr& indices,
                        Array_ptr& res) {
    using IndicesArrayType = typename arrow::TypeTraits<IndicesArrowType>::ArrayType;
    using ResultArrayType = typename arrow::TypeTraits<ResultArrowType>::ArrayType;
    auto dwn_indices = std::static_pointer_cast<IndicesArrayType>(indices);
    auto raw_indices = dwn_indices->raw_values();

    auto raw_res = res->data()->template GetMutableValues<typename ResultArrowType::c_type>(1);

    if (f) {
        auto df_filtered = evidence_values.take(indices);
        auto sample = f->sample(n, df_filtered, seed);

        auto dwn_sample = std::static_pointer_cast<ResultArrayType>(sample);
        auto raw_sample = dwn_sample->raw_values();

        for (auto i = 0; i < dwn_indices->length(); ++i) {
            raw_res[raw_indices[i]] = raw_sample[i];
        }
    } else {
        for (auto i = 0; i < dwn_indices->length(); ++i) {
            raw_res[raw_indices[i]] = util::nan<typename ResultArrowType::c_type>;
        }
    }
}

template <typename ResultArrowType>
void sample_impl(std::vector<Array_ptr>& slice_builders,
                 const std::vector<std::shared_ptr<Factor>>& factors,
                 int n,
                 const DataFrame& evidence_values,
                 unsigned int seed,
                 Array_ptr& res) {
    auto num_factors = factors.size();

    for (size_t i = 0; i < num_factors; ++i) {
        if (slice_builders[i]) {
            sample_factor_impl<arrow::Int32Type, ResultArrowType>(
                factors[i], n, evidence_values, seed + i, slice_builders[i], res);
        }
    }
}

template <typename BaseFactor, typename BaseFitter, typename FactorName>
Array_ptr DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>::sample(int n,
                                                                        const DataFrame& evidence_values,
                                                                        unsigned int seed) const {
    if (n < 0) {
        throw std::invalid_argument("n should be a non-negative number");
    }

    run_checks(evidence_values, false);

    if (evidence_values->num_rows() != n)
        throw std::domain_error("Evidence values do not have " + std::to_string(n) + " rows to sample.");
    if (evidence_values.null_count(evidence()) > 0)
        throw std::domain_error("Evidence values contain null rows in the evidence variables.");

    if (m_discrete_evidence.empty()) {
        return m_factors[0]->sample(n, evidence_values, seed);
    } else {
        auto num_factors = m_factors.size();
        auto slices = discrete_slice_indices(evidence_values, m_discrete_evidence, m_strides, num_factors);

        Array_ptr res;

        switch (data_type()->id()) {
            case Type::DOUBLE: {
                arrow::NumericBuilder<arrow::DoubleType> builder;
                RAISE_STATUS_ERROR(builder.AppendEmptyValues(n));
                RAISE_STATUS_ERROR(builder.Finish(&res));

                sample_impl<arrow::DoubleType>(slices, m_factors, n, evidence_values, seed, res);
                break;
            }
            case Type::FLOAT: {
                arrow::NumericBuilder<arrow::FloatType> builder;
                RAISE_STATUS_ERROR(builder.AppendEmptyValues(n));
                RAISE_STATUS_ERROR(builder.Finish(&res));

                sample_impl<arrow::FloatType>(slices, m_factors, n, evidence_values, seed, res);
                break;
            }
            default:
                throw std::runtime_error("DiscreteAdaptator only implemented for continuous factors.");
        }

        return res;
    }
}

template <typename BaseFactor, typename BaseFitter, typename FactorName>
py::tuple DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>::__getstate__() const {
    return py::make_tuple(this->variable(),
                          this->evidence(),
                          m_args->__getstate__(),
                          m_fitted,
                          m_discrete_evidence,
                          m_discrete_values,
                          m_continuous_evidence,
                          m_cardinality,
                          m_strides,
                          m_factors);
}

template <typename BaseFactor, typename BaseFitter, typename FactorName>
DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>
DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>::__setstate__(py::tuple& t) {
    auto pyargs = t[2].cast<py::tuple>();
    auto specific_args = pyargs[0].cast<bool>();

    auto args = py::module_::import("pickle").attr("loads")(pyargs[1]);
    auto res = [specific_args, &t, &args]() {
        if (specific_args)
            return py::type::of<DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>>()(t[0], t[1], args)
                .template cast<DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>>();
        else
            return py::type::of<DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>>()(t[0], t[1], *args)
                .template cast<DiscreteAdaptator<BaseFactor, BaseFitter, FactorName>>();
    }();

    res.m_fitted = t[3].cast<bool>();

    if (res.m_fitted) {
        res.m_discrete_evidence = t[4].cast<std::vector<std::string>>();
        res.m_discrete_values = t[5].cast<std::vector<std::vector<std::string>>>();
        res.m_continuous_evidence = t[6].cast<std::vector<std::string>>();
        res.m_cardinality = t[7].cast<VectorXi>();
        res.m_strides = t[8].cast<VectorXi>();
        // All the base factors are C++, so no need to keep Python object alive
        res.m_factors = t[9].cast<std::vector<std::shared_ptr<Factor>>>();
    }

    return res;
}

}  // namespace factors::discrete

#endif  // PYBNESIAN_FACTORS_DISCRETE_DISCRETEADAPTATOR_HPP