#ifndef PYBNESIAN_FACTORS_FACTORS_HPP
#define PYBNESIAN_FACTORS_FACTORS_HPP

#include <string>
#include <stdint.h>
#include <cstddef>
#include <stdexcept>
#include <random>
#include <pybind11/pybind11.h>
#include <dataset/dataset.hpp>

using dataset::DataFrame;

namespace py = pybind11;

namespace models {
class BayesianNetworkBase;
class ConditionalBayesianNetworkBase;
}  // namespace models

using models::BayesianNetworkBase, models::ConditionalBayesianNetworkBase;

namespace factors {

class ConditionalFactor;

class FactorType {
public:
    virtual ~FactorType() {}

    bool operator==(const FactorType& o) const { return this->hash() == o.hash(); }
    bool operator!=(const FactorType& o) const { return !(*this == o); }
    bool operator==(FactorType&& o) const { return this->hash() == o.hash(); }
    bool operator!=(FactorType&& o) const { return !(*this == o); }

    virtual bool is_python_derived() const { return false; }

    static std::shared_ptr<FactorType> keep_python_alive(std::shared_ptr<FactorType>& f) {
        if (f && f->is_python_derived()) {
            auto o = py::cast(f);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<FactorType*>();
            return std::shared_ptr<FactorType>(keep_python_state_alive, ptr);
        }

        return f;
    }

    virtual std::shared_ptr<ConditionalFactor> new_cfactor(const BayesianNetworkBase&,
                                                           const std::string&,
                                                           const std::vector<std::string>&) const = 0;
    virtual std::shared_ptr<ConditionalFactor> new_cfactor(const ConditionalBayesianNetworkBase&,
                                                           const std::string&,
                                                           const std::vector<std::string>&) const = 0;
    virtual std::shared_ptr<FactorType> opposite_semiparametric() const = 0;
    virtual std::string ToString() const = 0;

    virtual std::size_t hash() const { return m_hash; }

    virtual py::tuple __getstate__() const = 0;

protected:
    // Use memory address of object as hash value.
    mutable std::uintptr_t m_hash;
};

template <typename F>
void save_factor(const F& factor, std::string name) {
    auto open = py::module_::import("io").attr("open");

    if (name.size() < 7 || name.substr(name.size() - 7) != ".pickle") name += ".pickle";

    auto file = open(name, "wb");
    py::module_::import("pickle").attr("dump")(py::cast(&factor), file, 2);
    file.attr("close")();
}

class Factor {
public:
    Factor() = default;
    Factor(const std::vector<std::string>& variables) : m_variables(variables) {}

    const std::vector<std::string>& variables() const { return m_variables; }

    // std::shared_ptr<Factor> product(const std::shared_ptr<Factor>& f) const = 0;
private:
    std::vector<std::string> m_variables;
};

class ConditionalFactor : public Factor {
public:
    ConditionalFactor() = default;
    ConditionalFactor(const std::string& variable, const std::vector<std::string>& evidence)
        : m_variable(variable), m_evidence(evidence) {}

    virtual ~ConditionalFactor() {}

    virtual bool is_python_derived() const { return false; }

    static std::shared_ptr<ConditionalFactor> keep_python_alive(std::shared_ptr<ConditionalFactor>& f) {
        if (f && f->is_python_derived()) {
            auto o = py::cast(f);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<ConditionalFactor*>();
            return std::shared_ptr<ConditionalFactor>(keep_python_state_alive, ptr);
        }

        return f;
    }

    const std::string& variable() const { return m_variable; }

    const std::vector<std::string>& evidence() const { return m_evidence; }

    virtual std::shared_ptr<arrow::DataType> data_type() const = 0;

    virtual std::shared_ptr<FactorType> type() const = 0;
    virtual FactorType& type_ref() const = 0;

    virtual bool fitted() const = 0;
    virtual void fit(const DataFrame& df) = 0;
    virtual VectorXd logl(const DataFrame& df) const = 0;
    virtual double slogl(const DataFrame& df) const = 0;
    // VectorXd cdf(const DataFrame& df) const;

    virtual std::string ToString() const = 0;

    virtual Array_ptr sample(int n,
                             const DataFrame& evidence_values,
                             unsigned int seed = std::random_device{}()) const = 0;

    void save(const std::string& name) const { save_factor(*this, name); }

    virtual py::tuple __getstate__() const = 0;

private:
    std::string m_variable;
    std::vector<std::string> m_evidence;
};

}  // namespace factors

#endif  // PYBNESIAN_FACTORS_FACTORS_HPP