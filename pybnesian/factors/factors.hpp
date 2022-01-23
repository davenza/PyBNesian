#ifndef PYBNESIAN_FACTORS_FACTORS_HPP
#define PYBNESIAN_FACTORS_FACTORS_HPP

#include <string>
#include <stdint.h>
#include <cstddef>
#include <stdexcept>
#include <random>
#include <pybind11/pybind11.h>
#include <dataset/dataset.hpp>
#include <util/pickle.hpp>

using dataset::DataFrame;

namespace py = pybind11;

namespace models {
class BayesianNetworkBase;
class ConditionalBayesianNetworkBase;
}  // namespace models

using models::BayesianNetworkBase, models::ConditionalBayesianNetworkBase;

namespace factors {

class Factor;

class FactorType {
public:
    virtual ~FactorType() {}

    bool operator==(const FactorType& o) const { return this->hash() == o.hash(); }
    bool operator!=(const FactorType& o) const { return !(*this == o); }
    bool operator==(FactorType&& o) const { return this->hash() == o.hash(); }
    bool operator!=(FactorType&& o) const { return !(*this == o); }

    virtual bool is_python_derived() const { return false; }

    static std::shared_ptr<FactorType>& keep_python_alive(std::shared_ptr<FactorType>& f) {
        if (f && f->is_python_derived()) {
            auto o = py::cast(f);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<FactorType*>();
            f = std::shared_ptr<FactorType>(keep_python_state_alive, ptr);
        }

        return f;
    }

    static std::shared_ptr<FactorType> keep_python_alive(const std::shared_ptr<FactorType>& f) {
        if (f && f->is_python_derived()) {
            auto o = py::cast(f);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<FactorType*>();
            return std::shared_ptr<FactorType>(keep_python_state_alive, ptr);
        }

        return f;
    }

    static std::vector<std::shared_ptr<FactorType>>& keep_vector_python_alive(
        std::vector<std::shared_ptr<FactorType>>& v) {
        for (auto& f : v) {
            FactorType::keep_python_alive(f);
        }

        return v;
    }

    static std::vector<std::shared_ptr<FactorType>> keep_vector_python_alive(
        const std::vector<std::shared_ptr<FactorType>>& v) {
        std::vector<std::shared_ptr<FactorType>> fv;
        fv.reserve(v.size());

        for (const auto& f : v) {
            fv.push_back(FactorType::keep_python_alive(f));
        }

        return fv;
    }

    virtual std::shared_ptr<Factor> new_factor(const BayesianNetworkBase&,
                                               const std::string&,
                                               const std::vector<std::string>&,
                                               py::args = py::args{},
                                               py::kwargs = py::kwargs{}) const = 0;
    virtual std::shared_ptr<Factor> new_factor(const ConditionalBayesianNetworkBase&,
                                               const std::string&,
                                               const std::vector<std::string>&,
                                               py::args = py::args{},
                                               py::kwargs = py::kwargs{}) const = 0;
    virtual std::string ToString() const = 0;

    virtual std::size_t hash() const { return m_hash; }

    virtual py::tuple __getstate__() const = 0;

protected:
    // Use memory address of object as hash value.
    mutable std::uintptr_t m_hash;
};

// Create a C++ new factor taking into account the args/kwargs.
template <typename F>
std::shared_ptr<F> generic_new_factor(const std::string& variable,
                                      const std::vector<std::string>& evidence,
                                      py::args args,
                                      py::kwargs kwargs) {
    if (args.empty() && kwargs.empty())
        return std::make_shared<F>(variable, evidence);
    else {
        auto type = py::type::handle_of<F>();
        auto obj = type(variable, evidence, *args, **kwargs);
        return obj.template cast<std::shared_ptr<F>>();
    }
}

class Factor {
public:
    Factor() = default;
    Factor(const std::string& variable, const std::vector<std::string>& evidence)
        : m_variable(variable), m_evidence(evidence) {}

    virtual ~Factor() {}

    virtual bool is_python_derived() const { return false; }

    static std::shared_ptr<Factor>& keep_python_alive(std::shared_ptr<Factor>& f) {
        if (f && f->is_python_derived()) {
            auto o = py::cast(f);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<Factor*>();
            f = std::shared_ptr<Factor>(keep_python_state_alive, ptr);
        }

        return f;
    }

    static std::shared_ptr<Factor> keep_python_alive(const std::shared_ptr<Factor>& f) {
        if (f && f->is_python_derived()) {
            auto o = py::cast(f);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<Factor*>();
            return std::shared_ptr<Factor>(keep_python_state_alive, ptr);
        }

        return f;
    }

    static std::vector<std::shared_ptr<Factor>>& keep_vector_python_alive(std::vector<std::shared_ptr<Factor>>& v) {
        for (auto& f : v) {
            Factor::keep_python_alive(f);
        }

        return v;
    }

    static std::vector<std::shared_ptr<Factor>> keep_vector_python_alive(
        const std::vector<std::shared_ptr<Factor>>& v) {
        std::vector<std::shared_ptr<Factor>> fv;
        fv.reserve(v.size());

        for (const auto& f : v) {
            fv.push_back(Factor::keep_python_alive(f));
        }

        return fv;
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

    void save(const std::string& name) const { util::save_object(*this, name); }

    virtual py::tuple __getstate__() const = 0;

private:
    std::string m_variable;
    std::vector<std::string> m_evidence;
};

}  // namespace factors

#endif  // PYBNESIAN_FACTORS_FACTORS_HPP