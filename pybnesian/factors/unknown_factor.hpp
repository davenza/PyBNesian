#ifndef PYBNESIAN_FACTORS_UNKNOWN_FACTOR_HPP
#define PYBNESIAN_FACTORS_UNKNOWN_FACTOR_HPP

#include <factors/unknown_factor.hpp>

namespace py = pybind11;

namespace factors {

class UnknownFactorType : public FactorType {
public:
    UnknownFactorType(const UnknownFactorType&) = delete;
    void operator=(const UnknownFactorType&) = delete;

    static std::shared_ptr<UnknownFactorType> get() {
        static std::shared_ptr<UnknownFactorType> singleton = std::shared_ptr<UnknownFactorType>(new UnknownFactorType);
        return singleton;
    }

    static UnknownFactorType& get_ref() {
        static UnknownFactorType& ref = *UnknownFactorType::get();
        return ref;
    }

    std::shared_ptr<Factor> new_factor(const BayesianNetworkBase&,
                                       const std::string&,
                                       const std::vector<std::string>&,
                                       py::args = py::args{},
                                       py::kwargs = py::kwargs{}) const override {
        throw py::type_error(
            "UnknownFactorType cannot create a new Factor (UnknownFactorType::new_factor was called).");
    }

    std::shared_ptr<Factor> new_factor(const ConditionalBayesianNetworkBase&,
                                       const std::string&,
                                       const std::vector<std::string>&,
                                       py::args = py::args{},
                                       py::kwargs = py::kwargs{}) const override {
        throw py::type_error(
            "UnknownFactorType cannot create a new Factor (UnknownFactorType::new_factor was called).");
    }

    std::string ToString() const override { return "UnknownFactorType"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<UnknownFactorType> __setstate__(py::tuple&) { return UnknownFactorType::get(); }

private:
    UnknownFactorType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

}  // namespace factors

#endif  // PYBNESIAN_FACTORS_UNKNOWN_FACTOR_HPP