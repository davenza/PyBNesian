#ifndef PYBNESIAN_FACTORS_FACTORS_HPP
#define PYBNESIAN_FACTORS_FACTORS_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <factors/discrete/DiscreteFactor.hpp>

namespace factors {

class LinearGaussianCPDType : public FactorType {
public:
    LinearGaussianCPDType(const LinearGaussianCPDType&) = delete;
    void operator=(const LinearGaussianCPDType&) = delete;

    static std::shared_ptr<FactorType> get() {
        static std::shared_ptr<FactorType> singleton =
            std::shared_ptr<LinearGaussianCPDType>(new LinearGaussianCPDType);
        return singleton;
    }

    std::shared_ptr<Factor> new_factor(const std::string& variable,
                                       const std::vector<std::string>& parents) const override {
        return std::make_shared<LinearGaussianCPD>(variable, parents);
    }

    std::shared_ptr<FactorType> opposite_semiparametric() const override;

    std::string ToString() const override { return "LinearGaussianFactor"; }

private:
    LinearGaussianCPDType() { m_ptr = reinterpret_cast<std::uintptr_t>(this); }
};

class CKDEType : public FactorType {
public:
    CKDEType(const CKDEType&) = delete;
    void operator=(const CKDEType&) = delete;

    static std::shared_ptr<FactorType> get() {
        static std::shared_ptr<FactorType> singleton = std::shared_ptr<CKDEType>(new CKDEType);
        return singleton;
    }

    std::shared_ptr<Factor> new_factor(const std::string& variable,
                                       const std::vector<std::string>& parents) const override {
        return std::make_shared<CKDE>(variable, parents);
    }

    std::shared_ptr<FactorType> opposite_semiparametric() const override { return LinearGaussianCPDType::get(); }

    std::string ToString() const override { return "CKDEFactor"; }

private:
    CKDEType() { m_ptr = reinterpret_cast<std::uintptr_t>(this); }
};

inline std::shared_ptr<FactorType> CKDEType::opposite_semiparametric() const override { return CKDEType::get(); }

class DiscreteFactorType : public FactorType {
public:
    DiscreteFactorType(const DiscreteFactorType&) = delete;
    void operator=(const DiscreteFactorType&) = delete;

    static std::shared_ptr<FactorType> get() {
        static std::shared_ptr<FactorType> singleton = std::shared_ptr<DiscreteFactorType>(new DiscreteFactorType);
        return singleton;
    }

    std::shared_ptr<Factor> new_factor(const std::string& variable,
                                       const std::vector<std::string>& parents) const override {
        return std::make_shared<DiscreteFactor>(variable, parents);
    }

    std::shared_ptr<FactorType> opposite_semiparametric() const override { return nullptr; }

    std::string ToString() const override { return "DiscreteFactor"; }

private:
    DiscreteFactorType() { m_ptr = reinterpret_cast<std::uintptr_t>(this); }
};

}  // namespace factors

#endif  // PYBNESIAN_FACTORS_FACTORS_HPP