#ifndef PYBNESIAN_FACTORS_CONTINUOUS_SEMIPARAMETRICCPD_HPP
#define PYBNESIAN_FACTORS_CONTINUOUS_SEMIPARAMETRICCPD_HPP

#include <variant>
#include <factors/factors.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>

using factors::continuous::LinearGaussianCPD;
using factors::continuous::CKDE;
using factors::FactorType;

namespace factors::continuous {

    class SemiparametricCPD {
    public:

        SemiparametricCPD() = default;

        template<typename T>
        SemiparametricCPD(T factor) : m_cpd(factor) {
            static_assert(std::is_same_v<LinearGaussianCPD, T> ||
                        std::is_same_v<CKDE, T>, "SemiparametricCPD only allows LinearGaussianCPD or CKDE.");
        }

        const std::string& variable() const;
        const std::vector<std::string>& evidence() const;
        bool fitted() const;

        void fit(const DataFrame& df);
        VectorXd logl(const DataFrame& df) const;
        double slogl(const DataFrame& df) const;
        VectorXd cdf(const DataFrame& df) const;

        FactorType factor_type() const {
            if (std::holds_alternative<LinearGaussianCPD>(m_cpd))
                return FactorType::LinearGaussianCPD;
            else
                return FactorType::CKDE;
        }

        arrow::Type::type arrow_type() const;

        LinearGaussianCPD& as_lg() {
            try {
                return std::get<LinearGaussianCPD>(m_cpd);
            } catch(std::bad_variant_access&) {
                throw py::value_error("The SemiparametricBN is not a LinearGaussianCPD");
            }
        }

        const LinearGaussianCPD& as_lg() const {
            try {
                return std::get<LinearGaussianCPD>(m_cpd);
            } catch(std::bad_variant_access&) {
                throw py::value_error("The SemiparametricBN is not a LinearGaussianCPD");
            }
        }

        CKDE& as_ckde() {
            try {
                return std::get<CKDE>(m_cpd);
            } catch(std::bad_variant_access&) {
                throw py::value_error("The SemiparametricBN is not a CKDE");
            }
        }

        const CKDE& as_ckde() const {
            try {
                return std::get<CKDE>(m_cpd);
            } catch(std::bad_variant_access&) {
                throw py::value_error("The SemiparametricBN is not a CKDE");
            }
        }

        Array_ptr sample(int n, const DataFrame& evidence_values, 
                         unsigned int seed = std::random_device{}()) const;

        std::string ToString() const;

        py::tuple __getstate__() const;
        static SemiparametricCPD __setstate__(py::tuple& t);
        static SemiparametricCPD __setstate__(py::tuple&& t) {
            return __setstate__(t);
        }
    private:
        std::variant<LinearGaussianCPD, CKDE> m_cpd;
    };
}

#endif //PYBNESIAN_FACTORS_CONTINUOUS_SEMIPARAMETRICCPD_HPP
