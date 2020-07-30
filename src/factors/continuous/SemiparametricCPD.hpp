#ifndef PGM_DATASET_SEMIPARAMETRICCPD_HPP
#define PGM_DATASET_SEMIPARAMETRICCPD_HPP

#include <factors/factors.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>

using factors::continuous::LinearGaussianCPD;
using factors::continuous::CKDE;
using factors::FactorType;

namespace factors::continuous {

    class SemiparametricCPD {
    public:
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

        FactorType node_type() const {
            if (std::holds_alternative<LinearGaussianCPD>(m_cpd))
                return FactorType::LinearGaussianCPD;
            else
                return FactorType::CKDE;
        }

        LinearGaussianCPD& as_lg() {
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

        std::string ToString() const;
    private:
        std::variant<LinearGaussianCPD, CKDE> m_cpd;
    };
}

#endif //PGM_DATASET_SEMIPARAMETRICCPD_HPP