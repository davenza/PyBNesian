#ifndef PGM_DATASET_SEMIPARAMETRICCPD_HPP
#define PGM_DATASET_SEMIPARAMETRICCPD_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <models/SemiparametricBN_NodeType.hpp>

using factors::continuous::LinearGaussianCPD;
using factors::continuous::CKDE;
using models::NodeType;

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
        VectorXd logpdf(const DataFrame& df) const;
        double slogpdf(const DataFrame& df) const;

        NodeType node_type() const {
            if (std::holds_alternative<LinearGaussianCPD>(m_cpd))
                return NodeType::LinearGaussianCPD;
            else
                return NodeType::CKDE;
        }

        LinearGaussianCPD& as_lg() {
            try {
                return std::get<LinearGaussianCPD>(m_cpd);
            } catch(std::bad_variant_access) {
                throw py::value_error("The SemiparametricBN is not a LinearGaussianCPD");
            }
        }

        CKDE& as_ckde() {
            try {
                return std::get<CKDE>(m_cpd);
            } catch(std::bad_variant_access) {
                throw py::value_error("The SemiparametricBN is not a CKDE");
            }
        }

        std::string ToString() const;
    private:
        std::variant<LinearGaussianCPD, CKDE> m_cpd;
    };
}

#endif //PGM_DATASET_SEMIPARAMETRICCPD_HPP