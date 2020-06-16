#ifndef PGM_DATASET_SEMIPARAMETRICCPD_HPP
#define PGM_DATASET_SEMIPARAMETRICCPD_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>

using factors::continuous::LinearGaussianCPD;
using factors::continuous::CKDE;

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

        void fit(py::handle pyobject);
        void fit(const DataFrame& df);

        VectorXd logpdf(py::handle pyobject) const;
        VectorXd logpdf(const DataFrame& df) const;

        double slogpdf(py::handle pyobject) const;
        double slogpdf(const DataFrame& df) const;

    private:
        std::variant<LinearGaussianCPD, CKDE> m_cpd;
    };

}

#endif //PGM_DATASET_SEMIPARAMETRICCPD_HPP