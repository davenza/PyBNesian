#include <factors/continuous/SemiparametricCPD.hpp>


namespace factors::continuous {


    const std::string& SemiparametricCPD::variable() const { 
        return std::visit([](auto& cpd) -> const std::string& {
                    return cpd.variable();
                }, m_cpd);
    }

    const std::vector<std::string>& SemiparametricCPD::evidence() const { 
        return std::visit([](auto& cpd) -> const std::vector<std::string>& {
                    return cpd.evidence();
                }, m_cpd);
    }

    bool SemiparametricCPD::fitted() const { 
        return std::visit([](auto& cpd) {
                    return cpd.fitted();
                }, m_cpd);
    }

    void SemiparametricCPD::fit(const DataFrame& df) {
        std::visit([&df](auto& cpd) {
            cpd.fit(df);
        }, m_cpd);
    }

    VectorXd SemiparametricCPD::logpdf(const DataFrame& df) const {
        return std::visit([&df](auto& cpd) -> VectorXd {
                            return cpd.logpdf(df);
                        }, m_cpd);
    }

    double SemiparametricCPD::slogpdf(const DataFrame& df) const {
        return std::visit([&df](auto& cpd) {
                    return cpd.slogpdf(df);
                }, m_cpd);
    }

    std::string SemiparametricCPD::ToString() const {
        return std::visit([](auto& cpd) {
                    return cpd.ToString();
                }, m_cpd);
    }
}