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

    void SemiparametricCPD::fit(py::handle pyobject) {
        auto rb = dataset::to_record_batch(pyobject);
        auto df = DataFrame(rb);
        fit(df);
    }

    void SemiparametricCPD::fit(const DataFrame& df) {
        std::visit([&df](auto& cpd) {
            cpd.fit(df);
        }, m_cpd);
    }

    VectorXd SemiparametricCPD::logpdf(py::handle pyobject) const {
        auto rb = dataset::to_record_batch(pyobject);
        auto df = DataFrame(rb);
        return std::move(logpdf(df));
    }

    VectorXd SemiparametricCPD::logpdf(const DataFrame& df) const {
        return std::visit([&df](auto& cpd) -> VectorXd&& {
                    return std::move(cpd.logpdf(df));
                }, m_cpd);
    }

    double SemiparametricCPD::slogpdf(py::handle pyobject) const {
        auto rb = dataset::to_record_batch(pyobject);
        auto df = DataFrame(rb);
        return slogpdf(df);
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