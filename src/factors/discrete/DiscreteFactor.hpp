#ifndef PGM_DATASET_DISCRETE_FACTOR_HPP
#define PGM_DATASET_DISCRETE_FACTOR_HPP

#include <dataset/dataset.hpp>
#include <Eigen/Dense>

using dataset::DataFrame;
using Eigen::VectorXd, Eigen::VectorXi;

namespace factors::discrete {

    struct DiscreteFactor_Params {
        VectorXd values;
        VectorXi cardinality;
        VectorXi strides;
    };

    class DiscreteFactor {
    public:
        using ParamsClass = DiscreteFactor_Params;

        DiscreteFactor(std::string variable, std::vector<std::string> evidence) : m_variable(variable),
                                                                                  m_evidence(evidence),
                                                                                  m_variable_values(),
                                                                                  m_evidence_values(),
                                                                                  m_values(),
                                                                                  m_cardinality(),
                                                                                  m_strides() {}


        const std::string& variable() const { return m_variable; }
        const std::vector<std::string>& evidence() const { return m_evidence; }

        void fit(const DataFrame& df);

    private:
        std::string m_variable;
        std::vector<std::string> m_evidence;
        std::vector<std::string> m_variable_values;
        std::vector<std::vector<std::string>> m_evidence_values;
        VectorXd m_values;
        VectorXi m_cardinality;
        VectorXi m_strides;
    };
}

#endif //PGM_DATASET_DISCRETE_FACTOR_HPP