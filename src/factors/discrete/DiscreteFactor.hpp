#ifndef PGM_DATASET_DISCRETE_FACTOR_HPP
#define PGM_DATASET_DISCRETE_FACTOR_HPP


#include <iostream>
#include <dataset/dataset.hpp>
#include <Eigen/Dense>

using dataset::DataFrame;
using Eigen::VectorXd, Eigen::VectorXi;

namespace factors::discrete {

    struct DiscreteFactor_Params {
        VectorXd prob;
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
                                                                                  m_prob(),
                                                                                  m_cardinality(),
                                                                                  m_strides() {}


        const std::string& variable() const { return m_variable; }
        const std::vector<std::string>& evidence() const { return m_evidence; }

        void fit(const DataFrame& df);

        void print_variable_values() {
            if (m_variable_values.empty()) {
                std::cout << "No known values for variable " << m_variable << std::endl;
            } else {
                std::cout << "Values for variable " << m_variable << ": " << m_variable_values[0];
                for (auto it = m_variable_values.begin() + 1; it != m_variable_values.end(); ++it) {
                    std::cout << ", " << *it;
                }
                std::cout << std::endl;
            }
        }

        void print_evidence_values() {
            if (m_evidence_values.empty()) {
                std::cout << "No known evidence." << std::endl;
            } else {
                
                int i = 0;
                for (auto it = m_evidence.begin(); it != m_evidence.end(); ++it, ++i) {
                    std::cout << "Values for evidence variable " << *it << ": " << m_evidence_values[i][0];

                    for (auto it2 = m_evidence_values[i].begin() + 1; it2 != m_evidence_values[i].end(); ++it2) {
                        std::cout << ", " << *it2;
                    }
                    std::cout << std::endl;

                }
            }
        }

        void print_prob() {
            std::cout << "Probability values: " << std::endl;
            std::cout << m_prob << std::endl;
        }

        void print_cardinality() {
            std::cout << "Cardinality: " << std::endl;
            std::cout << m_cardinality << std::endl;
        }

        void print_strides() {
            std::cout << "Strides: " << std::endl;
            std::cout << m_strides << std::endl;
        }

    private:
        std::string m_variable;
        std::vector<std::string> m_evidence;
        std::vector<std::string> m_variable_values;
        std::vector<std::vector<std::string>> m_evidence_values;
        VectorXd m_prob;
        VectorXi m_cardinality;
        VectorXi m_strides;
    };
}

#endif //PGM_DATASET_DISCRETE_FACTOR_HPP