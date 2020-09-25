#ifndef PGM_DATASET_NORMAL_HPP
#define PGM_DATASET_NORMAL_HPP

#include <cmath>
#include <util/math_constants.hpp>
#include <Eigen/Dense>

using Eigen::VectorXd;

namespace distributions {


    struct UnivariateNormalSS {
        double x;
        double x2;
    };

    class Normal {
    public:  
        Normal(const std::string& variable, double mean, double variance) : m_variable(variable),
                                                                            m_mean(mean),
                                                                            m_variance(variance),
                                                                            m_natural1(mean / variance),
                                                                            m_natural2(-0.5 / variance) {}
    
    

        double logl(double v) {
            double d = m_mean - v;
            return -0.5*d*d / m_variance - 0.5*std::log(m_variance) - 0.5*std::log(2*util::pi<double>);
        }

        double natural_logl(double v) {
            return m_natural1*v + m_natural2*v*v + 0.25*m_natural1*m_natural1 / m_natural2 + 0.5*std::log(-2*m_natural2)
                    - 0.5*std::log(2*util::pi<double>);
        }


        UnivariateNormalSS fixed_ss(double s) {
            return UnivariateNormalSS {
                .x = s,
                .x2 = s * s
            };
        }

        UnivariateNormalSS expected_ss() {
            return UnivariateNormalSS {
                .x = m_mean,
                .x2 = m_mean*m_mean + m_variance
            };
        }

        void update(double mean, double variance) {
            m_mean = mean;
            m_variance = variance;
            m_natural1 = mean / variance;
            m_natural2 = -0.5 / variance;
        }
        
    private:
        std::string m_variable;
        double m_mean;
        double m_variance;
        double m_natural1;
        double m_natural2;
    };


    class ConditionalNormal {
    public:  
        ConditionalNormal(const std::string& variable, const std::vector<std::string>& evidence,
                            const std::vector<double> beta, const double variance) : m_variable(variable),
                                                                                     m_evidence(evidence),
                                                                                     m_beta(beta.size()),
                                                                                     m_variance(variance) {
            auto m_ptr = m_beta.data();
            auto vec_ptr = beta.data();
            std::memcpy(m_ptr, vec_ptr, sizeof(double) * beta.size());
        }
    

        double logl(double v) {
            // double d = m_mean - v;
            // return -0.5*d*d / m_variance - 0.5*std::log(m_variance) - 0.5*std::log(2*util::pi<double>);
        }

        double natural_logl(double v) {
            // return m_natural1*v + m_natural2*v*v + 0.25*m_natural1*m_natural1 / m_natural2 + 0.5*std::log(-2*m_natural2)
            //         - 0.5*std::log(2*util::pi<double>);
        }


        // UnivariateNormalSS fixed_ss(double s) {
        //     return UnivariateNormalSS {
        //         .x = s,
        //         .x2 = s * s
        //     };
        // }

        // UnivariateNormalSS expected_ss() {
        //     return UnivariateNormalSS {
        //         .x = m_mean,
        //         .x2 = m_mean*m_mean + m_variance
        //     };
        // }

        // void update(double mean, double variance) {
        //     m_mean = mean;
        //     m_variance = variance;
        //     m_natural1 = mean / variance;
        //     m_natural2 = -0.5 / variance;
        // }
        
    private:
        std::string m_variable;
        std::vector<std::string> m_evidence;
        VectorXd m_beta;
        double m_variance;
    };


}

#endif //PGM_DATASET_NORMAL_HPP