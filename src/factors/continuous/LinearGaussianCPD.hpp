#ifndef PYBNESIAN_FACTORS_CONTINUOUS_LINEARGAUSSIANCPD_HPP
#define PYBNESIAN_FACTORS_CONTINUOUS_LINEARGAUSSIANCPD_HPP

#include <random>
#include <factors/factors.hpp>
#include <dataset/dataset.hpp>

using Eigen::VectorXd;
using dataset::DataFrame;
using factors::NodeType;

namespace py = pybind11;

namespace factors::continuous {

    struct LinearGaussianCPD_Params {
        VectorXd beta;
        double variance;
    };

    class LinearGaussianCPD {
    public:
        using ParamsClass = LinearGaussianCPD_Params;
        
        LinearGaussianCPD() = default;
        LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence);
        LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence,
                            const std::vector<double> beta, const double variance);


        NodeType node_type() const {
            return NodeType::LinearGaussianCPD;
        }

        arrow::Type::type arrow_type() const {
            return Type::DOUBLE;
        }
        
        const std::string& variable() const { return m_variable; }
        const std::vector<std::string>& evidence() const { return m_evidence; }
        
        bool fitted() const { return m_fitted; }
        void fit(const DataFrame& df);
        VectorXd logl(const DataFrame& df) const;
        double slogl(const DataFrame& df) const;
        VectorXd cdf(const DataFrame& df) const;

        std::string ToString() const;

        const VectorXd& beta() const { return m_beta; }
        void set_beta(const VectorXd& new_beta) { 
            if (static_cast<size_t>(new_beta.rows()) != (m_evidence.size() + 1))
                throw std::invalid_argument("Wrong number of elements for the beta vector: " + std::to_string(new_beta.rows()) + 
                                        ". Expected size: " + std::to_string((m_evidence.size() + 1)));
            m_beta = new_beta; 
        }

        double variance() const { return m_variance; }
        void set_variance(double v) { m_variance = v; }

        Array_ptr sample(int n, 
                         const DataFrame& evidence_values, 
                         unsigned int seed = std::random_device{}()) const;
        
        void save(const std::string name) {
            save_factor(*this, name);
        }

        py::tuple __getstate__() const;
        static LinearGaussianCPD __setstate__(py::tuple& t);
        static LinearGaussianCPD __setstate__(py::tuple&& t) {
            return __setstate__(t);
        }
    private:
        std::string m_variable;
        std::vector<std::string> m_evidence;
        bool m_fitted;
        VectorXd m_beta;
        double m_variance;
    };

}

#endif //PYBNESIAN_FACTORS_CONTINUOUS_LINEARGAUSSIANCPD_HPP
