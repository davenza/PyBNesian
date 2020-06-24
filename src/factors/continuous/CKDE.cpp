#include <factors/continuous/CKDE.hpp>
#include <opencl/opencl_config.hpp>


using opencl::OpenCLConfig;

namespace factors::continuous {

    void KDE::fit(const DataFrame& df) {
        m_training_type = df.same_type(m_variables);

        bool contains_null = df.null_count(m_variables) > 0;

        switch(m_training_type) {
            case Type::DOUBLE: {
                if (contains_null)
                    _fit<arrow::DoubleType, true>(df);
                else 
                    _fit<arrow::DoubleType, false>(df);
                break;
            }
            case Type::FLOAT: {
                if (contains_null)
                    _fit<arrow::FloatType, true>(df);
                else 
                    _fit<arrow::FloatType, false>(df);
                break;
            }
            default:
                throw py::value_error("Wrong data type to fit KDE. [double] or [float] data is expected.");
        }
    }

    VectorXd KDE::logpdf(const DataFrame& df) const {
        // FIXME: Check the model is fitted.
        auto type_id = df.same_type(m_variables);

        if (type_id != m_training_type) {
            throw std::invalid_argument("Data type of training and test datasets is different.");
        }

        switch(type_id) {
            case Type::DOUBLE:
                return _logpdf<arrow::DoubleType>(df);
            case Type::FLOAT:
                return _logpdf<arrow::FloatType>(df);
            default:
                throw std::runtime_error("Unreachable code.");
        }
    }

    double KDE::slogpdf(const DataFrame& df) const {
        auto type_id = df.same_type(m_variables);

        if (type_id != m_training_type) {
            throw std::invalid_argument("Data type of training and test datasets is different.");
        }

        switch(type_id) {
            case Type::DOUBLE:
                return _slogpdf<arrow::DoubleType>(df);
            case Type::FLOAT:
                return _slogpdf<arrow::FloatType>(df);
            default:
                throw std::runtime_error("Unreachable code.");
        }
    }

    void CKDE::fit(const DataFrame& df) {

        auto type = df.same_type(m_variables);

        m_training_type = type;
        switch(type) {
            case Type::DOUBLE:
                _fit<arrow::DoubleType>(df);
                break;
            case Type::FLOAT:
                _fit<arrow::FloatType>(df);
                break;
            default:
                throw std::invalid_argument("Wrong data type to fit KDE. [double] or [float] data is expected.");
        }

        m_fitted = true;
    }

    VectorXd CKDE::logpdf(const DataFrame& df) const {
        // FIXME: Check the model is fitted.
        auto type_id = df.same_type(m_variables);

        if (type_id != m_training_type) {
            throw std::invalid_argument("Data type of training and test datasets is different.");
        }

        switch(type_id) {
            case Type::DOUBLE:
                return _logpdf<arrow::DoubleType>(df);
            case Type::FLOAT:
                return _logpdf<arrow::FloatType>(df);
            default:
                throw std::runtime_error("Unreachable code.");
        }
    }

    double CKDE::slogpdf(const DataFrame& df) const {
        auto type_id = df.same_type(m_variables);

        if (type_id != m_training_type) {
            throw std::invalid_argument("Data type of training and test datasets is different.");
        }

        switch(type_id) {
            case Type::DOUBLE:
                return _slogpdf<arrow::DoubleType>(df);
            case Type::FLOAT:
                return _slogpdf<arrow::FloatType>(df);
            default:
                throw std::runtime_error("Unreachable code.");
        }
    }

    std::string CKDE::ToString() const {
        std::stringstream stream;
        if (!m_evidence.empty()) {
            stream << "[CKDE] P(" << m_variable << " | " << m_evidence[0];
            for (auto& ev : m_evidence) {
                stream << ", " << ev;
            }
            if (m_fitted)
                stream << ") = CKDE with " << N << " instances";
            else
                stream << ") not fitted";
            return stream.str();
        } else {
            if (m_fitted)
                stream << "[CKDE] P(" << m_variable << ") = CKDE with " << N << " instances";
            else 
                stream << "[CKDE] P(" << m_variable << ") not fitted";
            return stream.str();
        }
    }
}