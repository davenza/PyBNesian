#include <factors/continuous/CKDE.hpp>
#include <opencl/opencl_config.hpp>


using opencl::OpenCLConfig;

namespace factors::continuous {


    void KDE::fit(py::handle pyobject) {
        auto rb = dataset::to_record_batch(pyobject);
        auto df = DataFrame(rb);
        fit(df);
    }

    
    void KDE::fit(const DataFrame& df) {
        m_training_type = df.same_type(m_variables);
        bool contains_null = df.null_count(m_variables);

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
                throw py::value_error("Wrong data type to fit CKDE. [double] or [float] data is expected.");
        }
    }

    VectorXd KDE::logpdf(py::handle pyobject) const {
        auto rb = dataset::to_record_batch(pyobject);
        auto df = DataFrame(rb);

        auto l = logpdf(df);
        return l;
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
}