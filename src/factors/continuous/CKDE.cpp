#include <factors/continuous/CKDE.hpp>
#include <opencl/opencl_config.hpp>
#include <util/validate_dtype.hpp>

using opencl::OpenCLConfig;

namespace factors::continuous {

    void KDE::copy_bandwidth_opencl() {
        auto d = m_variables.size();
        auto llt_cov = m_bandwidth.llt();
        auto llt_matrix = llt_cov.matrixLLT();
        
        m_lognorm_const = -llt_matrix.diagonal().array().log().sum() 
                          - 0.5 * m_variables.size() * std::log(2*util::pi<double>) 
                          - std::log(N);


        auto& opencl = OpenCLConfig::get();

        switch (m_training_type) {
            case Type::DOUBLE: {
                m_H_cholesky = opencl.copy_to_buffer(llt_matrix.data(), d*d);
                break;
            }
            case Type::FLOAT: {
                MatrixXf casted_cholesky = llt_matrix.template cast<float>();
                m_H_cholesky = opencl.copy_to_buffer(casted_cholesky.data(), d*d);
                break;
            }
            default:
                throw std::invalid_argument("Unreachable code.");
                
        }
    }

    DataFrame KDE::training_data() const {
        switch(m_training_type) {
            case Type::DOUBLE:
                return _training_data<arrow::DoubleType>();
            case Type::FLOAT:
                return _training_data<arrow::FloatType>();
            default:
                throw std::invalid_argument("Unreachable code.");
        }
    }

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

        m_fitted = true;
    }

    VectorXd KDE::logl(const DataFrame& df) const {
        // FIXME: Check the model is fitted.
        auto type_id = df.same_type(m_variables);

        if (type_id != m_training_type) {
            throw std::invalid_argument("Data type of training and test datasets is different.");
        }

        switch(type_id) {
            case Type::DOUBLE:
                return _logl<arrow::DoubleType>(df);
            case Type::FLOAT:
                return _logl<arrow::FloatType>(df);
            default:
                throw std::runtime_error("Unreachable code.");
        }
    }

    double KDE::slogl(const DataFrame& df) const {
        auto type_id = df.same_type(m_variables);

        if (type_id != m_training_type) {
            throw std::invalid_argument("Data type of training and test datasets is different.");
        }

        switch(type_id) {
            case Type::DOUBLE:
                return _slogl<arrow::DoubleType>(df);
            case Type::FLOAT:
                return _slogl<arrow::FloatType>(df);
            default:
                throw std::runtime_error("Unreachable code.");
        }
    }

    py::tuple KDE::__getstate__() const {
        switch (m_training_type) {
            case Type::DOUBLE:
                return __getstate__<arrow::DoubleType>();
            case Type::FLOAT:
                return __getstate__<arrow::FloatType>();
        }
    }

    KDE KDE::__setstate__(py::tuple& t) {
        if (t.size() != 8)
            throw std::runtime_error("Not valid KDE.");

        KDE kde(t[0].cast<std::vector<std::string>>());

        kde.m_fitted = t[1].cast<bool>();

        if (kde.m_fitted) {
            kde.m_bselector = static_cast<KDEBandwidth>(t[2].cast<int>());
            kde.m_bandwidth = t[3].cast<MatrixXd>();
            kde.m_lognorm_const = t[5].cast<double>();
            kde.N = t[6].cast<size_t>();
            kde.m_training_type = static_cast<arrow::Type::type>(t[7].cast<int>());

            auto llt_cov = kde.m_bandwidth.llt();
            auto llt_matrix = llt_cov.matrixLLT();

            auto& opencl = OpenCLConfig::get();

            switch (kde.m_training_type) {
                case Type::DOUBLE: {
                    kde.m_H_cholesky = opencl.copy_to_buffer(llt_matrix.data(), 
                                                             kde.m_variables.size()*kde.m_variables.size());
                    
                    auto training_data = t[4].cast<VectorXd>();
                    kde.m_training = opencl.copy_to_buffer(training_data.data(), 
                                                            kde.N*kde.m_variables.size());
                    break;
                }
                case Type::FLOAT: {
                    MatrixXf casted_cholesky = llt_matrix.template cast<float>();
                    kde.m_H_cholesky = opencl.copy_to_buffer(casted_cholesky.data(), 
                                                             kde.m_variables.size()*kde.m_variables.size());

                    auto training_data = t[4].cast<VectorXf>();
                    kde.m_training = opencl.copy_to_buffer(training_data.data(), 
                                                            kde.N*kde.m_variables.size());
                    break;
                }
                default:
                    throw std::runtime_error("Not valid data type in KDE.");
            }
        }

        return kde;
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

    VectorXd CKDE::logl(const DataFrame& df) const {
        // FIXME: Check the model is fitted.
        auto type_id = df.same_type(m_variables);

        if (type_id != m_training_type) {
            throw std::invalid_argument("Data type of training and test datasets is different.");
        }

        switch(type_id) {
            case Type::DOUBLE:
                return _logl<arrow::DoubleType>(df);
            case Type::FLOAT:
                return _logl<arrow::FloatType>(df);
            default:
                throw std::runtime_error("Unreachable code.");
        }
    }

    double CKDE::slogl(const DataFrame& df) const {
        auto type_id = df.same_type(m_variables);

        if (type_id != m_training_type) {
            throw std::invalid_argument("Data type of training and test datasets is different.");
        }

        switch(type_id) {
            case Type::DOUBLE:
                return _slogl<arrow::DoubleType>(df);
            case Type::FLOAT:
                return _slogl<arrow::FloatType>(df);
            default:
                throw std::runtime_error("Unreachable code.");
        }
    }

    Array_ptr CKDE::sample(int n, const DataFrame& evidence_values, long unsigned int seed) const {
        if (!m_evidence.empty()) {
            auto type_id = evidence_values.same_type(m_evidence);

            if (type_id != m_training_type) {
                throw std::invalid_argument("Data type of evidence values (" + util::to_type(type_id)->name() + 
                            ") is different from CKDE training data (" + util::to_type(m_training_type)->name() + ").");
            }
        }

        switch(m_training_type) {
            case Type::DOUBLE:
                return _sample<arrow::DoubleType>(n, evidence_values, seed);
            case Type::FLOAT:
                return _sample<arrow::FloatType>(n, evidence_values, seed);
            default:
                throw std::runtime_error("Unreachable code.");
        }
    }

    VectorXd CKDE::cdf(const DataFrame& df) const {
        auto type_id = df.same_type(m_variables);

        if (type_id != m_training_type) {
            throw std::invalid_argument("Data type of training and test datasets is different.");
        }

        switch(type_id) {
            case Type::DOUBLE:
                return _cdf<arrow::DoubleType>(df);
            case Type::FLOAT:
                return _cdf<arrow::FloatType>(df);
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

    py::tuple CKDE::__getstate__() const {
        switch (m_training_type) {
            case Type::DOUBLE:
                return __getstate__<arrow::DoubleType>();
            case Type::FLOAT:
                return __getstate__<arrow::FloatType>();
        }
    }

    CKDE CKDE::__setstate__(py::tuple& t) {
        if (t.size() != 4)
            throw std::runtime_error("Not valid CKDE.");

        CKDE ckde(t[0].cast<std::string>(), t[1].cast<std::vector<std::string>>());

        ckde.m_fitted = t[2].cast<bool>();

        if (ckde.m_fitted) {
            auto joint_tuple = t[3].cast<py::tuple>();
            auto kde_joint = KDE::__setstate__(joint_tuple);
            ckde.m_bselector = kde_joint.bandwidth_type();
            ckde.m_training_type = kde_joint.data_type();
            ckde.N = kde_joint.num_instances();
            ckde.m_joint = std::move(kde_joint);

            if (!ckde.m_evidence.empty()) {
                auto& joint_bandwidth = ckde.m_joint.bandwidth();
                auto d = ckde.m_variables.size();
                auto marg_bandwidth = joint_bandwidth.bottomRightCorner(d-1, d-1);

                cl::Buffer& training_buffer = ckde.m_joint.training_buffer();

                auto& opencl = OpenCLConfig::get();

                switch(ckde.m_training_type) {
                    case Type::DOUBLE: {
                        auto marg_buffer = opencl.copy_buffer<double>(training_buffer, ckde.N, ckde.N*(d-1));
                        ckde.m_marg.fit<arrow::DoubleType>(marg_bandwidth, marg_buffer, ckde.m_joint.data_type(), ckde.N);
                        break;
                    }
                    case Type::FLOAT: {
                        auto marg_buffer = opencl.copy_buffer<float>(training_buffer, ckde.N, ckde.N*(d-1));
                        ckde.m_marg.fit<arrow::FloatType>(marg_bandwidth, marg_buffer, ckde.m_joint.data_type(), ckde.N);
                        break;
                    }
                    default:
                        throw std::invalid_argument("Wrong data type in CKDE.");
                }
            }
        }

        return ckde;
    }
}