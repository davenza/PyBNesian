#include <factors/continuous/CKDE.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <models/BayesianNetwork.hpp>
#include <opencl/opencl_config.hpp>
#include <util/vech_ops.hpp>
#include <util/basic_eigen_ops.hpp>

using factors::discrete::DiscreteFactorType;
using models::BayesianNetworkBase, models::ConditionalBayesianNetworkBase;
using opencl::OpenCLConfig;

namespace factors::continuous {

std::shared_ptr<Factor> CKDEType::new_factor(const BayesianNetworkBase& m,
                                             const std::string& variable,
                                             const std::vector<std::string>& evidence,
                                             py::args args,
                                             py::kwargs kwargs) const {
    for (const auto& e : evidence) {
        if (m.node_type(e) == DiscreteFactorType::get()) {
            return generic_new_factor<HCKDE>(variable, evidence, args, kwargs);
        }
    }

    return generic_new_factor<CKDE>(variable, evidence, args, kwargs);
}

std::shared_ptr<Factor> CKDEType::new_factor(const ConditionalBayesianNetworkBase& m,
                                             const std::string& variable,
                                             const std::vector<std::string>& evidence,
                                             py::args args,
                                             py::kwargs kwargs) const {
    for (const auto& e : evidence) {
        if (m.node_type(e) == DiscreteFactorType::get()) {
            return generic_new_factor<HCKDE>(variable, evidence, args, kwargs);
        }
    }

    return generic_new_factor<CKDE>(variable, evidence, args, kwargs);
}

void CKDE::fit(const DataFrame& df) {
    auto type = df.same_type(m_variables);

    m_training_type = type;
    switch (type->id()) {
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
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _logl<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _logl<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

double CKDE::slogl(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
        case Type::DOUBLE:
            return _slogl<arrow::DoubleType>(df);
        case Type::FLOAT:
            return _slogl<arrow::FloatType>(df);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

Array_ptr CKDE::sample(int n, const DataFrame& evidence_values, unsigned int seed) const {
    if (n < 0) {
        throw std::invalid_argument("n should be a non-negative number");
    }

    check_fitted();
    if (!this->evidence().empty()) {
        auto type = evidence_values.same_type(this->evidence());

        if (type->id() != m_training_type->id()) {
            throw std::invalid_argument("Data type of evidence values (" + type->name() +
                                        ") is different from CKDE training data (" + m_training_type->name() + ").");
        }
    }

    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return _sample<arrow::DoubleType>(n, evidence_values, seed);
        case Type::FLOAT:
            return _sample<arrow::FloatType>(n, evidence_values, seed);
        default:
            throw std::runtime_error("Unreachable code.");
    }
}

VectorXd CKDE::cdf(const DataFrame& df) const {
    check_fitted();
    auto type = df.same_type(m_variables);

    if (type->id() != m_training_type->id()) {
        throw std::invalid_argument("Data type of training and test datasets is different.");
    }

    switch (type->id()) {
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
    const auto& e = this->evidence();
    if (!e.empty()) {
        stream << "[CKDE] P(" << this->variable() << " | " << e[0];

        for (size_t i = 1; i < e.size(); ++i) {
            stream << ", " << e[i];
        }

        if (m_fitted)
            stream << ") = CKDE with " << N << " instances";
        else
            stream << ") not fitted";
        return stream.str();
    } else {
        if (m_fitted)
            stream << "[CKDE] P(" << this->variable() << ") = CKDE with " << N << " instances";
        else
            stream << "[CKDE] P(" << this->variable() << ") not fitted";
        return stream.str();
    }
}

py::tuple CKDE::__getstate__() const {
    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return __getstate__<arrow::DoubleType>();
        case Type::FLOAT:
            return __getstate__<arrow::FloatType>();
        default:
            // Not fitted model.
            return __getstate__<arrow::DoubleType>();
    }
}

CKDE CKDE::__setstate__(py::tuple& t) {
    if (t.size() != 4) throw std::runtime_error("Not valid CKDE.");

    CKDE ckde(t[0].cast<std::string>(), t[1].cast<std::vector<std::string>>());

    ckde.m_fitted = t[2].cast<bool>();

    if (ckde.m_fitted) {
        auto joint_tuple = t[3].cast<py::tuple>();
        auto kde_joint = KDE::__setstate__(joint_tuple);
        ckde.m_bselector = kde_joint.bandwidth_type();
        ckde.m_training_type = kde_joint.data_type();
        ckde.N = kde_joint.num_instances();
        ckde.m_joint = std::move(kde_joint);

        if (!ckde.evidence().empty()) {
            auto& joint_bandwidth = ckde.m_joint.bandwidth();
            auto d = ckde.m_variables.size();
            auto marg_bandwidth = joint_bandwidth.bottomRightCorner(d - 1, d - 1);

            cl::Buffer& training_buffer = ckde.m_joint.training_buffer();

            auto& opencl = OpenCLConfig::get();

            switch (ckde.m_training_type->id()) {
                case Type::DOUBLE: {
                    auto marg_buffer = opencl.copy_buffer<double>(training_buffer, ckde.N, ckde.N * (d - 1));
                    ckde.m_marg.fit<arrow::DoubleType>(marg_bandwidth, marg_buffer, ckde.m_joint.data_type(), ckde.N);
                    break;
                }
                case Type::FLOAT: {
                    auto marg_buffer = opencl.copy_buffer<float>(training_buffer, ckde.N, ckde.N * (d - 1));
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

}  // namespace factors::continuous