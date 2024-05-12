#include <kde/ProductKDE.hpp>
#include <util/arrow_types.hpp>

namespace kde {

void ProductKDE::copy_bandwidth_opencl() {
    m_cl_bandwidth.clear();
    auto& opencl = OpenCLConfig::get();

    for (size_t i = 0; i < m_variables.size(); ++i) {
        switch (m_training_type->id()) {
            case Type::DOUBLE: {
                auto sqrt = std::sqrt(m_bandwidth(i));
                m_cl_bandwidth.push_back(opencl.copy_to_buffer(&sqrt, 1));
                break;
            }
            case Type::FLOAT: {
                auto casted = std::sqrt(static_cast<float>(m_bandwidth(i)));
                m_cl_bandwidth.push_back(opencl.copy_to_buffer(&casted, 1));
                break;
            }
            default:
                throw std::invalid_argument("Unreachable code.");
        }
    }

    m_lognorm_const = -0.5 * m_variables.size() * std::log(2 * util::pi<double>) -
                      0.5 * m_bandwidth.array().log().sum() - std::log(N);
}

DataFrame ProductKDE::training_data() const {
    check_fitted();
    switch (m_training_type->id()) {
        case Type::DOUBLE:
            return _training_data<arrow::DoubleType>();
        case Type::FLOAT:
            return _training_data<arrow::FloatType>();
        default:
            throw std::invalid_argument("Unreachable code.");
    }
}

void ProductKDE::fit(const DataFrame& df) {
    m_training_type = df.same_type(m_variables);

    bool contains_null = df.null_count(m_variables) > 0;

    switch (m_training_type->id()) {
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
            throw std::invalid_argument("Wrong data type to fit ProductKDE. [double] or [float] data is expected.");
    }

    m_fitted = true;
}

VectorXd ProductKDE::logl(const DataFrame& df) const {
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

double ProductKDE::slogl(const DataFrame& df) const {
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

py::tuple ProductKDE::__getstate__() const {
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

ProductKDE ProductKDE::__setstate__(py::tuple& t) {
    if (t.size() != 8) throw std::runtime_error("Not valid ProductKDE.");

    ProductKDE kde(t[0].cast<std::vector<std::string>>());

    kde.m_fitted = t[1].cast<bool>();
    kde.m_bselector = t[2].cast<std::shared_ptr<BandwidthSelector>>();
    BandwidthSelector::keep_python_alive(kde.m_bselector);

    if (kde.m_fitted) {
        kde.m_bandwidth = t[3].cast<VectorXd>();
        kde.m_lognorm_const = t[5].cast<double>();
        kde.N = static_cast<size_t>(t[6].cast<int>());
        kde.m_training_type = util::GetPrimitiveType(static_cast<arrow::Type::type>(t[7].cast<int>()));

        auto& opencl = OpenCLConfig::get();

        switch (kde.m_training_type->id()) {
            case Type::DOUBLE: {
                auto data = t[4].cast<std::vector<VectorXd>>();

                for (size_t i = 0; i < kde.m_variables.size(); ++i) {
                    kde.m_cl_bandwidth.push_back(opencl.copy_to_buffer(&kde.m_bandwidth(i), 1));
                    kde.m_training.push_back(opencl.copy_to_buffer(data[i].data(), kde.N));
                }

                break;
            }
            case Type::FLOAT: {
                auto data = t[4].cast<std::vector<VectorXf>>();

                for (size_t i = 0; i < kde.m_variables.size(); ++i) {
                    auto casted_bw = static_cast<float>(kde.m_bandwidth(i));

                    kde.m_cl_bandwidth.push_back(opencl.copy_to_buffer(&casted_bw, 1));
                    kde.m_training.push_back(opencl.copy_to_buffer(data[i].data(), kde.N));
                }

                break;
            }
            default:
                throw std::runtime_error("Not valid data type in ProductKDE.");
        }
    }

    return kde;
}

}  // namespace kde