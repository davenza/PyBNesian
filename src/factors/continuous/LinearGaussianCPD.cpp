#include <iostream>
#include <Python.h>
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>
#include <util/bit_util.hpp>
#include <Eigen/Dense>

#include <arrow/compute/kernels/cast.h>
#include <arrow/compute/context.h>
#include <chrono>
#include <iomanip>

namespace py = pybind11;
namespace pyarrow = arrow::py;

using arrow::Type;
using Eigen::Matrix, Eigen::Dynamic, Eigen::Map, Eigen::MatrixBase;

using dataset::DataFrame;

typedef std::shared_ptr<arrow::Array> Array_ptr;


#define BENCHMARK_PRE(N)                                                                                                                     \
            {                                                                                                                                \
                auto t1 = std::chrono::high_resolution_clock::now();                                                                         \
                for (auto i = 0; i < N; ++i) {                                                                                               \

#define BENCHMARK_POST(N)                                                                                                                    \
                }                                                                                                                            \
                auto t2 = std::chrono::high_resolution_clock::now();                                                                         \
                std::cout << "Time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count() / N) << " ns" << std::endl;\
            }                                                                                                                                \

#define BENCHMARK_CODE(code, N)                                                                                                              \
            {                                                                                                                                \
                auto t1 = std::chrono::high_resolution_clock::now();                                                                         \
                for (auto i = 0; i < N; ++i) {                                                                                               \
                    code                                                                                                                     \
                }                                                                                                                            \
                auto t2 = std::chrono::high_resolution_clock::now();                                                                         \
                std::cout << "Time (" << #code << "): " <<                                                                                   \
                        (std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count() / N) << " ns" << std::endl;                 \
            }                                                                                                                                \


namespace factors::continuous {


    LinearGaussianCPD::LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence) :
    m_variable(variable),
    m_evidence(evidence)
    {
        m_beta = VectorXd(evidence.size() + 1);
    };

    LinearGaussianCPD::LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence,
                                         const std::vector<double> beta, double variance) :
    m_variable(variable),
    m_evidence(evidence),
    m_variance(variance)
//    TODO: Error checking: Length of vectors
    {
        m_beta = VectorXd(beta.size());
        auto m_ptr = m_beta.data();
        auto vec_ptr = beta.data();
        std::memcpy(m_ptr, vec_ptr, sizeof(double) * beta.size());
    };

    void LinearGaussianCPD::fit(py::handle dataset) {
        auto rb = dataset::to_record_batch(dataset);
        auto df = DataFrame(rb);

        auto var_col = df->GetColumnByName(m_variable);

        if (var_col) {
            auto type = var_col->type();
            auto type_id = type->id();
            bool contains_null = (var_col->null_count() != 0);

            if (type_id != Type::DOUBLE && type_id != Type::FLOAT) {
                throw py::value_error("Wrong data type to fit the linear regression. (double) or (float) data is expected.");
            }

            for (auto &ev : m_evidence) {
                auto ev_col = df->GetColumnByName(ev);

                contains_null = (contains_null || (ev_col->null_count() != 0));
                if (!ev_col) {
                    throw py::value_error("Variable \"" + ev + "\" not found in dataset.");
                } else if (!type->Equals(ev_col->type())){
                    throw py::value_error("\"" + m_variable + "\" has different data type than \"" + ev + "\"");
                }
            }

            switch(type_id) {
                case Type::DOUBLE: {
                    if (contains_null)
                        _fit<DoubleType, true>(df);
                    else
                        _fit<DoubleType, false>(df);
                    break;
                }
                case Type::FLOAT: {
                    if (contains_null)
                        _fit<FloatType, true>(df);
                    else
                        _fit<FloatType, false>(df);
                    break;
                }
                default:
                    throw std::invalid_argument("Unreachable code.");
            }
        } else {
            throw py::value_error("Variable \"" + m_variable + "\" not found in dataset.");
        }

    }

    template<typename ArrowType, bool contains_null>
    void LinearGaussianCPD::_fit(DataFrame df) {
        if (m_evidence.empty()) {
            BENCHMARK_PRE(1000)
            auto v = df.to_eigen<false, ArrowType, contains_null>(m_variable);
            auto mean = v->mean();
            auto var = (v->array() - mean).matrix().squaredNorm();
            m_beta(0) = mean;
            m_variance = var / (v->rows() - 1);
            BENCHMARK_POST(1000)

            std::cout << "beta: " << std::setprecision(24) << m_beta(0) << std::endl;
            std::cout << "variance: " << std::setprecision(24) << m_variance << std::endl;
        } else if (m_evidence.size() == 1) {
            _fit_1parent<ArrowType, contains_null>(df);
        } else if (m_evidence.size() == 2) {
            _fit_2parent<ArrowType, contains_null>(df);
        } else {
           _fit_nparent<ArrowType, contains_null>(df);
        }
    }

    template<typename ArrowType, bool contains_null>
    void LinearGaussianCPD::_fit_1parent(DataFrame df) {
        BENCHMARK_PRE(1000)
        auto [y, x] = [this, &df]() {
           if constexpr(contains_null) {
               auto y_bitmap = df.combined_bitmap(m_variable);
               auto x_bitmap = df.combined_bitmap(m_evidence[0]);
               auto rows = df->num_rows();
               auto combined_bitmap = util::bit_util::combined_bitmap(y_bitmap, x_bitmap, rows);
               auto y = df.to_eigen<false, ArrowType>(m_variable, combined_bitmap);
               auto x = df.to_eigen<false, ArrowType>(m_evidence[0], combined_bitmap);
               return std::make_tuple(std::move(y), std::move(x));
           } else {
               auto y = df.to_eigen<false, ArrowType, contains_null>(m_variable);
               auto x = df.to_eigen<false, ArrowType, contains_null>(m_evidence[0]);
               return std::make_tuple(std::move(y), std::move(x));
           }
       }();

        auto rows = y->rows();

        auto my = y->mean();

        auto mx = x->mean();
        auto dy = (y->array() - my);
        auto dx = (x->array() - mx);
        auto var = dx.matrix().squaredNorm() / (rows - 1);
        auto cov = (dy * dx).sum() / (rows - 1);

        auto b = cov / var;
        auto a = my - b*mx;
        auto v = (dy - b*dx).matrix().squaredNorm() / (rows - 2);

        m_beta(0) = a;
        m_beta(1) = b;
        m_variance = v;
        BENCHMARK_POST(1000)

        std::cout << "beta: [" << std::setprecision(24) << m_beta(0) << ", " << m_beta(1) << "]" << std::endl;
        std::cout << "variance: " << std::setprecision(24) << m_variance << std::endl;
    }

    template<typename ArrowType, bool contains_null>
    void LinearGaussianCPD::_fit_2parent(DataFrame df) {
        BENCHMARK_PRE(1000)

        auto [y, x1, x2] = [this, &df]() {
           if constexpr(contains_null) {
               auto y_bitmap = df.combined_bitmap(m_variable);
               auto x_bitmap = df.combined_bitmap(m_evidence);
               auto rows = df->num_rows();
               auto combined_bitmap = util::bit_util::combined_bitmap(y_bitmap, x_bitmap, rows);
               auto y = df.to_eigen<false, ArrowType>(m_variable, combined_bitmap);
               auto x1 = df.to_eigen<false, ArrowType>(m_evidence[0], combined_bitmap);
               auto x2 = df.to_eigen<false, ArrowType>(m_evidence[1], combined_bitmap);
               return std::make_tuple(std::move(y), std::move(x1), std::move(x2));
           } else {
               auto y = df.to_eigen<false, ArrowType, contains_null>(m_variable);
               auto x1 = df.to_eigen<false, ArrowType, contains_null>(m_evidence[0]);
               auto x2 = df.to_eigen<false, ArrowType, contains_null>(m_evidence[1]);
               return std::make_tuple(std::move(y), std::move(x1), std::move(x2));
           }
       }();

        auto rows = y->rows();

        auto mean_x1 = x1->mean();
        auto dx1 = (x1->array() - mean_x1);
        auto var_x1 = dx1.matrix().squaredNorm() / (rows - 1);

        auto mean_x2 = x2->mean();
        auto dx2 = (x2->array() - mean_x2);
        auto var_x2 = dx2.matrix().squaredNorm() / (rows - 1);

        auto cov_xx = (dx1 * dx2).sum() / (rows - 1);

        auto mean_y = y->mean();
        auto dy = (y->array() - mean_y);
        auto cov_yx1 = (dy * dx1).sum() / (rows - 1);
        auto cov_yx2 = (dy * dx2).sum() / (rows - 1);

        auto den = var_x1 * var_x2 - cov_xx * cov_xx;
        auto b1 = (var_x2 * cov_yx1 - cov_xx * cov_yx2) / den;
        auto b2 = (cov_yx2 - b1 * cov_xx) / var_x2;

        m_beta(0) = mean_y - b1*mean_x1 - b2*mean_x2;
        m_beta(1) = b1;
        m_beta(2) = b2;

        m_variance = (dy - b1*dx1 - b2*dx2).matrix().squaredNorm() / (rows - 3);
        BENCHMARK_POST(1000)

        std::cout << "beta: [" << std::setprecision(24) << m_beta(0) << ", " << m_beta(1) << ", " << m_beta(2) << "]" << std::endl;
        std::cout << "variance: " << std::setprecision(24) << m_variance << std::endl;
    }

    template<typename ArrowType, bool contains_null>
    void LinearGaussianCPD::_fit_nparent(DataFrame df) {
        BENCHMARK_PRE(1000)

        auto [y, X] = [this, &df]() {    
            if constexpr(contains_null) {
                auto y_bitmap = df.combined_bitmap(m_variable);
                auto X_bitmap = df.combined_bitmap(m_evidence);
                auto rows = df->num_rows();
                auto combined_bitmap = util::bit_util::combined_bitmap(y_bitmap, X_bitmap, rows);
                auto y = df.to_eigen<false, ArrowType>(m_variable, combined_bitmap);
                auto X = df.to_eigen<true, ArrowType>(m_evidence, combined_bitmap);
                return std::make_tuple(std::move(y), std::move(X));
            } else {
                auto y = df.to_eigen<false, ArrowType, contains_null>(m_variable);
                auto X = df.to_eigen<true, ArrowType, contains_null>(m_evidence);
                return std::make_tuple(std::move(y), std::move(X));
            }
        }();

        auto rows = y->rows();

        if constexpr (std::is_same_v<typename ArrowType::c_type, double>) {
            m_beta = X->colPivHouseholderQr().solve(*y);

            auto r = *X * m_beta;
            m_variance = (*y - r).squaredNorm() / (rows - m_beta.rows());

        } else {
            auto b = X->colPivHouseholderQr().solve(*y);

            auto r = *X * b;
            m_variance = (*y - r).squaredNorm() / (rows - m_beta.rows());

            m_beta = b.template cast<double>();
        }

        BENCHMARK_POST(1000)


        std::cout << "beta: [" << std::setprecision(24) << m_beta(0) << ", " << m_beta(1) << ", " << m_beta(2) << ", " << m_beta(3) << "]" << std::endl;
        std::cout << "variance: " << std::setprecision(24) << m_variance << std::endl;
    }

}