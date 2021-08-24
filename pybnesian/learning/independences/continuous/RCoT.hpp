#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_RCOT_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_RCOT_HPP

#include <random>
#include <Eigen/Eigenvalues>
#include <learning/independences/independence.hpp>
#include <util/math_constants.hpp>
#include <util/basic_eigen_ops.hpp>
#include <util/chisquaresum.hpp>

using learning::independences::IndependenceTest;

namespace learning::independences::continuous {

template <typename MatrixType>
typename MatrixType::Scalar rf_sigma_impl(MatrixType& m) {
    using Scalar = typename MatrixType::Scalar;
    using VectorType = Matrix<Scalar, Dynamic, 1>;
    auto r = std::min(static_cast<decltype(m.rows())>(500), m.rows());
    VectorType distances(r * (r - 1) / 2);

    for (int i = r - 1, j = 0; i > 0; --i, j += i) {
        distances.segment(j, i) = (m.topRows(i).rowwise() - m.row(i)).matrix().rowwise().norm();
    }

    // Compute median
    int median_index = distances.rows() / 2;

    double median;
    if (distances.rows() % 2 == 1) {
        std::nth_element(distances.data(), distances.data() + median_index, distances.data() + distances.rows());
        median = distances[median_index];
    } else {
        std::sort(distances.data(), distances.data() + distances.rows());
        median = 0.5 * (distances[median_index - 1] + distances[median_index]);
    }

    if (median == 0) median = 1;

    return median;
}

class RCoT : public IndependenceTest {
public:
    RCoT(const DataFrame& df, int random_fourier_xy = 5, int random_fourier_z = 100)
        : m_df(df.normalize()),
          m_num_random_fourier_xy(random_fourier_xy),
          m_num_random_fourier_z(random_fourier_z),
          m_dfourier_x(),
          m_dfourier_y(),
          m_dfourier_z(),
          m_dsigma(),
          m_ffourier_x(),
          m_ffourier_y(),
          m_ffourier_z(),
          m_fsigma() {
        auto continuous_indices = df.continuous_columns();

        if (continuous_indices.size() < 2) {
            throw std::invalid_argument("DataFrame does not contain enough continuous columns.");
        }

        auto type = m_df.same_type(continuous_indices);

        switch (type->id()) {
            case Type::DOUBLE: {
                m_dfourier_x = MatrixXd(df.num_rows(), m_num_random_fourier_xy);
                m_dfourier_y = MatrixXd(df.num_rows(), m_num_random_fourier_xy);
                m_dfourier_z = MatrixXd(df.num_rows(), m_num_random_fourier_z);
                m_tmp_dcov = MatrixXd(df.num_rows(), m_num_random_fourier_xy * m_num_random_fourier_xy);
                m_dsigma = VectorXd(df->num_columns());

                for (auto c : continuous_indices) {
                    if (m_df.null_count(c) == 0) {
                        auto x_vec = m_df.to_eigen<false, arrow::DoubleType, false>(c);
                        m_dsigma(c) = rf_sigma_impl(*x_vec);
                    }
                }

                break;
            }
            case Type::FLOAT: {
                m_ffourier_x = MatrixXf(df.num_rows(), m_num_random_fourier_xy);
                m_ffourier_y = MatrixXf(df.num_rows(), m_num_random_fourier_xy);
                m_ffourier_z = MatrixXf(df.num_rows(), m_num_random_fourier_z);
                m_tmp_fcov = MatrixXf(df.num_rows(), m_num_random_fourier_xy * m_num_random_fourier_xy);
                m_fsigma = VectorXf(df->num_columns());

                for (auto c : continuous_indices) {
                    if (m_df.null_count(c) == 0) {
                        auto x_vec = m_df.to_eigen<false, arrow::FloatType, false>(c);
                        m_fsigma(c) = rf_sigma_impl(*x_vec);
                    }
                }

                break;
            }
            default:
                throw std::runtime_error("[RCoT] Unreachable code");
        }
    }

    double pvalue(const std::string& x, const std::string& y) const override;
    template <typename ArrowType>
    double pvalue(const std::string& x, const std::string& y) const;

    double pvalue(const std::string& x, const std::string& y, const std::string& z) const override;
    template <typename ArrowType>
    double pvalue(const std::string& x, const std::string& y, const std::string& z) const;

    double pvalue(const std::string& x, const std::string& y, const std::vector<std::string>& z) const override;
    template <typename ArrowType>
    double pvalue(const std::string& x, const std::string& y, const std::vector<std::string>& z) const;

    int num_variables() const override { return m_df->num_columns(); }

    std::vector<std::string> variable_names() const override { return m_df.column_names(); }

    const std::string& name(int i) const override { return m_df.name(i); }

    bool has_variables(const std::string& name) const override { return m_df.has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const override { return m_df.has_columns(cols); }

private:
    template <typename Scalar>
    Scalar rf_sigma(int index) const {
        if constexpr (std::is_same_v<Scalar, double>)
            return m_dsigma(index);
        else
            return m_fsigma(index);
    }

    template <typename Scalar>
    Matrix<Scalar, Dynamic, Dynamic>& fourier_x() const {
        if constexpr (std::is_same_v<Scalar, double>)
            return m_dfourier_x;
        else
            return m_ffourier_x;
    }

    template <typename Scalar>
    Matrix<Scalar, Dynamic, Dynamic>& fourier_y() const {
        if constexpr (std::is_same_v<Scalar, double>)
            return m_dfourier_y;
        else
            return m_ffourier_y;
    }

    template <typename Scalar>
    Matrix<Scalar, Dynamic, Dynamic>& fourier_z() const {
        if constexpr (std::is_same_v<Scalar, double>)
            return m_dfourier_z;
        else
            return m_ffourier_z;
    }

    template <typename Scalar>
    Matrix<Scalar, Dynamic, Dynamic>& tmp_cov() const {
        if constexpr (std::is_same_v<Scalar, double>)
            return m_tmp_dcov;
        else
            return m_tmp_fcov;
    }

    template <typename Mat>
    Matrix<typename Mat::Scalar, Dynamic, 1> eigenvalues_covariance(Mat& fourier_x, Mat& fourier_y) const;

    template <typename VectorType, typename FeatureType>
    double RIT_impl(
        VectorType& x, VectorType& y, FeatureType& feat_x, FeatureType& feat_y, double sigma_x, double sigma_y) const;
    template <bool contains_null, typename VectorType>
    double RIT(int x_index, int y_index, VectorType& x, VectorType& y) const;

    template <typename VectorType, typename MatType, typename FeatureType>
    double TestWithZ_impl(VectorType& x,
                          VectorType& y,
                          MatType& z,
                          FeatureType& feat_x,
                          FeatureType& feat_y,
                          FeatureType& feat_z,
                          double sigma_x,
                          double sigma_y,
                          double sigma_z) const;

    template <bool contains_null, typename VectorType>
    double RSingleZ(int x_index, int y_index, int z_index, VectorType& x, VectorType& y, VectorType& z) const;

    template <bool contains_null, typename VectorType, typename MatType>
    double RMultiZ(int x_index, int y_index, VectorType& x, VectorType& y, MatType& z) const;

    DataFrame m_df;
    int m_num_random_fourier_xy;
    int m_num_random_fourier_z;
    // Cache fourier matrices and sigmas (double or float).
    mutable MatrixXd m_dfourier_x;
    mutable MatrixXd m_dfourier_y;
    mutable MatrixXd m_dfourier_z;
    mutable MatrixXd m_tmp_dcov;
    VectorXd m_dsigma;
    mutable MatrixXf m_ffourier_x;
    mutable MatrixXf m_ffourier_y;
    mutable MatrixXf m_ffourier_z;
    mutable MatrixXf m_tmp_fcov;
    VectorXf m_fsigma;
};

template <typename InputMatrix, typename OutputMatrix>
void random_fourier_features(InputMatrix& m,
                             typename InputMatrix::Scalar sigma,
                             int num_features,
                             OutputMatrix& fourier_features) {
    static_assert(std::is_same_v<typename InputMatrix::Scalar, typename OutputMatrix::Scalar>,
                  "Input/Output matrices must have the same type");

    using Scalar = typename InputMatrix::Scalar;
    using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
    using VectorType = Matrix<Scalar, Dynamic, 1>;

    MatrixType W(m.cols(), num_features);
    VectorType b(num_features);

    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<Scalar> normal;
    for (auto j = 0; j < W.cols(); ++j) {
        for (auto i = 0; i < W.rows(); ++i) {
            W(i, j) = normal(rng);
        }
    }
    W *= (1 / sigma);

    std::uniform_real_distribution<Scalar> unif;
    for (auto i = 0; i < num_features; ++i) {
        b(i) = unif(rng);
    }
    b *= 2 * util::pi<Scalar>;

    fourier_features.noalias() = (m * W).rowwise() + b.transpose();
    fourier_features = fourier_features.array().cos().matrix();
    fourier_features = fourier_features * util::root_two<Scalar>;
}

template <typename Mat, typename TmpMat>
Matrix<typename Mat::Scalar, Dynamic, 1> eigenvalues_covariance_impl(Mat& fourier_x, Mat& fourier_y, TmpMat& tmp_mat) {
    using Scalar = typename Mat::Scalar;
    using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;

    for (int i = 0; i < fourier_x.cols(); ++i) {
        tmp_mat.block(0, i * fourier_y.cols(), tmp_mat.rows(), fourier_y.cols()) =
            fourier_y.array().colwise() * fourier_x.col(i).array();
    }
    auto cov = util::sse_mat(tmp_mat) * (1 / static_cast<Scalar>(fourier_x.rows()));
    auto eigen_solver = Eigen::SelfAdjointEigenSolver<MatrixType>(cov, Eigen::DecompositionOptions::EigenvaluesOnly);
    return eigen_solver.eigenvalues();
}

template <typename Mat>
Matrix<typename Mat::Scalar, Dynamic, 1> RCoT::eigenvalues_covariance(Mat& fourier_x, Mat& fourier_y) const {
    using Scalar = typename Mat::Scalar;
    auto& cc = tmp_cov<Scalar>();

    if (fourier_x.rows() != cc.rows()) {
        auto tmp = cc.topRows(fourier_x.rows());
        return eigenvalues_covariance_impl(fourier_x, fourier_y, tmp);
    } else {
        return eigenvalues_covariance_impl(fourier_x, fourier_y, cc);
    }
}

template <typename VectorType>
Matrix<typename VectorType::Scalar, Dynamic, 1> filter_positive_elements(const VectorType& v) {
    using Scalar = typename VectorType::Scalar;
    using NewVectorType = Matrix<Scalar, Dynamic, 1>;
    std::vector<Scalar> positive;
    for (int i = 0; i < v.rows(); ++i) {
        if (v(i) > 0) positive.push_back(v(i));
    }

    NewVectorType res(positive.size());
    for (size_t i = 0; i < positive.size(); ++i) {
        res(i) = positive[i];
    }

    return res;
}

template <typename VectorType, typename FeatureType>
double RCoT::RIT_impl(
    VectorType& x, VectorType& y, FeatureType& feat_x, FeatureType& feat_y, double sigma_x, double sigma_y) const {
    random_fourier_features(x, sigma_x, m_num_random_fourier_xy, feat_x);
    random_fourier_features(y, sigma_y, m_num_random_fourier_xy, feat_y);

    util::normalize_cols(feat_x);
    util::normalize_cols(feat_y);

    auto Cxy = util::cov(feat_x, feat_y);
    auto sta = x.rows() * Cxy.squaredNorm();
    auto eigs = eigenvalues_covariance(feat_x, feat_y);
    auto pos_eigs = filter_positive_elements(eigs);

    if (pos_eigs.rows() < 4) {
        auto pvalue = util::hbe_complement(pos_eigs, sta);
        if (pvalue < 0) return 0;
        return pvalue;
    }

    try {
        auto pvalue = util::lpb4_complement(pos_eigs, sta);
        if (pvalue < 0) return 0;
        return pvalue;
    } catch (std::exception&) {
        auto pvalue = util::hbe_complement(pos_eigs, sta);
        if (pvalue < 0) return 0;
        return pvalue;
    }
}

template <bool contains_null, typename VectorType>
double RCoT::RIT(int x_index, int y_index, VectorType& x, VectorType& y) const {
    using Scalar = typename VectorType::Scalar;

    if constexpr (contains_null) {
        Scalar sigma_x = rf_sigma_impl(x);
        Scalar sigma_y = rf_sigma_impl(y);

        auto feat_x = fourier_x<Scalar>().topRows(x.rows());
        auto feat_y = fourier_y<Scalar>().topRows(y.rows());

        return RIT_impl(x, y, feat_x, feat_y, sigma_x, sigma_y);
    } else {
        Scalar sigma_x = rf_sigma<Scalar>(x_index);
        Scalar sigma_y = rf_sigma<Scalar>(y_index);

        auto& feat_x = fourier_x<Scalar>();
        auto& feat_y = fourier_y<Scalar>();

        return RIT_impl(x, y, feat_x, feat_y, sigma_x, sigma_y);
    }
}

template <typename VectorType, typename MatType, typename FeatureType>
double RCoT::TestWithZ_impl(VectorType& x,
                            VectorType& y,
                            MatType& z,
                            FeatureType& feat_x,
                            FeatureType& feat_y,
                            FeatureType& feat_z,
                            double sigma_x,
                            double sigma_y,
                            double sigma_z) const {
    random_fourier_features(x, sigma_x, m_num_random_fourier_xy, feat_x);
    random_fourier_features(y, sigma_y, m_num_random_fourier_xy, feat_y);
    random_fourier_features(z, sigma_z, m_num_random_fourier_z, feat_z);

    util::normalize_cols(feat_x);
    util::normalize_cols(feat_y);
    util::normalize_cols(feat_z);

    auto Cxy = util::cov(feat_x, feat_y);

    auto Czz = util::cov(feat_z);
    Czz.diagonal().array() += 1e-10;

    auto i_Czz = Czz.inverse();

    auto Cxz = util::cov(feat_x, feat_z);
    auto Czy = util::cov(feat_z, feat_y);

    auto z_i_Czz = feat_z * i_Czz;
    feat_x = feat_x - z_i_Czz * Cxz.transpose();
    feat_y = feat_y - z_i_Czz * Czy;

    auto Cxy_z = Cxy - Cxz * i_Czz * Czy;

    auto sta = x.rows() * Cxy_z.squaredNorm();
    auto eigs = eigenvalues_covariance(feat_x, feat_y);
    auto pos_eigs = filter_positive_elements(eigs);

    if (m_num_random_fourier_z == 1 || pos_eigs.rows() < 4) {
        auto pvalue = util::hbe_complement(pos_eigs, sta);
        if (pvalue < 0) return 0;
        return pvalue;
    }

    try {
        auto pvalue = util::lpb4_complement(pos_eigs, sta);
        if (pvalue < 0) return 0;
        return pvalue;
    } catch (std::exception&) {
        auto pvalue = util::hbe_complement(pos_eigs, sta);
        if (pvalue < 0) return 0;
        return pvalue;
    }
}

template <bool contains_null, typename VectorType>
double RCoT::RSingleZ(int x_index, int y_index, int z_index, VectorType& x, VectorType& y, VectorType& z) const {
    using Scalar = typename VectorType::Scalar;

    if constexpr (contains_null) {
        Scalar sigma_x = rf_sigma_impl(x);
        Scalar sigma_y = rf_sigma_impl(y);
        Scalar sigma_z = rf_sigma_impl(z);

        auto feat_x = fourier_x<Scalar>().topRows(x.rows());
        auto feat_y = fourier_y<Scalar>().topRows(y.rows());
        auto feat_z = fourier_z<Scalar>().topRows(z.rows());

        return TestWithZ_impl(x, y, z, feat_x, feat_y, feat_z, sigma_x, sigma_y, sigma_z);
    } else {
        Scalar sigma_x = rf_sigma<Scalar>(x_index);
        Scalar sigma_y = rf_sigma<Scalar>(y_index);
        Scalar sigma_z = rf_sigma<Scalar>(z_index);

        auto& feat_x = fourier_x<Scalar>();
        auto& feat_y = fourier_y<Scalar>();
        auto& feat_z = fourier_z<Scalar>();

        return TestWithZ_impl(x, y, z, feat_x, feat_y, feat_z, sigma_x, sigma_y, sigma_z);
    }
}

template <bool contains_null, typename VectorType, typename MatType>
double RCoT::RMultiZ(int x_index, int y_index, VectorType& x, VectorType& y, MatType& z) const {
    using Scalar = typename VectorType::Scalar;

    if constexpr (contains_null) {
        Scalar sigma_x = rf_sigma_impl(x);
        Scalar sigma_y = rf_sigma_impl(y);
        Scalar sigma_z = rf_sigma_impl(z);

        auto feat_x = fourier_x<Scalar>().topRows(x.rows());
        auto feat_y = fourier_y<Scalar>().topRows(y.rows());
        auto feat_z = fourier_z<Scalar>().topRows(z.rows());

        return TestWithZ_impl(x, y, z, feat_x, feat_y, feat_z, sigma_x, sigma_y, sigma_z);
    } else {
        Scalar sigma_x = rf_sigma<Scalar>(x_index);
        Scalar sigma_y = rf_sigma<Scalar>(y_index);
        Scalar sigma_z = rf_sigma_impl(z);

        auto& feat_x = fourier_x<Scalar>();
        auto& feat_y = fourier_y<Scalar>();
        auto& feat_z = fourier_z<Scalar>();

        return TestWithZ_impl(x, y, z, feat_x, feat_y, feat_z, sigma_x, sigma_y, sigma_z);
    }
}

using DynamicRCoT = DynamicIndependenceTestAdaptator<RCoT>;

}  // namespace learning::independences::continuous

#endif  // PYBNESIAN_LEARNING_INDEPENDENCES_CONTINUOUS_RCOT_HPP
