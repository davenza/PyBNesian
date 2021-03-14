#include <learning/independences/continuous/mutual_information.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <algorithm>

#include <iomanip>

namespace learning::independences::continuous {

double mi_pair(const DataFrame& df, int k) {
    KDTree kdtree(df);
    auto knn_results = kdtree.query(df, k + 1, std::numeric_limits<double>::infinity());

    VectorXd eps(df->num_rows());
    for (auto i = 0; i < df->num_rows(); ++i) {
        eps(i) = knn_results[i].first(k);
    }

    VectorXi nv1(df->num_rows());
    VectorXi nv2(df->num_rows());

    auto raw_values1 = df.data<arrow::FloatType>(0);
    auto raw_values2 = df.data<arrow::FloatType>(1);
    for (int i = 0, rows = static_cast<int>(df->num_rows()); i < rows; ++i) {
        auto eps_i = static_cast<int>(eps(i));

        auto v1 = static_cast<int>(raw_values1[i]);
        auto v2 = static_cast<int>(raw_values2[i]);

        nv1(i) = std::min(1 + v1, eps_i) + std::min(rows - v1, eps_i) - 1;
        nv2(i) = std::min(1 + v2, eps_i) + std::min(rows - v2, eps_i) - 1;
    }

    double res = 0;
    for (int i = 0; i < df->num_rows(); ++i) {
        res -= boost::math::digamma(nv1(i)) + boost::math::digamma(nv2(i));
    }

    res /= df->num_rows();
    res += boost::math::digamma(k) + boost::math::digamma(df->num_rows());

    return res;
}

double mi_triple(const DataFrame& df, int k) {
    KDTree kdtree(df);
    auto knn_results = kdtree.query(df, k + 1, std::numeric_limits<double>::infinity());

    VectorXd eps(df->num_rows());
    for (auto i = 0; i < df->num_rows(); ++i) {
        eps(i) = knn_results[i].first(k);
    }

    VectorXi n_xz = VectorXi::Zero(df->num_rows());
    VectorXi n_yz = VectorXi::Zero(df->num_rows());
    VectorXi n_z(df->num_rows());

    auto raw_x = df.data<arrow::FloatType>(0);
    auto raw_y = df.data<arrow::FloatType>(1);
    auto raw_z = df.data<arrow::FloatType>(2);

    IndexComparator comp_z(raw_z);
    std::vector<size_t> sort_z(df->num_rows());
    std::iota(sort_z.begin(), sort_z.end(), 0);
    std::sort(sort_z.begin(), sort_z.end(), comp_z);

    for (int i = 0, rows = static_cast<int>(df->num_rows()); i < rows; ++i) {
        auto eps_i = static_cast<int>(eps(i));
        auto x_i = static_cast<int>(raw_x[i]);
        auto y_i = static_cast<int>(raw_y[i]);
        auto z_i = static_cast<int>(raw_z[i]);

        n_z(i) = std::min(1 + z_i, eps_i) + std::min(rows - z_i, eps_i) - 1;

        if (z_i < eps_i) {
            for (int j = 0, end = z_i + eps_i; j < end; ++j) {
                auto index = sort_z[j];
                auto x_value = raw_x[index];
                auto y_value = raw_y[index];
                if (std::abs(x_i - x_value) < eps_i) ++n_xz(i);
                if (std::abs(y_i - y_value) < eps_i) ++n_yz(i);
            }
        } else if (z_i > (rows - eps_i)) {
            for (int j = z_i - eps_i + 1, end = df->num_rows(); j < end; ++j) {
                auto index = sort_z[j];
                auto x_value = raw_x[index];
                auto y_value = raw_y[index];
                if (std::abs(x_i - x_value) < eps_i) ++n_xz(i);
                if (std::abs(y_i - y_value) < eps_i) ++n_yz(i);
            }
        } else {
            for (int j = z_i - eps_i + 1, end = z_i + eps_i; j < end; ++j) {
                auto index = sort_z[j];
                auto x_value = raw_x[index];
                auto y_value = raw_y[index];
                if (std::abs(x_i - x_value) < eps_i) ++n_xz(i);
                if (std::abs(y_i - y_value) < eps_i) ++n_yz(i);
            }
        }
    }

    double res = 0;
    for (int i = 0; i < df->num_rows(); ++i) {
        res += boost::math::digamma(n_z(i)) - boost::math::digamma(n_xz(i)) - boost::math::digamma(n_yz(i));
    }

    res /= df->num_rows();
    res += boost::math::digamma(k);

    return res;
}

double mi_general(const DataFrame& df, int k) {
    KDTree kdtree(df);
    auto knn_results = kdtree.query(df, k + 1, std::numeric_limits<double>::infinity());

    VectorXd eps(df->num_rows());
    for (auto i = 0; i < df->num_rows(); ++i) {
        eps(i) = knn_results[i].first(k);
    }

    std::vector<size_t> indices(df->num_columns() - 2);
    std::iota(indices.begin(), indices.end(), 2);
    auto z_df = df.loc(indices);
    KDTree ztree(z_df);
    auto [n_xz, n_yz, n_z] = ztree.count_ball_subspaces(z_df, df.col(0), df.col(1), eps);

    double res = 0;
    for (int i = 0; i < df->num_rows(); ++i) {
        res += boost::math::digamma(n_z(i)) - boost::math::digamma(n_xz(i)) - boost::math::digamma(n_yz(i));
    }

    res /= df->num_rows();
    res += boost::math::digamma(k);

    return res;
}

double KMutualInformation::mi(const std::string& x, const std::string& y) const {
    auto subset_df = m_ranked_df.loc(x, y);
    return mi_pair(subset_df, m_k);
}

double KMutualInformation::mi(const std::string& x, const std::string& y, const std::string& z) const {
    auto subset_df = m_ranked_df.loc(x, y, z);
    return mi_triple(subset_df, m_k);
}

double KMutualInformation::mi(const std::string& x, const std::string& y, const std::vector<std::string>& z) const {
    auto subset_df = m_ranked_df.loc(x, y, z);
    return mi_general(subset_df, m_k);
}

double KMutualInformation::pvalue(const std::string& x, const std::string& y) const {
    auto value = mi(x, y);

    auto shuffled_df = m_ranked_df.loc(Copy(x), y);

    auto x_begin = shuffled_df.template mutable_data<arrow::FloatType>(0);
    auto x_end = x_begin + shuffled_df->num_rows();
    std::mt19937 rng{m_seed};

    int count_greater = 0;
    for (int i = 0; i < m_samples; ++i) {
        std::shuffle(x_begin, x_end, rng);
        auto shuffled_value = mi_pair(shuffled_df, m_k);

        if (shuffled_value >= value) ++count_greater;
    }

    return static_cast<double>(count_greater) / m_samples;
}

double KMutualInformation::pvalue(const std::string& x, const std::string& y, const std::string& z) const {
    auto original_mi = mi(x, y, z);
    auto z_df = m_df.loc(z);
    auto shuffled_df = m_ranked_df.loc(Copy(x), y, z);
    auto original_rank_x = m_ranked_df.template data<arrow::FloatType>(x);

    return shuffled_pvalue(original_mi, original_rank_x, z_df, shuffled_df, MITriple{});
}

double KMutualInformation::pvalue(const std::string& x, const std::string& y, const std::vector<std::string>& z) const {
    auto original_mi = mi(x, y, z);
    auto z_df = m_df.loc(z);
    auto shuffled_df = m_ranked_df.loc(Copy(x), y, z);
    auto original_rank_x = m_ranked_df.template data<arrow::FloatType>(x);

    return shuffled_pvalue(original_mi, original_rank_x, z_df, shuffled_df, MIGeneral{});
}

}  // namespace learning::independences::continuous