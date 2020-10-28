#include <learning/independences/continuous/mutual_information.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <algorithm>

#include <iomanip>

namespace learning::independences {


    double mi_pair(const DataFrame& df, int k) {
        KDTree kdtree(df);
        auto knn_results = kdtree.query(df, k+1, std::numeric_limits<double>::infinity());
        
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
        auto knn_results = kdtree.query(df, k+1, std::numeric_limits<double>::infinity());
        
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
        auto knn_results = kdtree.query(df, k+1, std::numeric_limits<double>::infinity());
        
        VectorXd eps(df->num_rows());
        for (auto i = 0; i < df->num_rows(); ++i) {
            eps(i) = knn_results[i].first(k);
        }

        std::vector<size_t> indices(df->num_columns() - 2);
        std::iota(indices.begin(), indices.end(), 2);
        auto [n_xz, n_yz, n_z] = kdtree.count_conditional_subspaces(df, 0, 1, indices.begin(), indices.end(), eps);

        double res = 0;
        for (int i = 0; i < df->num_rows(); ++i) {
            res += boost::math::digamma(n_z(i)) - boost::math::digamma(n_xz(i)) - boost::math::digamma(n_yz(i));
        }
        
        res /= df->num_rows();
        res += boost::math::digamma(k);

        return res;
    }
    
    // double KMutualInformation::mi(int v1, int v2) const {
    //     auto [nv1, nv2] = [this, v1, v2]() {
    //         if (m_rebuild_tree) {
    //             auto subset_df = m_df.loc(v1, v2);

    //             KDTree kdtree(subset_df);
    //             auto knn_results = kdtree.query(subset_df, m_k+1, std::numeric_limits<float>::infinity());
                
    //             VectorXd eps(subset_df->num_rows());
    //             for (auto i = 0; i < subset_df->num_rows(); ++i) {
    //                 eps(i) = knn_results[i].first(m_k);
    //             }

    //             auto nv1 = kdtree.count_subspace_eps(subset_df, 0, eps);
    //             auto nv2 = kdtree.count_subspace_eps(subset_df, 1, eps);

    //             return std::make_pair(nv1, nv2);
    //         } else {
    //             std::vector<size_t> subset_indices = {static_cast<size_t>(v1), static_cast<size_t>(v2)};
    //             auto knn_results = m_tree.query_subset(m_df, subset_indices, m_k+1, std::numeric_limits<float>::infinity());
                
    //             VectorXd eps(m_df->num_rows());
    //             for (auto i = 0; i < m_df->num_rows(); ++i) {
    //                 eps(i) = knn_results[i].first(m_k);
    //             }
    //             auto nv1 = m_tree.count_subspace_eps(m_df, v1, eps);
    //             auto nv2 = m_tree.count_subspace_eps(m_df, v2, eps);

    //             return std::make_pair(nv1, nv2);
    //         }
    //     }();

    //     double res = 0;
    //     for (int i = 0; i < m_df->num_rows(); ++i) {
    //         res -= boost::math::digamma(nv1(i)) + boost::math::digamma(nv2(i));
    //     }
        
    //     res /= m_df->num_rows();
    //     res += boost::math::digamma(m_k) + boost::math::digamma(m_df->num_rows());

    //     return res;
    // }

    // double KMutualInformation::mi(int v1, int v2, int cond) const {
        // auto [nv1, nv2, ncond] = [this, v1, v2, cond]() {
        //     if (m_rebuild_tree) {
        //         auto subset_df = m_df.loc(v1, v2, cond);

        //         KDTree kdtree(subset_df);
        //         auto knn_results = kdtree.query(subset_df, m_k+1, std::numeric_limits<float>::infinity());
                
        //         VectorXd eps(subset_df->num_rows());
        //         for (auto i = 0; i < subset_df->num_rows(); ++i) {
        //             eps(i) = knn_results[i].first(m_k);
        //         }


        //         auto nv1 = kdtree.count_subspace_eps(subset_df, 0, eps);
        //         auto nv2 = kdtree.count_subspace_eps(subset_df, 1, eps);

        //         return std::make_pair(nv1, nv2);
        //     } else {
        //         std::vector<size_t> subset_indices = {static_cast<size_t>(v1), 
        //                                               static_cast<size_t>(v2),
        //                                               static_cast<size_t>(cond)};

        //         auto knn_results = m_tree.query_subset(m_df, m_k+1, std::numeric_limits<float>::infinity(), subset_indices);
                
        //         VectorXd eps(m_df->num_rows());
        //         for (auto i = 0; i < m_df->num_rows(); ++i) {
        //             eps(i) = knn_results[i].first(m_k);
        //         }

        //         auto nv1 = m_tree.count_subspace_eps(m_df, v1, eps);
        //         auto nv2 = m_tree.count_subspace_eps(m_df, v2, eps);

        //         return std::make_pair(nv1, nv2);
        //     }
        // }();

        // double res = 0;
        // for (int i = 0; i < m_df->num_rows(); ++i) {
        //     res += boost::math::digamma(ncond(i)) - boost::math::digamma(nv1(i)) - boost::math::digamma(nv2(i));
        // }
        
        // res /= m_df->num_rows();
        // res += boost::math::digamma(m_k);

        // return res;
    // }
}