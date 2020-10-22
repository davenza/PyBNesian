#include <learning/independences/continuous/mutual_information.hpp>
#include <boost/math/special_functions/digamma.hpp>

#include <iomanip>


namespace learning::independences {

    std::tuple<VectorXi, VectorXi, VectorXi> bruteforce_eps_neighbors(const DataFrame& df, const VectorXd& eps) { 
        auto dwn_df = df.downcast_vector<arrow::FloatType>();

        VectorXi count_xz = VectorXi::Zero(df->num_rows());
        VectorXi count_yz = VectorXi::Zero(df->num_rows());
        VectorXi count_z = VectorXi::Zero(df->num_rows());
        for (size_t i = 0; i < df->num_rows(); ++i) {
            for(size_t j = 0; j < df->num_rows(); ++j) {
                
                float max_z = 0;
                for(size_t k = 2; k < dwn_df.size(); ++k) {
                    max_z = std::max(max_z, std::abs(dwn_df[k]->Value(i) - dwn_df[k]->Value(j)));
                }

                if (max_z < eps(i)) {
                    ++count_z(i);

                    if (std::abs(dwn_df[0]->Value(i) - dwn_df[0]->Value(j)) < eps(i)) ++count_xz(i);
                    if (std::abs(dwn_df[1]->Value(i) - dwn_df[1]->Value(j)) < eps(i)) ++count_yz(i);
                }
            }
        }

        return std::make_tuple(count_xz, count_yz, count_z);
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