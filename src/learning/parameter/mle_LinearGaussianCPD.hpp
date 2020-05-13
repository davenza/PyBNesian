#ifndef PGM_DATASET_MLE_LINEARGAUSSIAN_CPD_HPP
#define PGM_DATASET_MLE_LINEARGAUSSIAN_CPD_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <learning/parameter/mle.hpp>

using factors::continuous::LinearGaussianCPD;

namespace learning::parameter {


    template<>
    typename LinearGaussianCPD::ParamsClass 
    MLE<LinearGaussianCPD>::estimate(DataFrame& df, const std::string& variable,  const std::vector<std::string>& evidence) {
        // if (evidence.empty()) {
        //     auto v = df.to_eigen<false, ArrowType, contains_null>(variable);
        //     auto mean = v->mean();
        //     auto var = (v->array() - mean).matrix().squaredNorm();

            
        //     return LinearGaussianCPD_Params {
        //         .beta = VectorXd(mean),
        //         .variance = var / (v->rows() - 1)
        //     }

        // } else if (m_evidence.size() == 1) {
        //     return _fit_1parent<ArrowType, contains_null>(df);
        // } else if (m_evidence.size() == 2) {
        //     return _fit_2parent<ArrowType, contains_null>(df);
        // } else {
        //    return _fit_nparent<ArrowType, contains_null>(df);
        // }
    }




    template<typename ArrowType, bool contains_null>
    typename LinearGaussianCPD::ParamsClass _fit_1parent(DataFrame& df, const std::string& variable,  const std::vector<std::string>& evidence) {
        auto [y, x] = [&df, &variable, &evidence]() {
           if constexpr(contains_null) {
               auto y_bitmap = df.combined_bitmap(variable);
               auto x_bitmap = df.combined_bitmap(evidence[0]);
               auto rows = df->num_rows();
               auto combined_bitmap = util::bit_util::combined_bitmap(y_bitmap, x_bitmap, rows);
               auto y = df.to_eigen<false, ArrowType>(variable, combined_bitmap);
               auto x = df.to_eigen<false, ArrowType>(evidence[0], combined_bitmap);
               return std::make_tuple(std::move(y), std::move(x));
           } else {
               auto y = df.to_eigen<false, ArrowType, contains_null>(variable);
               auto x = df.to_eigen<false, ArrowType, contains_null>(evidence[0]);
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

        return typename LinearGaussianCPD::ParamsClass {
            .beta = VectorXd(a, b),
            .variance = v
        };
    }



}

#endif //PGM_DATASET_MLE_LINEARGAUSSIAN_CPD_HPP