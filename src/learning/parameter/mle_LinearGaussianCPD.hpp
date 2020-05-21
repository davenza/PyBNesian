#ifndef PGM_DATASET_MLE_LINEARGAUSSIAN_CPD_HPP
#define PGM_DATASET_MLE_LINEARGAUSSIAN_CPD_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <learning/parameter/mle.hpp>

using factors::continuous::LinearGaussianCPD;

namespace learning::parameter {

    template<typename ArrowType, bool contains_null, typename VarType, typename EvidenceIter>
    typename LinearGaussianCPD::ParamsClass _fit_1parent(const DataFrame& df,
                                                         const VarType& variable,  
                                                         EvidenceIter evidence_begin) {
        
        auto [y, x] = [&df, &variable, &evidence_begin]() {
           if constexpr(contains_null) {
               auto y_bitmap = df.combined_bitmap(variable);
               auto x_bitmap = df.combined_bitmap(*evidence_begin);
               auto rows = df->num_rows();
               auto combined_bitmap = util::bit_util::combined_bitmap(y_bitmap, x_bitmap, rows);
               auto y = df.to_eigen<false, ArrowType>(variable, combined_bitmap);
               auto x = df.to_eigen<false, ArrowType>(*evidence_begin, combined_bitmap);
               return std::make_tuple(std::move(y), std::move(x));
           } else {
               auto y = df.to_eigen<false, ArrowType, contains_null>(variable);
               auto x = df.to_eigen<false, ArrowType, contains_null>(*evidence_begin);
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

        auto beta = VectorXd(2);
        beta << a, b;

        return typename LinearGaussianCPD::ParamsClass {
            .beta = beta,
            .variance = v
        };
    }

    template<typename ArrowType, bool contains_null, typename VarType, typename EvidenceIter>
    typename LinearGaussianCPD::ParamsClass _fit_2parent(const DataFrame& df,
                                                         const VarType& variable,  
                                                         EvidenceIter evidence_begin,
                                                         EvidenceIter evidence_end) {
        
        auto [y, x1, x2] = [&df, &variable, &evidence_begin, &evidence_end]() {
           if constexpr(contains_null) {
               auto y_bitmap = df.combined_bitmap(variable);
               auto x_bitmap = df.combined_bitmap(evidence_begin, evidence_end);
               auto rows = df->num_rows();
               auto combined_bitmap = util::bit_util::combined_bitmap(y_bitmap, x_bitmap, rows);
               auto y = df.to_eigen<false, ArrowType>(variable, combined_bitmap);
               auto x1 = df.to_eigen<false, ArrowType>(*evidence_begin, combined_bitmap);
               auto x2 = df.to_eigen<false, ArrowType>(*(evidence_begin + 1), combined_bitmap);
               return std::make_tuple(std::move(y), std::move(x1), std::move(x2));
           } else {
               auto y = df.to_eigen<false, ArrowType, contains_null>(variable);
               auto x1 = df.to_eigen<false, ArrowType, contains_null>(*evidence_begin);
               auto x2 = df.to_eigen<false, ArrowType, contains_null>(*(evidence_begin + 1));
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

        auto a = mean_y - b1*mean_x1 - b2*mean_x2;
        auto v = (dy - b1*dx1 - b2*dx2).matrix().squaredNorm() / (rows - 3);

        auto b = VectorXd(3);
        b << a, b1, b2;

        return typename LinearGaussianCPD::ParamsClass {
            .beta = b,
            .variance = v
        };
    }

    template<typename ArrowType, bool contains_null, typename VarType, typename EvidenceIter>
    typename LinearGaussianCPD::ParamsClass _fit_nparent(const DataFrame& df,
                                                         const VarType& variable,  
                                                         EvidenceIter evidence_begin,
                                                         EvidenceIter evidence_end) {
        
        auto [y, X] = [&df, &variable, &evidence_begin, &evidence_end]() {    
            if constexpr(contains_null) {
                auto y_bitmap = df.combined_bitmap(variable);
                auto X_bitmap = df.combined_bitmap(evidence_begin, evidence_end);
                auto rows = df->num_rows();
                auto combined_bitmap = util::bit_util::combined_bitmap(y_bitmap, X_bitmap, rows);
                auto y = df.to_eigen<false, ArrowType>(variable, combined_bitmap);
                auto X = df.to_eigen<true, ArrowType>(evidence_begin, evidence_end, combined_bitmap);
                return std::make_tuple(std::move(y), std::move(X));
            } else {
                auto y = df.to_eigen<false, ArrowType, contains_null>(variable);
                auto X = df.to_eigen<true, ArrowType, contains_null>(evidence_begin, evidence_end);
                return std::make_tuple(std::move(y), std::move(X));
            }
        }();

        auto rows = y->rows();

        const auto b = X->colPivHouseholderQr().solve(*y).eval();
        auto r = (*X) * b;
        auto v = ((*y) - r).squaredNorm() / (rows - b.rows());

        if constexpr (std::is_same_v<typename ArrowType::c_type, double>) {
            return typename LinearGaussianCPD::ParamsClass {
                .beta = b,
                .variance = v
            };
        } else {
            return typename LinearGaussianCPD::ParamsClass {
                .beta = b.template cast<double>(),
                .variance = v
            };
        }
    }

    template<typename ArrowType, bool contains_null, typename VarType, typename EvidenceIter>
    typename LinearGaussianCPD::ParamsClass _fit(const DataFrame& df,
                                                 const VarType& variable,  
                                                 EvidenceIter evidence_begin,
                                                 EvidenceIter evidence_end) {
        auto evidence_size = std::distance(evidence_begin, evidence_end);
        if (evidence_size == 0) {
            auto v = df.to_eigen<false, ArrowType, contains_null>(variable);
            auto mean = v->mean();
            auto var = (v->array() - mean).matrix().squaredNorm();

            auto b = VectorXd(1);
            b(0) = mean;
            return typename LinearGaussianCPD::ParamsClass {
                .beta = b,
                .variance = var / (v->rows() - 1)
            };
        } else if (evidence_size == 1) {
            return _fit_1parent<ArrowType, contains_null>(df, variable, evidence_begin);
        } else if (evidence_size == 2) {
            return _fit_2parent<ArrowType, contains_null>(df, variable, evidence_begin, evidence_end);
        } else {
            return _fit_nparent<ArrowType, contains_null>(df, variable, evidence_begin, evidence_end);
        }
    }

    template<>
    template<typename VarType, typename EvidenceIter>
    typename LinearGaussianCPD::ParamsClass MLE<LinearGaussianCPD>::estimate(const DataFrame& df, 
                                                                             const VarType& variable,  
                                                                             EvidenceIter evidence_begin,
                                                                             EvidenceIter evidence_end) {

        auto var_col = df.loc(variable);

        if (var_col) {
            auto type = var_col->type();
            auto type_id = type->id();
            bool contains_null = (var_col->null_count() != 0);

            if (type_id != Type::DOUBLE && type_id != Type::FLOAT) {
                throw py::value_error("Wrong data type to fit the linear regression. (double) or (float) data is expected.");
            }

            for (auto ev = evidence_begin; ev != evidence_end; ++ev) {
                auto ev_col = df.loc(*ev);

                if constexpr (std::is_convertible_v<VarType, const std::string&> && 
                              std::is_convertible_v<typename std::iterator_traits<EvidenceIter>::value_type, const std::string&>) 
                {
                    if (!ev_col)
                        throw py::value_error("Variable \"" + static_cast<const std::string&>(*ev) + "\" not found in dataset.");
                    else if (!type->Equals(ev_col->type()))
                        throw py::value_error("\"" + variable + "\" has different data type than \"" + static_cast<const std::string&>(*ev) + "\"");
                } 
                else if constexpr (std::is_integral_v<VarType> && 
                                     std::is_integral_v<typename std::iterator_traits<EvidenceIter>::value_type>) 
                {
                    if (!ev_col)
                        throw py::value_error("Variable index [" + std::to_string(*ev) + "] not found in dataset.");
                    else if (!type->Equals(ev_col->type()))
                        throw py::value_error("Variable index [" + std::to_string(variable) + "] has different data type than "
                                              "variable index [" + std::to_string(*ev) + "]");
                } else {
                    static_assert(!(std::is_convertible_v<VarType, const std::string&> && 
                                    std::is_convertible_v<typename std::iterator_traits<EvidenceIter>::value_type, const std::string&>) ||
                                  !(std::is_integral_v<VarType> && std::is_integral_v<typename std::iterator_traits<EvidenceIter>::value_type>), 
                                    "Wrong arguments provided to MLE.estimate().");
                }

                contains_null = (contains_null || (ev_col->null_count() != 0));
            }

            switch(type_id) {
                case Type::DOUBLE: {
                    if (contains_null)
                        return _fit<DoubleType, true>(df, variable, evidence_begin, evidence_end);
                    else
                        return _fit<DoubleType, false>(df, variable, evidence_begin, evidence_end);
                    break;
                }
                case Type::FLOAT: {
                    if (contains_null)
                        return _fit<FloatType, true>(df, variable, evidence_begin, evidence_end);
                    else
                        return _fit<FloatType, false>(df, variable, evidence_begin, evidence_end);
                    break;
                }
                default:
                    throw std::invalid_argument("Unreachable code.");
            }
        } else {/*  */
            if constexpr (std::is_convertible_v<VarType, const std::string&>)
                throw py::value_error("Variable \"" + variable + "\" not found in dataset.");
            else if constexpr (std::is_integral_v<VarType>)
                throw py::value_error("Variable index [" + std::to_string(variable) + "] not found in dataset.");
            else
                static_assert(!(std::is_convertible_v<VarType, const std::string&> && 
                                std::is_convertible_v<typename std::iterator_traits<EvidenceIter>::value_type, const std::string&>) ||
                                !(std::is_integral_v<VarType> && std::is_integral_v<typename std::iterator_traits<EvidenceIter>::value_type>), 
                                "Wrong arguments provided to MLE.estimate().");
        }
    }

}

#endif //PGM_DATASET_MLE_LINEARGAUSSIAN_CPD_HPP