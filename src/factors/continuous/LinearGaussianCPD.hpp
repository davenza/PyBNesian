#ifndef PGM_DATASET_LINEARGAUSSIANCPD_HPP
#define PGM_DATASET_LINEARGAUSSIANCPD_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <dataset/dataset.hpp>

using Eigen::VectorXd, Eigen::Matrix, Eigen::Dynamic;

using dataset::DataFrame;

namespace py = pybind11;

namespace factors::continuous {



    struct LinearGaussianCPD_Params {
        VectorXd beta;
        double variance;
    };

    class LinearGaussianCPD {
    public:
        using ParamsClass = LinearGaussianCPD_Params;
        
        LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence);
        LinearGaussianCPD(const std::string variable, const std::vector<std::string> evidence,
                            const std::vector<double> beta, const double variance);
  
        void fit(py::handle pyobject);
        void fit(const DataFrame& df);

        VectorXd logpdf(py::handle pyobject) const;
        VectorXd logpdf(const DataFrame& df) const;

        double slogpdf(py::handle pyobject) const;
        double slogpdf(const DataFrame& df) const;

    private:
        std::string m_variable;
        std::vector<std::string> m_evidence;
        VectorXd m_beta;
        double m_variance;
    };

// namespace LinearGaussianCPD_impl {

//     template<typename ArrowType, typename VarType, typename EvidenceIter>
//     VectorXd logpdf_impl_null(const DataFrame& df, const VectorXd& beta, double variance, const VarType& var, EvidenceIter begin, EvidenceIter end) {
//         // using VectorType = Matrix<typename ArrowType::c_type, Dynamic, 1>;
        
//         // VectorType means = VectorType::Constant(beta[0]);

//         // return means;
//     }


//     template<typename ArrowType, typename VarType, typename EvidenceIter>
//     VectorXd logpdf_impl(const DataFrame& df, const VectorXd& beta, double variance, const VarType& var, EvidenceIter begin, EvidenceIter end) {
//         // using VectorType = Matrix<typename ArrowType::c_type, Dynamic, 1>;
        
//         // VectorType means = VectorXd::Constant(beta[0]);

//         // int idx = 1;
//         // for (auto it = begin; it != end; ++it, ++idx) {
//         //     auto col_m = df.to_eigen<false>(*it);
//         //     means += static_cast<typename ArrowType::c_type>(beta[idx]) * col_m;
//         // }

//         // return means;
//     }

//     template<typename VarType, typename EvidenceIter>
//     VectorXd logpdf(const DataFrame& df, const VectorXd& beta, double variance, const VarType& var, EvidenceIter begin, EvidenceIter end) {

//         std::pair<EvidenceIter, EvidenceIter> itpair = std::make_pair(begin, end);

//         switch(df.col(var)->type_id()) {
//             case Type::DOUBLE: {
//                 if(df.null_count(var, itpair) == 0)
//                     return logpdf_impl<arrow::DoubleType, VarType, EvidenceIter>(df, beta, variance, var, begin, end);
//                 else
//                     return logpdf_impl_null<arrow::DoubleType, VarType, EvidenceIter>(df, beta, variance, var, begin, end);
//             }
//             case Type::FLOAT: {
//                 if(df.null_count(var, itpair) == 0)
//                     return logpdf_impl<arrow::FloatType, VarType, EvidenceIter>(df, beta, variance, var, begin, end);
//                 else
//                     return logpdf_impl_null<arrow::FloatType, VarType, EvidenceIter>(df, beta, variance, var, begin, end);
//             }

//         }
//     }

//     // template<typename VarType, typename EvidenceIter>
//     // double slogpdf(const DataFrame& df, const VectorXd& beta, double variance, const VarType& var, EvidenceIter begin, EvidenceIter end) {
//     //     return logpdf(df, beta, variance, var, begin, end).sum();
//     // }
// }


}

#endif //PGM_DATASET_LINEARGAUSSIANCPD_HPP
