#include <learning/independences/continuous/linearcorrelation.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <Eigen/Dense>

using boost::math::students_t_distribution;
using boost::math::cdf;

using Eigen::LLT, Eigen::Ref, Eigen::DiagonalMatrix;

namespace learning::independences::continuous {

    double cor_pvalue(double cor, int df) {
        double statistic = cor * sqrt(df) / sqrt(1 - cor * cor);

        students_t_distribution tdist(df);

        return 2 * cdf(tdist, statistic);
    }

    double cor_general(MatrixXd& cov) {
        LLT<Ref<MatrixXd>> llt(cov);

        auto Lmatrix = llt.matrixL();
        MatrixXd identity = MatrixXd::Identity(cov.rows(), cov.rows());

        Lmatrix.solveInPlace(identity);

    }

}