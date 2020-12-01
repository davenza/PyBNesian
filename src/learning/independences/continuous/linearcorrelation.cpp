#include <learning/independences/continuous/linearcorrelation.hpp>
#include <boost/math/distributions/students_t.hpp>

using boost::math::students_t_distribution;
using boost::math::cdf, boost::math::complement;


namespace learning::independences::continuous {

    double cor_pvalue(double cor, int df) {
        double statistic = cor * sqrt(df) / sqrt(1 - cor * cor);
        students_t_distribution tdist(static_cast<double>(df));
        return 2 * cdf(complement(tdist, fabs(statistic)));
    }
}