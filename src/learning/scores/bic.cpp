#include <learning/scores/scores.hpp>
// #include <learning/parameter/mle_LinearGaussianCPD.hpp>
#include <learning/parameter/mle.hpp>
#include <util/math_constants.hpp>


using learning::parameter::MLE;

namespace learning::scores {

    template<>
    double BIC<LinearGaussianCPD>::local_score(DataFrame& df, const std::string& variable, const std::vector<std::string>& evidence) {

        MLE<LinearGaussianCPD> mle;

        auto mle_params = mle.estimate(df, variable, evidence);

        auto rows = df->num_rows();
        auto loglik = (1-rows) / 2 - (rows / 2)*std::log(2*util::pi<double>) - rows * std::log(std::sqrt(mle_params.variance));

        return loglik - std::log(rows) * 0.5 * (evidence.size() + 2);
    }
}