#ifndef PYBNESIAN_UTIL_VALIDATE_SCORES_HPP
#define PYBNESIAN_UTIL_VALIDATE_SCORES_HPP

#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/scores/holdout_likelihood.hpp>
#include <util/bn_traits.hpp>

using learning::scores::ScoreType, learning::scores::BIC, learning::scores::CVLikelihood, learning::scores::HoldoutLikelihood;

namespace util {

    bool bic_compatible(const BayesianNetworkBase& bn);
    bool cvl_compatible(const BayesianNetworkBase& bn);
    bool holdout_compatible(const BayesianNetworkBase& bn);
    bool compatible_score(const BayesianNetworkBase& bn, ScoreType score);

}

#endif //PYBNESIAN_UTIL_VALIDATE_SCORES_HPP