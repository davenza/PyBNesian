#ifndef PYBNESIAN_UTIL_VALIDATE_SCORES_HPP
#define PYBNESIAN_UTIL_VALIDATE_SCORES_HPP

#include <learning/scores/bic.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <util/bn_traits.hpp>

using learning::scores::ScoreType, learning::scores::BIC, learning::scores::CVLikelihood;

namespace util {

    template<typename Model>
    bool compatible_score(ScoreType score) {
        switch (score) {
            case ScoreType::BIC:
                return util::is_compatible_score_v<Model, BIC>;
            case ScoreType::PREDICTIVE_LIKELIHOOD:
                return util::is_compatible_score_v<Model, CVLikelihood>;
            default:
                throw std::invalid_argument("Wrong score type.");
        }
    };

}

#endif //PYBNESIAN_UTIL_VALIDATE_SCORES_HPP