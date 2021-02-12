#include <util/validate_scores.hpp>

namespace util {

    bool bic_compatible(const BayesianNetworkBase& bn) {
        switch(bn.type()) {
            case BayesianNetworkType::Gaussian:
            case BayesianNetworkType::Discrete:
                return true;
            default:
                return false;
        }
    }

    bool cvl_compatible(const BayesianNetworkBase& bn) {
        switch(bn.type()) {
            case BayesianNetworkType::Gaussian:
            case BayesianNetworkType::Semiparametric:
            case BayesianNetworkType::Discrete:
                return true;
            default:
                return false;
        }
    }

    bool holdout_compatible(const BayesianNetworkBase& bn) {
        switch(bn.type()) {
            case BayesianNetworkType::Gaussian:
            case BayesianNetworkType::Semiparametric:
            case BayesianNetworkType::Discrete:
                return true;
            default:
                return false;
        }
    }

    bool validatedl_compatible(const BayesianNetworkBase& bn) {
        switch(bn.type()) {
            case BayesianNetworkType::Gaussian:
            case BayesianNetworkType::Semiparametric:
            case BayesianNetworkType::Discrete:
                return true;
            default:
                return false;
        }
    }

    bool compatible_score(const BayesianNetworkBase& bn, ScoreType score) {
        switch (score) {
            case ScoreType::BIC:
                return bic_compatible(bn);
            case ScoreType::CVLikelihood:
                return cvl_compatible(bn);
            case ScoreType::HoldoutLikelihood:
                return holdout_compatible(bn);
            case ScoreType::ValidatedLikelihood:
                return validatedl_compatible(bn);
            default:
                throw std::invalid_argument("Wrong score type.");
        }
    }
}
