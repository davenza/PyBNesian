#include <util/validate_scores.hpp>

namespace util {

    bool bic_compatible(const BayesianNetworkBase& bn) {
        switch(bn.type()) {
            case BayesianNetworkType::GBN:
                return true;
            case BayesianNetworkType::DISCRETEBN:
                return true;
            default:
                return false;
        }
    }

    bool cvl_compatible(const BayesianNetworkBase& bn) {
        switch(bn.type()) {
            case BayesianNetworkType::GBN:
                return true;
            case BayesianNetworkType::SPBN:
                return true;
            case BayesianNetworkType::DISCRETEBN:
                return true;
            default:
                return false;
        }
    }

    bool holdout_compatible(const BayesianNetworkBase& bn) {
        switch(bn.type()) {
            case BayesianNetworkType::GBN:
                return true;
            case BayesianNetworkType::SPBN:
                return true;
            case BayesianNetworkType::DISCRETEBN:
                return true;
            default:
                return false;
        }
    }

    bool compatible_score(const BayesianNetworkBase& bn, ScoreType score) {
        switch (score) {
            case ScoreType::BIC:
                return bic_compatible(bn);
            case ScoreType::PREDICTIVE_LIKELIHOOD:
                return cvl_compatible(bn);
            case ScoreType::HOLDOUT_LIKELIHOOD:
                return holdout_compatible(bn);
            default:
                throw std::invalid_argument("Wrong score type.");
        }
    }
}