#ifndef PGM_DATASET_SCORES_HPP
#define PGM_DATASET_SCORES_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>

using factors::continuous::LinearGaussianCPD;
using namespace dataset;

namespace learning::scores {


    template<typename Derived>
    class StructureScore {

    public:
        double local_score(DataFrame& df, const std::string& variable, const std::vector<std::string>& evidence) {
            return static_cast<Derived*>(this)->local_score(df, variable, evidence);
        }
    };


    template<typename CPD>
    class BIC : StructureScore<BIC<CPD>> {

    public:
        double local_score(DataFrame& df, const std::string& variable, const std::vector<std::string>& evidence);
    };




}

#endif //PGM_DATASET_SCORES_HPP