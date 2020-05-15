#ifndef PGM_DATASET_SCORES_HPP
#define PGM_DATASET_SCORES_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>

using factors::continuous::LinearGaussianCPD;
using namespace dataset;

namespace learning::scores {

    // template<typename Score>
    // struct score_traits;


    template<typename Model>
    class BIC {
    public:
        inline static constexpr bool is_decomposable = true;

        static double score(const DataFrame& df, Model& model);

        template<std::enable_if_t<is_decomposable, int> = 0>
        static double local_score(const DataFrame& df, const std::string& variable, const std::vector<std::string>& evidence);
    };





}

#endif //PGM_DATASET_SCORES_HPP