#include <models/BayesianNetwork.hpp>
#include <learning/scores/scores.hpp>
#include <learning/parameter/mle.hpp>
#include <util/math_constants.hpp>


using models::GaussianNetwork;
using learning::parameter::MLE;

namespace learning::scores {

    // template<typename DagType>
    // double BIC<GaussianNetwork<DagType>>::score(const DataFrame& df, 
    //                                             GaussianNetwork<DagType>& model) {
    //     // double score = 0;

    //     // for (auto node : model.nodes()) {
    //     //     auto parents = model.get_parents(node);
            
    //     //     score += local_score(df, node, parents);
    //     // }

    //     // return score;
    // }

    template<>
    template<std::enable_if_t<BIC<GaussianNetwork>::is_decomposable, int> = 0>
    double BIC<GaussianNetwork>::local_score(const DataFrame& df, 
                                             const std::string& variable, 
                                             const std::vector<std::string>& evidence) {

        MLE<LinearGaussianCPD> mle;

        auto mle_params = mle.estimate(df, variable, evidence);

        auto rows = df->num_rows();
        auto loglik = (1-rows) / 2 - (rows / 2)*std::log(2*util::pi<double>) - rows * std::log(std::sqrt(mle_params.variance));

        return loglik - std::log(rows) * 0.5 * (evidence.size() + 2);
    }

    template<>
    template<std::enable_if_t<BIC<GaussianNetwork>::is_decomposable, int> = 0>
    double BIC<GaussianNetwork>::local_score(const DataFrame& df, 
                                             int variable_index, 
                                             const std::vector<int>& evidence_index) {

        MLE<LinearGaussianCPD> mle;

        auto mle_params = mle.estimate(df, variable_index, evidence_index);

        auto rows = df->num_rows();
        auto loglik = (1-rows) / 2 - (rows / 2)*std::log(2*util::pi<double>) - rows * std::log(std::sqrt(mle_params.variance));

        return loglik - std::log(rows) * 0.5 * (evidence_index.size() + 2);
    }
}