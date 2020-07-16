#ifndef PGM_DATASET_BIC_HPP
#define PGM_DATASET_BIC_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <dataset/dataset.hpp>
#include <models/BayesianNetwork.hpp>
#include <learning/parameters/mle.hpp>
#include <util/math_constants.hpp>
#include <util/bn_traits.hpp>

using factors::continuous::LinearGaussianCPD;
using namespace dataset;
using models::GaussianNetwork;
using learning::parameters::MLE;

namespace learning::scores {


    class BIC {
    public:
        BIC(const DataFrame& df) : m_df(df) {}

        template<typename Model>
        double score(const Model& model) {
            double s = 0;
            for (auto node = 0; node < model.num_nodes(); ++node) {
                s += local_score(model, node);
            }

            return s;
        }

        template<typename Model, typename VarType, std::enable_if_t<util::is_gaussian_network_v<Model>, int> = 0>
        double local_score(const Model& model, const VarType& variable) const {
            auto parents = model.parent_indices(variable);
            return local_score(model, variable, parents.begin(), parents.end());
        }
    
        template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<util::is_gaussian_network_v<Model>, int> = 0>
        double local_score(const Model& model, 
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

        std::string ToString() const {
            return "BIC";
        }

    private:
        const DataFrame m_df;        
    };

    template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<util::is_gaussian_network_v<Model>, int>>
    double BIC::local_score(const Model&,
                            const VarType& variable, 
                            const EvidenceIter evidence_begin,
                            const EvidenceIter evidence_end) const {
        MLE<LinearGaussianCPD> mle;

        auto mle_params = mle.estimate(m_df, variable, evidence_begin, evidence_end);

        auto rows = m_df.valid_count(variable, std::make_pair(evidence_begin, evidence_end));
        auto num_evidence = std::distance(evidence_begin, evidence_end);
        auto loglik = 0.5 * (1 + num_evidence - rows) - 0.5 * rows*std::log(2*util::pi<double>) - rows * std::log(std::sqrt(mle_params.variance));

        return loglik - std::log(rows) * 0.5 * (num_evidence + 2);
    }
}

#endif //PGM_DATASET_BIC_HPP