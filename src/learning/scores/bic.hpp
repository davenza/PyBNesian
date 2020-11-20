#ifndef PYBNESIAN_LEARNING_SCORES_BIC_HPP
#define PYBNESIAN_LEARNING_SCORES_BIC_HPP

#include <learning/scores/scores.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <learning/parameters/mle_LinearGaussianCPD.hpp>

using learning::scores::ScoreType, learning::scores::ScoreImpl;
using factors::continuous::LinearGaussianCPD;
using namespace dataset;
using models::GaussianNetwork;
using learning::parameters::MLE;

namespace learning::scores {


    class BIC : public ScoreImpl<BIC, GaussianNetwork> {
    public:
        BIC(const DataFrame& df) : m_df(df) {}

        template<typename Model>
        double score(const Model& model) const {
            double s = 0;
            for (auto node = 0; node < model.num_nodes(); ++node) {
                s += local_score(model, node);
            }

            return s;
        }

        template<typename VarType>
        double local_score(const GaussianNetwork& model, const VarType& variable) const {
            auto parents = model.parent_indices(variable);
            return local_score(model, variable, parents.begin(), parents.end());
        }
    
        template<typename VarType, typename EvidenceIter>
        double local_score(const GaussianNetwork& model, 
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

        std::string ToString() const override {
            return "BIC";
        }

        bool is_decomposable() const override {
            return true;
        }

        ScoreType type() const override {
            return ScoreType::BIC;
        }

    private:
        const DataFrame m_df;        
    };

    template<typename VarType, typename EvidenceIter>
    double BIC::local_score(const GaussianNetwork&,
                            const VarType& variable, 
                            const EvidenceIter evidence_begin,
                            const EvidenceIter evidence_end) const {
        MLE<LinearGaussianCPD> mle;

        auto mle_params = mle.estimate(m_df, variable, evidence_begin, evidence_end);

        auto rows = m_df.valid_rows(variable, std::make_pair(evidence_begin, evidence_end));
        auto num_evidence = std::distance(evidence_begin, evidence_end);
        auto loglik = 0.5 * (1 + num_evidence - rows) - 0.5 * rows*std::log(2*util::pi<double>) - rows * std::log(std::sqrt(mle_params.variance));

        return loglik - std::log(rows) * 0.5 * (num_evidence + 2);
    }
}

#endif //PYBNESIAN_LEARNING_SCORES_BIC_HPP