#ifndef PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP
#define PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP

#include <dataset/dataset.hpp>
#include <dataset/holdout_adaptator.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <learning/scores/scores.hpp>
#include <learning/operators/operators.hpp>

using dataset::HoldOut;
using models::GaussianNetwork, models::SemiparametricBN;
using learning::scores::ScoreImpl;
using learning::operators::Operator, learning::operators::ArcOperator, learning::operators::ChangeNodeType, learning::operators::OperatorType;

namespace learning::scores {

    class HoldoutLikelihood : public ScoreImpl<HoldoutLikelihood,
                                               GaussianNetwork,
                                               SemiparametricBN> {
    public:

        HoldoutLikelihood(const DataFrame& df, double test_ratio) : m_holdout(df, test_ratio) { }
        HoldoutLikelihood(const DataFrame& df, double test_ratio, int seed) : m_holdout(df, test_ratio, seed) { }

        template<typename Model>
        double score(const Model& model) const {
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

        template<typename Model, typename VarType, util::enable_if_semiparametricbn_t<Model, int> = 0>
        double local_score(const Model& model, const VarType& variable) const {
            FactorType variable_type = model.node_type(variable);
            auto parents = model.parent_indices(variable);
            return local_score<>(variable_type, variable, parents.begin(), parents.end());
        }

        template<typename Model, typename VarType, typename EvidenceIter, util::enable_if_semiparametricbn_t<Model, int> = 0>
        double local_score(const Model& model, 
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const {
            FactorType variable_type = model.node_type(variable);
            return local_score<>(variable_type, variable, evidence_begin, evidence_end);
        }

        template<typename VarType, typename EvidenceIter>
        double local_score(FactorType variable_type, 
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

        double local_score(FactorType variable_type, const int variable, 
                                   const typename std::vector<int>::const_iterator evidence_begin, 
                                   const typename std::vector<int>::const_iterator evidence_end) const override {
            return local_score<>(variable_type, variable, evidence_begin, evidence_end);
        }

        double local_score(FactorType variable_type, const std::string& variable, 
                                   const typename std::vector<std::string>::const_iterator evidence_begin, 
                                   const typename std::vector<std::string>::const_iterator evidence_end) const override {
            return local_score<>(variable_type, variable, evidence_begin, evidence_end);
        }

        const DataFrame& training_data() const { return m_holdout.training_data(); }
        const DataFrame& test_data() const { return m_holdout.test_data(); }

        const HoldOut& holdout() { return m_holdout; }

        std::string ToString() const override {
            return "HoldoutLikelihood";
        }

        bool is_decomposable() const override {
            return true;
        }

        ScoreType type() const override {
            return ScoreType::BIC;
        }
    private:
        HoldOut m_holdout;
    };

    template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<util::is_gaussian_network_v<Model>, int>>
    double HoldoutLikelihood::local_score(const Model&,
                                          const VarType& variable, 
                                          const EvidenceIter evidence_begin, 
                                          const EvidenceIter evidence_end) const {
        LinearGaussianCPD cpd(training_data().name(variable), training_data().names(evidence_begin, evidence_end));
        cpd.fit(training_data());
        return cpd.slogl(test_data());
    }

    template<typename VarType, typename EvidenceIter>
    double HoldoutLikelihood::local_score(FactorType variable_type,
                                          const VarType& variable,
                                          const EvidenceIter evidence_begin, 
                                          const EvidenceIter evidence_end) const {
        if (variable_type == FactorType::LinearGaussianCPD) {
            LinearGaussianCPD cpd(training_data().name(variable), training_data().names(evidence_begin, evidence_end));
            cpd.fit(training_data());
            return cpd.slogl(test_data());
        } else {
            CKDE cpd(training_data().name(variable), training_data().names(evidence_begin, evidence_end));
            cpd.fit(training_data());
            return cpd.slogl(test_data());
        }
    }

}




#endif //PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP