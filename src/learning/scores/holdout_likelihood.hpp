#ifndef PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP
#define PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP

#include <dataset/dataset.hpp>
#include <dataset/holdout_adaptator.hpp>

using dataset::HoldOut;

namespace learning::scores {

    class HoldoutLikelihood {
    public:

        HoldoutLikelihood(const DataFrame& df, double test_ratio, int seed = std::random_device{}()) : m_holdout(df, test_ratio, seed) { }

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
            auto parents = model.get_parent_indices(variable);
            return local_score(variable, parents.begin(), parents.end());
        }
        
        template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<util::is_gaussian_network_v<Model>, int> = 0>
        double local_score(const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

        template<typename Model, typename VarType, util::enable_if_semiparametricbn_t<Model, int> = 0>
        double local_score(const Model& model, const VarType& variable) const {
            NodeType variable_type = model.node_type(variable);
            auto parents = model.get_parent_indices(variable);
            return local_score(variable, variable_type, parents.begin(), parents.end());
        }

        template<typename VarType, typename EvidenceIter>
        double local_score(const VarType& variable, 
                           NodeType variable_type, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

        const DataFrame& training_data() const { return m_holdout.training_data(); }
        const DataFrame& test_data() const { return m_holdout.test_data(); }
    private:
        HoldOut m_holdout;
    };

    template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<util::is_gaussian_network_v<Model>, int> = 0>
    double HoldoutLikelihood::local_score(const VarType& variable, 
                       const EvidenceIter evidence_begin, 
                       const EvidenceIter evidence_end) const {
        LinearGaussianCPD cpd(training_data().name(variable), training_data().names(evidence_begin, evidence_end));
        cpd.fit(training_data());
        return cpd.slogpdf(test_data());
    }

    template<typename VarType, typename EvidenceIter>
    double HoldoutLikelihood::local_score(const VarType& variable, 
                                          NodeType variable_type, 
                                          const EvidenceIter evidence_begin, 
                                          const EvidenceIter evidence_end) const {
        if (variable_type == NodeType::LinearGaussianCPD) {
            LinearGaussianCPD cpd(training_data().name(variable), training_data().names(evidence_begin, evidence_end));
            cpd.fit(training_data());
            return cpd.slogpdf(test_data());
        } else {
            CKDE cpd(training_data().name(variable), training_data().names(evidence_begin, evidence_end));
            cpd.fit(training_data());
            return cpd.slogpdf(test_data());
        }
    }

}




#endif //PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP