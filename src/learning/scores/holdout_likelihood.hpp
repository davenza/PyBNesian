#ifndef PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP
#define PGM_DATASET_HOLDOUT_LIKELIHOOD_HPP

#include <dataset/dataset.hpp>
#include <dataset/holdout_adaptator.hpp>

#include <learning/operators/operators.hpp>

using dataset::HoldOut;
using learning::operators::Operator, learning::operators::ArcOperator, learning::operators::ChangeNodeType, learning::operators::OperatorType;

namespace learning::scores {

    class HoldoutLikelihood {
    public:

        HoldoutLikelihood(const DataFrame& df, double test_ratio) : m_holdout(df, test_ratio) { }
        HoldoutLikelihood(const DataFrame& df, double test_ratio, int seed) : m_holdout(df, test_ratio, seed) { }

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

        template<typename Model, typename VarType, util::enable_if_semiparametricbn_t<Model, int> = 0>
        double local_score(const Model& model, const VarType& variable) const {
            FactorType variable_type = model.node_type(variable);
            auto parents = model.parent_indices(variable);
            return local_score(variable, variable_type, parents.begin(), parents.end());
        }

        template<typename Model, typename VarType, typename EvidenceIter, util::enable_if_semiparametricbn_t<Model, int> = 0>
        double local_score(const Model& model, 
                           const VarType& variable, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const {
            FactorType variable_type = model.node_type(variable);
            return local_score(variable, variable_type, evidence_begin, evidence_end);
        }

        template<typename VarType, typename EvidenceIter>
        double local_score(const VarType& variable, 
                           FactorType variable_type, 
                           const EvidenceIter evidence_begin, 
                           const EvidenceIter evidence_end) const;

        template<typename Model>
        double delta_score(const Model& model, Operator* op, VectorXd& current_local_scores) {
            switch(op->type()) {
                case OperatorType::ADD_ARC: {
                    auto dwn_op = dynamic_cast<ArcOperator*>(op);
                    auto target_index = model.index(dwn_op->target());
                    auto parents = model.parent_indices(target_index);
                    auto source_index = model.index(dwn_op->source());
                    parents.push_back(source_index);

                    double prev = current_local_scores(target_index);
                    current_local_scores(target_index) = local_score(model, target_index, parents.begin(), parents.end());

                    return current_local_scores(target_index) - prev;
                }
                case OperatorType::REMOVE_ARC: {
                    auto dwn_op = dynamic_cast<ArcOperator*>(op);
                    auto target_index = model.index(dwn_op->target());
                    auto parents = model.parent_indices(target_index);
                    auto source_index = model.index(dwn_op->source());

                    std::iter_swap(std::find(parents.begin(), parents.end(), source_index), parents.end() - 1);

                    double prev = current_local_scores(target_index);
                    current_local_scores(target_index) = local_score(model, target_index, parents.begin(), parents.end() - 1);

                    return current_local_scores(target_index) - prev;
                }
                case OperatorType::FLIP_ARC: {
                    auto dwn_op = dynamic_cast<ArcOperator*>(op);
                    auto target_index = model.index(dwn_op->target());
                    auto target_parents = model.parent_indices(target_index);
                    auto source_index = model.index(dwn_op->source());
                    auto source_parents = model.parent_indices(source_index);

                    std::iter_swap(std::find(target_parents.begin(), target_parents.end(), source_index), target_parents.end() - 1);
                    source_parents.push_back(target_index);

                    double prev_source = current_local_scores(source_index);
                    double prev_target = current_local_scores(target_index);
                    current_local_scores(source_index) = local_score(model, source_index, source_parents.begin(), source_parents.end());
                    current_local_scores(target_index) = local_score(model, target_index, target_parents.begin(), target_parents.end() - 1);

                    return current_local_scores(source_index) +
                           current_local_scores(target_index) -
                           prev_source -
                           prev_target;
                }
                case OperatorType::CHANGE_NODE_TYPE: {
                    auto dwn_op = dynamic_cast<ChangeNodeType*>(op);
                    auto node_index = model.index(dwn_op->node());
                    auto new_node_type = dwn_op->node_type();
                    auto parents = model.parent_indices(node_index);
                    
                    double prev = current_local_scores(node_index);
                    current_local_scores(node_index) = local_score(node_index, new_node_type, parents.begin(), parents.end());
                    return current_local_scores(node_index) - prev;
                }
                default:
                    throw std::invalid_argument("Unreachable code. Wrong operator in HoldoutLikelihood::delta_score().");
            }
        }
    
        const DataFrame& training_data() const { return m_holdout.training_data(); }
        const DataFrame& test_data() const { return m_holdout.test_data(); }

        const HoldOut& holdout() { return m_holdout; }
    private:
        HoldOut m_holdout;
    };

    template<typename Model, typename VarType, typename EvidenceIter, std::enable_if_t<util::is_gaussian_network_v<Model>, int> = 0>
    double HoldoutLikelihood::local_score(const Model&,
                                          const VarType& variable, 
                                          const EvidenceIter evidence_begin, 
                                          const EvidenceIter evidence_end) const {
        LinearGaussianCPD cpd(training_data().name(variable), training_data().names(evidence_begin, evidence_end));
        cpd.fit(training_data());
        return cpd.slogpdf(test_data());
    }

    template<typename VarType, typename EvidenceIter>
    double HoldoutLikelihood::local_score(const VarType& variable, 
                                          FactorType variable_type, 
                                          const EvidenceIter evidence_begin, 
                                          const EvidenceIter evidence_end) const {
        if (variable_type == FactorType::LinearGaussianCPD) {
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