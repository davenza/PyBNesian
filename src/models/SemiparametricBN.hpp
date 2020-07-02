#ifndef PGM_DATASET_SEMIPARAMETRICBN_HPP
#define PGM_DATASET_SEMIPARAMETRICBN_HPP

#include <models/BayesianNetwork.hpp>
#include <graph/dag.hpp>
#include <util/util_types.hpp>

using graph::AdjMatrixDag;
using models::BayesianNetwork;
using util::NodeTypeVector;

namespace models {

    template<typename D = AdjMatrixDag>
    class SemiparametricBN : public BayesianNetwork<SemiparametricBN<D>> {
    public:
        using DagType = D;
        using CPD = SemiparametricCPD;
        using node_descriptor = typename BayesianNetwork<SemiparametricBN<D>>::node_descriptor;

        SemiparametricBN(const std::vector<std::string>& nodes, 
                         NodeTypeVector& node_types) : BayesianNetwork<SemiparametricBN<D>>(nodes),
                                                       m_node_types(nodes.size()) {
            
            for(auto& p : node_types) {
                m_node_types[this->index(p.first)] = p.second;
            }
        }
        SemiparametricBN(const ArcVector& arcs, 
                         NodeTypeVector& node_types) : BayesianNetwork<SemiparametricBN<D>>(arcs),
                                                       m_node_types(this->num_nodes()) {
            for(auto& p : node_types) {
                m_node_types[this->index(p.first)] = p.second;
            }
        }
        SemiparametricBN(const std::vector<std::string>& nodes, 
                         const ArcVector& arcs, 
                         NodeTypeVector& node_types) : BayesianNetwork<SemiparametricBN<D>>(nodes, arcs),
                                                       m_node_types(nodes.size()) {
            for(auto& p : node_types) {
                m_node_types[this->index(p.first)] = p.second;
            }
        }

        SemiparametricBN(const std::vector<std::string>& nodes) : 
                                                    BayesianNetwork<SemiparametricBN<D>>(nodes),
                                                    m_node_types(nodes.size()) {}
        SemiparametricBN(const ArcVector& arcs) : 
                                                    BayesianNetwork<SemiparametricBN<D>>(arcs),
                                                    m_node_types(this->num_nodes()) {}
        SemiparametricBN(const std::vector<std::string>& nodes, const ArcVector& arcs) : 
                                                    BayesianNetwork<SemiparametricBN<D>>(nodes, arcs),
                                                    m_node_types(nodes.size()) {}

        static void requires(const DataFrame& df) {
            requires_continuous_data(df);
        }

        NodeType node_type(node_descriptor node) const {
            return node_type(index(node));
        }
        
        NodeType node_type(int node_index) const {
            return m_node_types[node_index];
        }

        NodeType node_type(const std::string& node) const {
            return node_type(this->index(node));
        }

        void set_node_type(node_descriptor node, NodeType new_type) {
            set_node_type(this->index(node), new_type);
        }

        void set_node_type(int node_index, NodeType new_type) {
            m_node_types[node_index] = new_type;
        }

        void set_node_type(const std::string& node, NodeType new_type) {
            set_node_type(this->index(node), new_type);
        }

        CPD create_cpd(const std::string& node) {
            auto pa = this->parents(node);
            switch(node_type(node)) {
                case NodeType::LinearGaussianCPD:
                    return LinearGaussianCPD(node, pa);
                case NodeType::CKDE:
                    return CKDE(node, pa);
                default:
                    throw std::runtime_error("Unreachable code.");
            }
        }
    
        bool must_refit_cpd(const CPD& cpd) const {
            bool must_refit = BayesianNetwork<SemiparametricBN<D>>::must_refit_cpd(cpd);
            
            return must_refit || (cpd.node_type() != m_node_types[this->index(cpd.variable())]);
        }
        
        void compatible_cpd(const CPD& cpd) const {
            BayesianNetwork<SemiparametricBN<D>>::compatible_cpd(cpd);

            int index = this->index(cpd.variable());
            if (m_node_types[index] != cpd.node_type()) {
                throw std::invalid_argument(
                    "CPD defined with a different node type. Expected node type: " + m_node_types[index].ToString() +
                    ". CPD node type: " + cpd.node_type().ToString());
            }
        }

    private:
        std::vector<NodeType> m_node_types;
    };

    template<typename D>
    struct BN_traits<SemiparametricBN<D>> {
        using DagType = D;
        using CPD = SemiparametricCPD;
    };
}
#endif //PGM_DATASET_GAUSSIANNETWORK_HPP