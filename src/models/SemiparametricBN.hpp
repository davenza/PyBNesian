#ifndef PGM_DATASET_SEMIPARAMETRICBN_HPP
#define PGM_DATASET_SEMIPARAMETRICBN_HPP

#include <models/BayesianNetwork.hpp>
#include <graph/dag.hpp>

using graph::AdjMatrixDag;
using models::BayesianNetwork;

namespace models {

    template<typename D = AdjMatrixDag>
    class SemiparametricBN : public BayesianNetwork<SemiparametricBN<D>> {
    public:
        using DagType = D;
        using CPD = SemiparametricCPD;
        using node_descriptor = typename BayesianNetwork<SemiparametricBN<D>>::node_descriptor;

        SemiparametricBN(const std::vector<std::string>& nodes, std::vector<NodeType> node_types) : 
                                                                                BayesianNetwork<DagType>(nodes),
                                                                                m_node_types(node_types) {}
        SemiparametricBN(const ArcVector& arcs, std::vector<NodeType> node_types) : 
                                                                            BayesianNetwork<DagType>(arcs),
                                                                            m_node_types(node_types) {}
        SemiparametricBN(const std::vector<std::string>& nodes, const ArcVector& arcs, 
                            std::vector<NodeType> node_types) : BayesianNetwork<DagType>(nodes, arcs),
                                                                m_node_types(node_types) {}

        SemiparametricBN(const std::vector<std::string>& nodes) : 
                                                    BayesianNetwork<DagType>(nodes),
                                                    m_node_types(nodes.size()) {}
        SemiparametricBN(const ArcVector& arcs) : 
                                                    BayesianNetwork<DagType>(arcs),
                                                    m_node_types(this->num_nodes()) {}
        SemiparametricBN(const std::vector<std::string>& nodes, const ArcVector& arcs) : 
                                                    BayesianNetwork<DagType>(nodes, arcs),
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
            set_node_type(index(node), new_type);
        }

        void set_node_type(int node_index, NodeType new_type) {
            m_node_types[node_index] = new_type;
        }

        void set_node_type(const std::string& node, NodeType new_type) {
            set_node_type(this->index(node), new_type);
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