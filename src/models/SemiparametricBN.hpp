#ifndef PYBNESIAN_SEMIPARAMETRICBN_HPP
#define PYBNESIAN_SEMIPARAMETRICBN_HPP

#include <models/BayesianNetwork.hpp>
#include <util/util_types.hpp>

using models::BayesianNetwork, models::SemiparametricBNBase;
using util::FactorTypeVector;

namespace models {

    class SemiparametricBN;

    template<>
    struct BN_traits<SemiparametricBN> {
        using CPD = SemiparametricCPD;
    };

    // template<typename D = AdjMatrixDag>
    class SemiparametricBN : public BayesianNetwork<SemiparametricBN>, public SemiparametricBNBase {
    public:
        // using DagType = D;
        using CPD = SemiparametricCPD;
        // using node_descriptor = typename BayesianNetwork<SemiparametricBN<D>>::node_descriptor;

        SemiparametricBN(const std::vector<std::string>& nodes, 
                         FactorTypeVector& node_types) : BayesianNetwork<SemiparametricBN>(nodes),
                                                       m_factor_types(nodes.size()) {
            
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }
        SemiparametricBN(const ArcVector& arcs, 
                         FactorTypeVector& node_types) : BayesianNetwork<SemiparametricBN>(arcs),
                                                       m_factor_types(this->num_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }
        SemiparametricBN(const std::vector<std::string>& nodes, 
                         const ArcVector& arcs, 
                         FactorTypeVector& node_types) : BayesianNetwork<SemiparametricBN>(nodes, arcs),
                                                       m_factor_types(nodes.size()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }
        SemiparametricBN(const Dag& graph, FactorTypeVector& node_types) : 
                                                    BayesianNetwork<SemiparametricBN>(graph),
                                                    m_factor_types(graph.num_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }
        SemiparametricBN(Dag&& graph, FactorTypeVector& node_types) :
                                                    BayesianNetwork<SemiparametricBN>(std::move(graph)),
                                                    m_factor_types(this->num_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        SemiparametricBN(const std::vector<std::string>& nodes) : 
                                                    BayesianNetwork<SemiparametricBN>(nodes),
                                                    m_factor_types(nodes.size()) {}
        SemiparametricBN(const ArcVector& arcs) : 
                                                    BayesianNetwork<SemiparametricBN>(arcs),
                                                    m_factor_types(this->num_nodes()) {}
        SemiparametricBN(const std::vector<std::string>& nodes, const ArcVector& arcs) : 
                                                    BayesianNetwork<SemiparametricBN>(nodes, arcs),
                                                    m_factor_types(nodes.size()) {}
        SemiparametricBN(const Dag& graph) : BayesianNetwork<SemiparametricBN>(graph),
                                                    m_factor_types(this->num_nodes()) {}
        SemiparametricBN(Dag&& graph) : BayesianNetwork<SemiparametricBN>(std::move(graph)),
                                                    m_factor_types(this->num_nodes()) {}

        static void requires(const DataFrame& df) {
            requires_continuous_data(df);
        }

        FactorType node_type(int node_index) const override {
            return m_factor_types[node_index];
        }

        FactorType node_type(const std::string& node) const override {
            return node_type(this->index(node));
        }

        void set_node_type(int node_index, FactorType new_type) override {
            m_factor_types[node_index] = new_type;
        }

        void set_node_type(const std::string& node, FactorType new_type) override {
            set_node_type(this->index(node), new_type);
        }

        void force_type_whitelist(const FactorTypeVector& type_whitelist) {
            for (auto& nt : type_whitelist) {
                set_node_type(nt.first, nt.second);
            }
        }

        std::unordered_map<std::string, FactorType> node_types() const {
            std::unordered_map<std::string, FactorType> res;

            for (size_t i = 0; i < physical_num_nodes(); ++i) {
                if (is_valid(i)) {
                    res.insert({name(i), m_factor_types[i]});
                }
            }

            return res; 
        }

        CPD create_cpd(const std::string& node) {
            auto pa = this->parents(node);
            switch(node_type(node)) {
                case FactorType::LinearGaussianCPD:
                    return LinearGaussianCPD(node, pa);
                case FactorType::CKDE:
                    return CKDE(node, pa);
                default:
                    throw std::runtime_error("Unreachable code.");
            }
        }
    
        bool must_construct_cpd(const CPD& cpd) const {
            bool must_construct = BayesianNetwork<SemiparametricBN>::must_construct_cpd(cpd);
            
            return must_construct || (cpd.node_type() != m_factor_types[this->index(cpd.variable())]);
        }
        
        void compatible_cpd(const CPD& cpd) const {
            BayesianNetwork<SemiparametricBN>::compatible_cpd(cpd);

            int index = this->index(cpd.variable());
            if (m_factor_types[index] != cpd.node_type()) {
                throw std::invalid_argument(
                    "CPD defined with a different node type. Expected node type: " + m_factor_types[index].ToString() +
                    ". CPD node type: " + cpd.node_type().ToString());
            }
        }

        std::string ToString() const override {
            return "SemiparametricBN";
        }

        BayesianNetworkType type() const override {
            return BayesianNetworkType::SPBN;
        }

        py::tuple __getstate__() const {
            return BayesianNetwork<SemiparametricBN>::__getstate__();
        }

        static SemiparametricBN __setstate__(py::tuple& t) {
            return BayesianNetwork<SemiparametricBN>::__setstate__(t);
        }

        static SemiparametricBN __setstate__(py::tuple&& t) {
            return BayesianNetwork<SemiparametricBN>::__setstate__(t);
        }

        py::tuple __getstate_extra__() const;
        void __setstate_extra__(py::tuple& t);
        void __setstate_extra__(py::tuple&& t);
    private:
        std::vector<FactorType> m_factor_types;
    };
}
#endif //PYBNESIAN_GAUSSIANNETWORK_HPP