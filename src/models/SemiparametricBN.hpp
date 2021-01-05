#ifndef PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP
#define PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP

#include <models/BayesianNetwork.hpp>
#include <models/ConditionalBayesianNetwork.hpp>
#include <factors/continuous/SemiparametricCPD.hpp>
#include <util/virtual_clone.hpp>

using factors::continuous::SemiparametricCPD;
using util::FactorStringTypeVector;
using util::clone_inherit;

namespace models {

    class SemiparametricBN;
    class ConditionalSemiparametricBN;

    template<>
    struct BN_traits<SemiparametricBN> {
        using CPD = SemiparametricCPD;
    };

    template<>
    struct BN_traits<ConditionalSemiparametricBN> {
        using CPD = SemiparametricCPD;
    };

    template<typename Derived, template<typename> typename BNType>
    class SemiparametricBNImpl : public BNType<Derived>, public SemiparametricBNBase {
    public:
        using CPD = typename BN_traits<Derived>::CPD;

        // /////////////////////////////////////
        // BayesianNetwork constructors
        // /////////////////////////////////////
        template<typename D = Derived, util::enable_if_template_instantation_t<BayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes, 
                             FactorStringTypeVector& node_types) : BNType<Derived>(nodes),
                                                                   m_factor_types(nodes.size()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, util::enable_if_template_instantation_t<BayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const ArcStringVector& arcs, 
                             FactorStringTypeVector& node_types) : BNType<Derived>(arcs),
                                                                   m_factor_types(this->num_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, util::enable_if_template_instantation_t<BayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes, 
                             const ArcStringVector& arcs, 
                             FactorStringTypeVector& node_types) : BNType<Derived>(nodes, arcs),
                                                                   m_factor_types(nodes.size()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }
        
        template<typename D = Derived, util::enable_if_template_instantation_t<BayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const Dag& graph,
                             FactorStringTypeVector& node_types) : BNType<Derived>(graph),
                                                                   m_factor_types(graph.num_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, util::enable_if_template_instantation_t<BayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(Dag&& graph, FactorStringTypeVector& node_types) :
                                                    BNType<Derived>(std::move(graph)),
                                                    m_factor_types(this->num_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, util::enable_if_template_instantation_t<BayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes) : 
                                                    BNType<Derived>(nodes),
                                                    m_factor_types(nodes.size()) {}

        template<typename D = Derived, util::enable_if_template_instantation_t<BayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const ArcStringVector& arcs) :
                                                    BNType<Derived>(arcs),
                                                    m_factor_types(this->num_nodes()) {}

        template<typename D = Derived, util::enable_if_template_instantation_t<BayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes, const ArcStringVector& arcs) : 
                                                    BNType<Derived>(nodes, arcs),
                                                    m_factor_types(nodes.size()) {}

        template<typename D = Derived, util::enable_if_template_instantation_t<BayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const Dag& graph) : BNType<Derived>(graph),
                                                 m_factor_types(graph.num_nodes()) {}

        template<typename D = Derived, util::enable_if_template_instantation_t<BayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(Dag&& graph) : BNType<Derived>(std::move(graph)),
                                            m_factor_types(this->num_nodes()) {}


        // /////////////////////////////////////
        // ConditionalBayesianNetwork constructors
        // /////////////////////////////////////

        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalBayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             FactorStringTypeVector& node_types) : BNType<Derived>(nodes, interface_nodes),
                                                                   m_factor_types(nodes.size()) {
            
            for(auto& p : node_types) {
                m_factor_types[this->collapsed_index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalBayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             const ArcStringVector& arcs, 
                             FactorStringTypeVector& node_types) : BNType<Derived>(nodes, interface_nodes, arcs),
                                                                   m_factor_types(nodes.size()) {
            for(auto& p : node_types) {
                m_factor_types[this->collapsed_index(p.first)] = p.second;
            }
        }
        
        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalBayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             const Dag& graph,
                             FactorStringTypeVector& node_types) : BNType<Derived>(nodes, interface_nodes, graph),
                                                                   m_factor_types(nodes.size()) {
            for(auto& p : node_types) {
                m_factor_types[this->collapsed_index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalBayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             Dag&& graph,
                             FactorStringTypeVector& node_types) :
                                                    BNType<Derived>(nodes, interface_nodes, std::move(graph)),
                                                    m_factor_types(nodes.size()) {
            for(auto& p : node_types) {
                m_factor_types[this->collapsed_index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalBayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes) : 
                                                    BNType<Derived>(nodes, interface_nodes),
                                                    m_factor_types(nodes.size()) {}

        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalBayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             const ArcStringVector& arcs) : 
                                                    BNType<Derived>(nodes, interface_nodes, arcs),
                                                    m_factor_types(nodes.size()) {}

        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalBayesianNetwork, BNType<D>, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             const Dag& graph) : BNType<Derived>(nodes, interface_nodes, graph),
                                                    m_factor_types(nodes.size()) {}

        template<typename D = Derived, util::enable_if_template_instantation_t<ConditionalBayesianNetwork, BNType<D>>>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             Dag&& graph) : BNType<Derived>(nodes, interface_nodes, std::move(graph)),
                                            m_factor_types(nodes.size()) {}

        FactorType node_type(int node_index) const override {
            return node_type(this->name(node_index));
        }

        FactorType node_type(const std::string& node) const override {
            return m_factor_types[this->collapsed_index(node)];
        }

        void set_node_type(int node_index, FactorType new_type) override {
            set_node_type(this->name(node_index), new_type);
        }

        void set_node_type(const std::string& node, FactorType new_type) override {
            m_factor_types[this->collapsed_index(node)] = new_type;
        }

        std::unordered_map<std::string, FactorType> node_types() const {
            std::unordered_map<std::string, FactorType> res;

            for (const auto& n : this->nodes()) {
                res.insert({n, this->node_type(n)});
            }

            return res; 
        }

        void force_type_whitelist(const FactorStringTypeVector& type_whitelist) {
            for (auto& nt : type_whitelist) {
                set_node_type(nt.first, nt.second);
            }
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
            bool must_construct = BNType<Derived>::must_construct_cpd(cpd);
            
            return must_construct || (cpd.node_type() != m_factor_types[this->collapsed_index(cpd.variable())]);
        }

        void compatible_cpd(const CPD& cpd) const {
            BNType<Derived>::compatible_cpd(cpd);

            int index = this->collapsed_index(cpd.variable());
            if (m_factor_types[index] != cpd.node_type()) {
                throw std::invalid_argument(
                    "CPD defined with a different node type. Expected node type: " + m_factor_types[index].ToString() +
                    ". CPD node type: " + cpd.node_type().ToString());
            }
        }

        BayesianNetworkType type() const override {
            return BayesianNetworkType::SPBN;
        }

        // std::unique_ptr<BayesianNetworkBase> clone() const override {
        //     return std::make_unique<Derived>(static_cast<const Derived&>(*this));
        // }

        py::tuple __getstate__() const {
            return BNType<Derived>::__getstate__();
        }

        static Derived __setstate__(py::tuple& t) {
            return BNType<Derived>::__setstate__(t);
        }

        static Derived __setstate__(py::tuple&& t) {
            return BNType<Derived>::__setstate__(t);
        }
    private:
        std::vector<FactorType> m_factor_types;
    };

    class SemiparametricBN : public clone_inherit<SemiparametricBN,
                                                  SemiparametricBNImpl<SemiparametricBN, BayesianNetwork>> {
    public:
        // using SemiparametricBNImpl<SemiparametricBN, BayesianNetwork>::SemiparametricBNImpl;

        using clone_inherit<SemiparametricBN,
                            SemiparametricBNImpl<SemiparametricBN, BayesianNetwork>>::clone_inherit;

        std::string ToString() const override {
            return "SemiparametricBN";
        }

        py::tuple __getstate_extra__() const;
        void __setstate_extra__(py::tuple& t);
        void __setstate_extra__(py::tuple&& t);
    };

    class ConditionalSemiparametricBN : public clone_inherit<ConditionalSemiparametricBN,
                                                             SemiparametricBNImpl<ConditionalSemiparametricBN, ConditionalBayesianNetwork>> {
    public:
        // using SemiparametricBNImpl<ConditionalSemiparametricBN, ConditionalBayesianNetwork>::SemiparametricBNImpl;

        using clone_inherit<ConditionalSemiparametricBN,
                            SemiparametricBNImpl<ConditionalSemiparametricBN, ConditionalBayesianNetwork>>::clone_inherit;

        std::string ToString() const override {
            return "ConditionalSemiparametricBN";
        }

        py::tuple __getstate_extra__() const;
        void __setstate_extra__(py::tuple& t);
        void __setstate_extra__(py::tuple&& t);
    };
}
#endif //PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP