#ifndef PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP
#define PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP

#include <models/BayesianNetwork.hpp>
// #include <models/ConditionalBayesianNetwork.hpp>
#include <factors/continuous/SemiparametricCPD.hpp>
#include <util/virtual_clone.hpp>

using factors::continuous::SemiparametricCPD;
using util::FactorStringTypeVector;
using util::clone_inherit;

namespace models {

    template<typename Derived, typename BaseImpl>
    class SemiparametricBNImpl : public BaseImpl, public SemiparametricBNBase {
    public:
        using CPD = typename BN_traits<Derived>::CPD;

        using BaseImpl::BaseImpl;
        // /////////////////////////////////////
        // BayesianNetwork constructors
        // /////////////////////////////////////
        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes, 
                             FactorStringTypeVector& node_types) : BaseImpl(nodes),
                                                                   m_factor_types(nodes.size()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const ArcStringVector& arcs, 
                             FactorStringTypeVector& node_types) : BaseImpl(arcs),
                                                                   m_factor_types(this->num_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes, 
                             const ArcStringVector& arcs, 
                             FactorStringTypeVector& node_types) : BaseImpl(nodes, arcs),
                                                                   m_factor_types(nodes.size()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }
        
        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const Dag& graph,
                             FactorStringTypeVector& node_types) : BaseImpl(graph),
                                                                   m_factor_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(Dag&& graph, FactorStringTypeVector& node_types) 
                                            : BaseImpl(std::move(graph)),
                                              m_factor_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        // /////////////////////////////////////
        // ConditionalBayesianNetwork constructors
        // /////////////////////////////////////
        template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             FactorStringTypeVector& node_types) 
                                            : BaseImpl(nodes, interface_nodes),
                                              m_factor_types(this->num_raw_nodes()) {
            
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             const ArcStringVector& arcs, 
                             FactorStringTypeVector& node_types) 
                                            : BaseImpl(nodes, interface_nodes, arcs),
                                              m_factor_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }
        
        template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const ConditionalDag& graph,
                             FactorStringTypeVector& node_types) : BaseImpl(graph),
                                                                   m_factor_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(ConditionalDag&& graph,
                             FactorStringTypeVector& node_types) :
                                                    BaseImpl(std::move(graph)),
                                                    m_factor_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                m_factor_types[this->index(p.first)] = p.second;
            }
        }

        FactorType node_type(int node_index) const override {
            return m_factor_types[this->check_index(node_index)];
        }

        FactorType node_type(const std::string& node) const override {
            return m_factor_types[this->check_index(node)];
        }

        void set_node_type(int node_index, FactorType new_type) override {
            m_factor_types[this->check_index(node_index)] = new_type;
        }

        void set_node_type(const std::string& node, FactorType new_type) override {
            m_factor_types[this->check_index(node)] = new_type;
        }

        std::unordered_map<std::string, FactorType> node_types() const override {
            std::unordered_map<std::string, FactorType> res;

            for (const auto& n : this->nodes()) {
                res.insert({n, this->node_type(n)});
            }

            return res; 
        }

        int add_node(const std::string& node) override {
            auto new_index = BaseImpl::add_node(node);
            // Update factor type size.
            if (static_cast<size_t>(new_index) >= m_factor_types.size())
                m_factor_types.resize(new_index + 1);

            m_factor_types[new_index] = FactorType::LinearGaussianCPD;
            return new_index;
        }

        // template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        // enable_if_conditional_bn_t<Derived, void> set_node(int index) override {
        //     BayesianNetworkImpl<Derived, Base>::set_node(index);
        //     m_factor_types[this->check_index(index)] = FactorType::LinearGaussianCPD;
        // }

        // template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        // enable_if_conditional_bn_t<Derived, void> set_node(const std::string& name) override {
        //     BayesianNetworkImpl<Derived, Base>::set_node(name);
        //     m_factor_types[this->check_index(name)] = FactorType::LinearGaussianCPD;            
        // }

        void force_type_whitelist(const FactorStringTypeVector& type_whitelist) {
            for (auto& nt : type_whitelist) {
                set_node_type(nt.first, nt.second);
            }
        }

        CPD create_cpd(const std::string& node) const {
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
            bool must_construct = BaseImpl::must_construct_cpd(cpd);
            
            return must_construct || (cpd.node_type() != m_factor_types[this->index(cpd.variable())]);
        }

        void compatible_cpd(const CPD& cpd) const {
            BaseImpl::compatible_cpd(cpd);

            int index = this->index(cpd.variable());
            if (m_factor_types[index] != cpd.node_type()) {
                throw std::invalid_argument(
                    "CPD defined with a different node type. Expected node type: " + m_factor_types[index].ToString() +
                    ". CPD node type: " + cpd.node_type().ToString());
            }
        }

        BayesianNetworkType type() const override {
            return BayesianNetworkType::Semiparametric;
        }

        py::tuple __getstate__() const {
            return BaseImpl::__getstate__();
        }

        static Derived __setstate__(py::tuple& t) {
            return BaseImpl::__setstate__(t);
        }

        static Derived __setstate__(py::tuple&& t) {
            return BaseImpl::__setstate__(t);
        }

        py::tuple __getstate_extra__() const;
        void __setstate_extra__(py::tuple& t);
        void __setstate_extra__(py::tuple&& t);
    private:
        std::vector<FactorType> m_factor_types;
    };

    template<template<BayesianNetworkType> typename _BNClass>
    struct BN_traits<_BNClass<Semiparametric>> {
        using CPD = SemiparametricCPD;
        using DagClass = std::conditional_t<
                            util::GenericInstantation<BayesianNetworkType>::is_template_instantation_v<
                                                                                BayesianNetwork,
                                                                                _BNClass<Semiparametric>>,
                            Dag,
                            ConditionalDag>;
        template<BayesianNetworkType Type>
        using BNClass = _BNClass<Type>;
    };

    template<>
    class BayesianNetwork<Semiparametric> : public clone_inherit<BayesianNetwork<Semiparametric>,
                                                                 SemiparametricBNImpl<BayesianNetwork<Semiparametric>,
                                                                                      BayesianNetworkImpl<
                                                                                        BayesianNetwork<Semiparametric>,
                                                                                        BayesianNetworkBase
                                                                                      >>
                                                                 > {
    public:
        inline static constexpr auto TYPE = Semiparametric;
        using clone_inherit<BayesianNetwork<Semiparametric>,
                            SemiparametricBNImpl<BayesianNetwork<Semiparametric>,
                                                 BayesianNetworkImpl<
                                                    BayesianNetwork<Semiparametric>,
                                                    BayesianNetworkBase
                                                 >>>::clone_inherit;

        std::string ToString() const override {
            return "SemiparametricBN";
        }
    };

    template<>
    class ConditionalBayesianNetwork<Semiparametric> : public clone_inherit<ConditionalBayesianNetwork<Semiparametric>,
                                                        SemiparametricBNImpl<ConditionalBayesianNetwork<Semiparametric>,
                                                                             ConditionalBayesianNetworkImpl<
                                                                                ConditionalBayesianNetwork<Semiparametric>
                                                                             >>> {
    public:
        inline static constexpr auto TYPE = Semiparametric;
        using clone_inherit<ConditionalBayesianNetwork<Semiparametric>,
                            SemiparametricBNImpl<ConditionalBayesianNetwork<Semiparametric>,
                                                 ConditionalBayesianNetworkImpl<
                                                    ConditionalBayesianNetwork<Semiparametric>
                                                >>>::clone_inherit;

        std::string ToString() const override {
            return "ConditonalSemiparametricBN";
        }
    };




}
#endif //PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP