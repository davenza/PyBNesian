#ifndef PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP
#define PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP

#include <models/BayesianNetwork.hpp>
#include <factors/continuous/SemiparametricCPD.hpp>
#include <util/virtual_clone.hpp>

using factors::continuous::SemiparametricCPD;
using util::FactorStringTypeVector;
using util::clone_inherit;

namespace models {

    template<typename Derived, template<typename> typename BaseImpl>
    class SemiparametricBNImpl : public BaseImpl<Derived>, public SemiparametricBNBase {
    public:
        using CPD = typename BN_traits<Derived>::CPD;

        using BaseImpl<Derived>::BaseImpl;
        using DagClass = typename BN_traits<Derived>::DagClass;

        SemiparametricBNImpl(const DagClass& graph) : BaseImpl<Derived>(graph),
                                                      m_node_types(this->num_raw_nodes()) {
            std::fill(m_node_types.begin(), m_node_types.end(), NodeType::LinearGaussianCPD);
        }

        SemiparametricBNImpl(DagClass&& graph) : BaseImpl<Derived>(std::move(graph)),
                                                 m_node_types(this->num_raw_nodes()) {
            std::fill(m_node_types.begin(), m_node_types.end(), NodeType::LinearGaussianCPD);
        }

        SemiparametricBNImpl(const DagClass& graph,
                             FactorStringTypeVector& node_types) : BaseImpl<Derived>(graph),
                                                                   m_node_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                if (!static_cast<Derived&>(*this).can_have_cpd(p.first))
                    throw std::invalid_argument("Node " + p.first + " in node_types list, not present in Bayesian network");

                if (p.second != NodeType::LinearGaussianCPD && p.second != NodeType::CKDE)
                    throw std::invalid_argument("Wrong factor type (" + p.second.ToString() + ") for variable " + p.first);
                
                m_node_types[this->index(p.first)] = p.second;
            }
        }

        SemiparametricBNImpl(DagClass&& graph, FactorStringTypeVector& node_types) 
                                            : BaseImpl<Derived>(std::move(graph)),
                                              m_node_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                if (!static_cast<Derived&>(*this).can_have_cpd(p.first))
                    throw std::invalid_argument("Node " + p.first + " in node_types list, not present in Bayesian network");
                
                if (p.second != NodeType::LinearGaussianCPD && p.second != NodeType::CKDE)
                    throw std::invalid_argument("Wrong factor type (" + p.second.ToString() + ") for variable " + p.first);

                m_node_types[this->index(p.first)] = p.second;
            }
        }

        // /////////////////////////////////////
        // BayesianNetwork constructors
        // /////////////////////////////////////
        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes) : BaseImpl<Derived>(nodes),
                                                                      m_node_types(this->num_raw_nodes()) {
            std::fill(m_node_types.begin(), m_node_types.end(), NodeType::LinearGaussianCPD);
        }

        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const ArcStringVector& arcs) : BaseImpl<Derived>(arcs),
                                                            m_node_types(this->num_raw_nodes()) {
            std::fill(m_node_types.begin(), m_node_types.end(), NodeType::LinearGaussianCPD);
        }

        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes, 
                             const ArcStringVector& arcs) : BaseImpl<Derived>(nodes, arcs),
                                                            m_node_types(this->num_raw_nodes()) {
            std::fill(m_node_types.begin(), m_node_types.end(), NodeType::LinearGaussianCPD);
        }

        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes, 
                             FactorStringTypeVector& node_types) : BaseImpl<Derived>(nodes),
                                                                   m_node_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                if (p.second != NodeType::LinearGaussianCPD && p.second != NodeType::CKDE)
                    throw std::invalid_argument("Wrong factor type (" + p.second.ToString() + ") for variable " + p.first);
                m_node_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const ArcStringVector& arcs, 
                             FactorStringTypeVector& node_types) : BaseImpl<Derived>(arcs),
                                                                   m_node_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                if (p.second != NodeType::LinearGaussianCPD && p.second != NodeType::CKDE)
                    throw std::invalid_argument("Wrong factor type (" + p.second.ToString() + ") for variable " + p.first);
                m_node_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, enable_if_unconditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes, 
                             const ArcStringVector& arcs, 
                             FactorStringTypeVector& node_types) : BaseImpl<Derived>(nodes, arcs),
                                                                   m_node_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                if (p.second != NodeType::LinearGaussianCPD && p.second != NodeType::CKDE)
                    throw std::invalid_argument("Wrong factor type (" + p.second.ToString() + ") for variable " + p.first);
                m_node_types[this->index(p.first)] = p.second;
            }
        }
        
        // /////////////////////////////////////
        // ConditionalBayesianNetwork constructors
        // /////////////////////////////////////
        template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes) 
                                            : BaseImpl<Derived>(nodes, interface_nodes),
                                              m_node_types(this->num_raw_nodes()) {
            std::fill(m_node_types.begin(), m_node_types.end(), NodeType::LinearGaussianCPD);
        }

        template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             const ArcStringVector& arcs) 
                                            : BaseImpl<Derived>(nodes, interface_nodes, arcs),
                                              m_node_types(this->num_raw_nodes()) {
            std::fill(m_node_types.begin(), m_node_types.end(), NodeType::LinearGaussianCPD);
        }

        template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             FactorStringTypeVector& node_types) 
                                            : BaseImpl<Derived>(nodes, interface_nodes),
                                              m_node_types(this->num_raw_nodes()) {
            
            for(auto& p : node_types) {
                if (p.second != NodeType::LinearGaussianCPD && p.second != NodeType::CKDE)
                    throw std::invalid_argument("Wrong factor type (" + p.second.ToString() + ") for variable " + p.first);
                m_node_types[this->index(p.first)] = p.second;
            }
        }

        template<typename D = Derived, enable_if_conditional_bn_t<D, int> = 0>
        SemiparametricBNImpl(const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             const ArcStringVector& arcs, 
                             FactorStringTypeVector& node_types) 
                                            : BaseImpl<Derived>(nodes, interface_nodes, arcs),
                                              m_node_types(this->num_raw_nodes()) {
            for(auto& p : node_types) {
                if (p.second != NodeType::LinearGaussianCPD && p.second != NodeType::CKDE)
                    throw std::invalid_argument("Wrong factor type (" + p.second.ToString() + ") for variable " + p.first);
                m_node_types[this->index(p.first)] = p.second;
            }
        }

        NodeType node_type(int node_index) const override {
            return m_node_types[this->check_index(node_index)];
        }

        NodeType node_type(const std::string& node) const override {
            return m_node_types[this->check_index(node)];
        }

        void set_node_type(int node_index, NodeType new_type) override {
            if (new_type != NodeType::LinearGaussianCPD && new_type != NodeType::CKDE)
                throw std::invalid_argument("Wrong factor type (" + new_type.ToString() + ") for variable " 
                                                + std::to_string(node_index));
            m_node_types[this->check_index(node_index)] = new_type;
        }

        void set_node_type(const std::string& node, NodeType new_type) override {
            if (new_type != NodeType::LinearGaussianCPD && new_type != NodeType::CKDE)
                throw std::invalid_argument("Wrong factor type (" + new_type.ToString() + ") for variable " + node);
            m_node_types[this->check_index(node)] = new_type;
        }

        std::unordered_map<std::string, NodeType> node_types() const override {
            std::unordered_map<std::string, NodeType> res;

            for (const auto& n : this->nodes()) {
                res.insert({n, this->node_type(n)});
            }

            return res; 
        }

        int add_node(const std::string& node) override {
            auto new_index = BaseImpl<Derived>::add_node(node);
            // Update factor type size.
            if (static_cast<size_t>(new_index) >= m_node_types.size())
                m_node_types.resize(new_index + 1);

            m_node_types[new_index] = NodeType::LinearGaussianCPD;
            return new_index;
        }

        void force_type_whitelist(const FactorStringTypeVector& type_whitelist) override {
            for (auto& nt : type_whitelist) {
                set_node_type(nt.first, nt.second);
            }
        }

        CPD create_cpd(const std::string& node) const {
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

        bool must_construct_cpd(const CPD& cpd) const {
            bool must_construct = BaseImpl<Derived>::must_construct_cpd(cpd);
            
            return must_construct || (cpd.underlying_node_type() != m_node_types[this->index(cpd.variable())]);
        }

        void compatible_cpd(const CPD& cpd) const {
            BaseImpl<Derived>::compatible_cpd(cpd);

            int index = this->index(cpd.variable());
            if (m_node_types[index] != cpd.underlying_node_type()) {
                throw std::invalid_argument(
                    "CPD defined with a different node type. Expected node type: " + m_node_types[index].ToString() +
                    ". CPD node type: " + cpd.underlying_node_type().ToString());
            }
        }

        BayesianNetworkType type() const override {
            return BayesianNetworkType::Semiparametric;
        }

        py::tuple __getstate__() const {
            return BaseImpl<Derived>::__getstate__();
        }

        static Derived __setstate__(py::tuple& t) {
            return BaseImpl<Derived>::__setstate__(t);
        }

        static Derived __setstate__(py::tuple&& t) {
            return BaseImpl<Derived>::__setstate__(t);
        }

        py::tuple __getstate_extra__() const;
        void __setstate_extra__(py::tuple& t);
        void __setstate_extra__(py::tuple&& t) {
            __setstate_extra__(t);
        }
    private:
        std::vector<NodeType> m_node_types;
    };

    template<typename Derived, template<typename> typename BaseImpl>
    py::tuple SemiparametricBNImpl<Derived, BaseImpl>::__getstate_extra__() const {
        std::vector<py::tuple> types;
        types.reserve(this->num_nodes());

        for (const auto& name : this->nodes()) {
            auto idx = this->index(name);

            types.push_back(py::make_tuple(name, node_type(idx).ToString()));
        }

        return py::cast(types);
    }

    template<typename Derived, template<typename> typename BaseImpl>
    void SemiparametricBNImpl<Derived, BaseImpl>::__setstate_extra__(py::tuple& t) {
        std::vector<py::tuple> types = t.cast<std::vector<py::tuple>>();

        for (const auto& type : types) {
            const auto& name = type[0].cast<std::string>();
            std::string type_string = type[1].cast<std::string>();

            this->set_node_type(name, NodeType::from_string(type_string));
        }
    }

    template<template<BayesianNetworkType::Value> typename _BNClass>
    struct BN_traits<_BNClass<BayesianNetworkType::Semiparametric>,
                     std::enable_if_t<(is_unconditional_bn_v<_BNClass<BayesianNetworkType::Semiparametric>> ||
                                       is_conditional_bn_v<_BNClass<BayesianNetworkType::Semiparametric>>),
                                       void>
    > {
        using CPD = SemiparametricCPD;
        using BaseClass = std::conditional_t<
                                is_unconditional_bn_v<_BNClass<BayesianNetworkType::Semiparametric>>,
                                BayesianNetworkBase,
                                ConditionalBayesianNetworkBase
                        >;
        using DagClass = std::conditional_t<is_unconditional_bn_v<_BNClass<BayesianNetworkType::Semiparametric>>,
                                Dag,
                                ConditionalDag
                        >;
        template<BayesianNetworkType::Value Type>
        using BNClass = _BNClass<Type>;
        inline static constexpr auto TYPE = BayesianNetworkType::Semiparametric;
    };

    template<>
    class BayesianNetwork<BayesianNetworkType::Semiparametric> 
        : public clone_inherit<SemiparametricBN, SemiparametricBNImpl<SemiparametricBN, BayesianNetworkImpl>> {
    public:
        inline static constexpr auto TYPE = BN_traits<SemiparametricBN>::TYPE;
        using clone_inherit::clone_inherit;

        std::string ToString() const override {
            return "SemiparametricBN";
        }
    };

    template<>
    class ConditionalBayesianNetwork<BayesianNetworkType::Semiparametric>
        : public clone_inherit<ConditionalSemiparametricBN, SemiparametricBNImpl<ConditionalSemiparametricBN,
                                                                                 ConditionalBayesianNetworkImpl>> {
    public:
        inline static constexpr auto TYPE = BN_traits<ConditionalSemiparametricBN>::TYPE;
        using clone_inherit::clone_inherit;


        void set_node(int index) override {
            clone_inherit::set_node(index);
            this->set_node_type(index, NodeType::LinearGaussianCPD);
        }

        void set_node(const std::string& name) override {
            clone_inherit::set_node(name);
            this->set_node_type(name, NodeType::LinearGaussianCPD);
        }

        std::string ToString() const override {
            return "ConditionalSemiparametricBN";
        }
    };

    template<>
    class DynamicBayesianNetwork<BayesianNetworkType::Semiparametric>
        : public clone_inherit<DynamicSemiparametricBN, DynamicBayesianNetworkImpl<DynamicSemiparametricBN>> {
    public:
        inline static constexpr auto TYPE = BN_traits<DynamicSemiparametricBN>::TYPE;
        using clone_inherit::clone_inherit;

        std::string ToString() const override {
            return "DynamicSemiparametricBN";
        }
    };
}
#endif //PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP