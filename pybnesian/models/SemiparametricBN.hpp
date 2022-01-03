#ifndef PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP
#define PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP

#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <util/virtual_clone.hpp>

using factors::continuous::LinearGaussianCPDType, factors::continuous::CKDEType, factors::discrete::DiscreteFactorType;
using util::clone_inherit;
using util::FactorTypeVector;

namespace models {

class SemiparametricBNType : public BayesianNetworkType {
public:
    SemiparametricBNType(const SemiparametricBNType&) = delete;
    void operator=(const SemiparametricBNType&) = delete;

    static std::shared_ptr<SemiparametricBNType> get() {
        static std::shared_ptr<SemiparametricBNType> singleton =
            std::shared_ptr<SemiparametricBNType>(new SemiparametricBNType);
        return singleton;
    }

    static SemiparametricBNType& get_ref() {
        static SemiparametricBNType& ref = *SemiparametricBNType::get();
        return ref;
    }

    std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const override;
    std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override;

    bool is_homogeneous() const override { return false; }

    std::shared_ptr<FactorType> default_node_type() const override {
        throw std::runtime_error("default_node_type() for SemiparametricBN is not defined.");
    }

    std::vector<std::shared_ptr<FactorType>> data_default_node_type(
        const std::shared_ptr<DataType>& dt) const override {
        switch (dt->id()) {
            case Type::DOUBLE:
            case Type::FLOAT:
                return {LinearGaussianCPDType::get(), CKDEType::get()};
            case Type::DICTIONARY:
                return {DiscreteFactorType::get()};
            default:
                throw std::invalid_argument("Data type [" + dt->ToString() +
                                            "] not compatible with SemiparametricBNType");
        }
    }

    bool compatible_node_type(const BayesianNetworkBase& m,
                              const std::string& var,
                              const std::shared_ptr<FactorType>& nt) const override {
        if (*nt != LinearGaussianCPDType::get_ref() && *nt != CKDEType::get_ref() &&
            *nt != DiscreteFactorType::get_ref())
            return false;

        if (*nt == DiscreteFactorType::get_ref()) {
            auto parents = m.parents(var);

            for (const auto& p : parents) {
                if (*m.node_type(p) != DiscreteFactorType::get_ref()) return false;
            }
        }

        return true;
    }

    bool compatible_node_type(const ConditionalBayesianNetworkBase& m,
                              const std::string& var,
                              const std::shared_ptr<FactorType>& nt) const override {
        if (*nt != LinearGaussianCPDType::get_ref() && *nt != CKDEType::get_ref() &&
            *nt != DiscreteFactorType::get_ref())
            return false;

        if (*nt == DiscreteFactorType::get_ref()) {
            auto parents = m.parents(var);

            for (const auto& p : parents) {
                if (!m.is_interface(p) && *m.node_type(p) != DiscreteFactorType::get_ref()) return false;
            }
        }

        return true;
    }

    bool can_have_arc(const BayesianNetworkBase& m,
                      const std::string& source,
                      const std::string& target) const override {
        return *m.node_type(target) != DiscreteFactorType::get_ref() ||
               *m.node_type(source) == DiscreteFactorType::get_ref();
    }

    bool can_have_arc(const ConditionalBayesianNetworkBase& m,
                      const std::string& source,
                      const std::string& target) const override {
        return can_have_arc(static_cast<const BayesianNetworkBase&>(m), source, target);
    }

    std::vector<std::shared_ptr<FactorType>> alternative_node_type(const BayesianNetworkBase& model,
                                                                   const std::string& variable) const override {
        auto v = std::vector<std::shared_ptr<FactorType>>();

        if (*model.node_type(variable) == LinearGaussianCPDType::get_ref()) {
            v.reserve(1);
            v.push_back(CKDEType::get());
        } else if (*model.node_type(variable) == CKDEType::get_ref()) {
            v.reserve(1);
            v.push_back(LinearGaussianCPDType::get());
        }

        return v;
    }

    std::vector<std::shared_ptr<FactorType>> alternative_node_type(const ConditionalBayesianNetworkBase& model,
                                                                   const std::string& variable) const override {
        return alternative_node_type(static_cast<const BayesianNetworkBase&>(model), variable);
    }

    std::string ToString() const override { return "SemiparametricNetworkType"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<SemiparametricBNType> __setstate__(py::tuple&) { return SemiparametricBNType::get(); }

    static std::shared_ptr<SemiparametricBNType> __setstate__(py::tuple&&) { return SemiparametricBNType::get(); }

private:
    SemiparametricBNType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

class SemiparametricBN : public clone_inherit<SemiparametricBN, BayesianNetwork> {
public:
    SemiparametricBN(const std::vector<std::string>& nodes) : clone_inherit(SemiparametricBNType::get(), nodes) {}

    SemiparametricBN(const std::vector<std::string>& nodes, const FactorTypeVector& node_types)
        : clone_inherit(SemiparametricBNType::get(), nodes, node_types) {}

    SemiparametricBN(const ArcStringVector& arcs) : clone_inherit(SemiparametricBNType::get(), arcs) {}

    SemiparametricBN(const ArcStringVector& arcs, const FactorTypeVector& node_types)
        : clone_inherit(SemiparametricBNType::get(), arcs, node_types) {}

    SemiparametricBN(const std::vector<std::string>& nodes, const ArcStringVector& arcs)
        : clone_inherit(SemiparametricBNType::get(), nodes, arcs) {}

    SemiparametricBN(const std::vector<std::string>& nodes,
                     const ArcStringVector& arcs,
                     const FactorTypeVector& node_types)
        : clone_inherit(SemiparametricBNType::get(), nodes, arcs, node_types) {}

    SemiparametricBN(const Dag& graph) : clone_inherit(SemiparametricBNType::get(), graph) {}

    SemiparametricBN(const Dag& graph, const FactorTypeVector& node_types)
        : clone_inherit(SemiparametricBNType::get(), graph, node_types) {}

    SemiparametricBN(Dag&& graph) : clone_inherit(SemiparametricBNType::get(), std::move(graph)) {}

    SemiparametricBN(Dag&& graph, const FactorTypeVector& node_types)
        : clone_inherit(SemiparametricBNType::get(), std::move(graph), node_types) {}

    std::string ToString() const override { return "SemiparametricBN"; }
};

class ConditionalSemiparametricBN : public clone_inherit<ConditionalSemiparametricBN, ConditionalBayesianNetwork> {
public:
    ConditionalSemiparametricBN(const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes)
        : clone_inherit(SemiparametricBNType::get(), nodes, interface_nodes) {}

    ConditionalSemiparametricBN(const std::vector<std::string>& nodes,
                                const std::vector<std::string>& interface_nodes,
                                const FactorTypeVector& node_types)
        : clone_inherit(SemiparametricBNType::get(), nodes, interface_nodes, node_types) {}

    ConditionalSemiparametricBN(const std::vector<std::string>& nodes,
                                const std::vector<std::string>& interface_nodes,
                                const ArcStringVector& arcs)
        : clone_inherit(SemiparametricBNType::get(), nodes, interface_nodes, arcs) {}

    ConditionalSemiparametricBN(const std::vector<std::string>& nodes,
                                const std::vector<std::string>& interface_nodes,
                                const ArcStringVector& arcs,
                                const FactorTypeVector& node_types)
        : clone_inherit(SemiparametricBNType::get(), nodes, interface_nodes, arcs, node_types) {}

    ConditionalSemiparametricBN(const ConditionalDag& graph) : clone_inherit(SemiparametricBNType::get(), graph) {}

    ConditionalSemiparametricBN(const ConditionalDag& graph, const FactorTypeVector& node_types)
        : clone_inherit(SemiparametricBNType::get(), graph, node_types) {}

    ConditionalSemiparametricBN(ConditionalDag&& graph)
        : clone_inherit(SemiparametricBNType::get(), std::move(graph)) {}

    ConditionalSemiparametricBN(ConditionalDag&& graph, const FactorTypeVector& node_types)
        : clone_inherit(SemiparametricBNType::get(), std::move(graph), node_types) {}

    std::string ToString() const override { return "ConditionalSemiparametricBN"; }
};

class DynamicSemiparametricBN : public clone_inherit<DynamicSemiparametricBN, DynamicBayesianNetwork> {
public:
    DynamicSemiparametricBN(const std::vector<std::string>& variables, int markovian_order)
        : clone_inherit(SemiparametricBNType::get(), variables, markovian_order) {}

    DynamicSemiparametricBN(const std::vector<std::string>& variables,
                            int markovian_order,
                            std::shared_ptr<BayesianNetworkBase> static_bn,
                            std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn)
        : clone_inherit(variables, markovian_order, static_bn, transition_bn) {
        if (static_bn->type_ref() != SemiparametricBNType::get_ref())
            throw std::invalid_argument("Bayesian networks are not semiparametric.");
    }

    std::string ToString() const override { return "DynamicSemiparametricBN"; }
};

}  // namespace models
#endif  // PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP