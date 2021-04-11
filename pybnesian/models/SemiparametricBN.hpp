#ifndef PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP
#define PYBNESIAN_MODELS_SEMIPARAMETRICBN_HPP

#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/continuous/CKDE.hpp>
#include <util/virtual_clone.hpp>

using factors::continuous::LinearGaussianCPDType, factors::continuous::CKDEType;
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

    std::shared_ptr<FactorType> default_node_type() const override { return LinearGaussianCPDType::get(); }

    bool compatible_node_type(const BayesianNetworkBase& m, const std::string& variable) const override {
        auto nt = m.node_type(variable);
        if (*nt != LinearGaussianCPDType::get_ref() && *nt != CKDEType::get_ref()) return false;

        return true;
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