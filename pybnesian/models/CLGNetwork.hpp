#ifndef PYBNESIAN_MODELS_CLGNETWORK_HPP
#define PYBNESIAN_MODELS_CLGNETWORK_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>

using factors::continuous::LinearGaussianCPDType;
using factors::discrete::DiscreteFactorType;

namespace models {

class CLGNetworkType : public BayesianNetworkType {
public:
    CLGNetworkType(const CLGNetworkType&) = delete;
    void operator=(const CLGNetworkType&) = delete;

    static std::shared_ptr<CLGNetworkType> get() {
        static std::shared_ptr<CLGNetworkType> singleton = std::shared_ptr<CLGNetworkType>(new CLGNetworkType);
        return singleton;
    }

    static CLGNetworkType& get_ref() {
        static CLGNetworkType& ref = *CLGNetworkType::get();
        return ref;
    }

    std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const override;
    std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override;

    bool is_homogeneous() const override { return false; }

    std::shared_ptr<FactorType> default_node_type() const override {
        throw std::runtime_error("default_node_type() for CLGNetwork is not defined.");
    }

    std::vector<std::shared_ptr<FactorType>> data_default_node_type(
        const std::shared_ptr<DataType>& dt) const override {
        switch (dt->id()) {
            case Type::DOUBLE:
            case Type::FLOAT:
                return {LinearGaussianCPDType::get()};
            case Type::DICTIONARY:
                return {DiscreteFactorType::get()};
            default:
                throw std::invalid_argument("Data type [" + dt->ToString() + "] not compatible with CLGNetworkType");
        }
    }

    bool compatible_node_type(const BayesianNetworkBase& m,
                              const std::string& var,
                              const std::shared_ptr<FactorType>& nt) const override {
        if (*nt != LinearGaussianCPDType::get_ref() && *nt != DiscreteFactorType::get_ref()) return false;

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
        if (*nt != LinearGaussianCPDType::get_ref() && *nt != DiscreteFactorType::get_ref()) return false;

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
        return *m.node_type(target) == LinearGaussianCPDType::get_ref() ||
               *m.node_type(source) != LinearGaussianCPDType::get_ref();
    }

    bool can_have_arc(const ConditionalBayesianNetworkBase& m,
                      const std::string& source,
                      const std::string& target) const override {
        return can_have_arc(static_cast<const BayesianNetworkBase&>(m), source, target);
    }

    std::string ToString() const override { return "CLGNetworkType"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<CLGNetworkType> __setstate__(py::tuple&) { return CLGNetworkType::get(); }

    static std::shared_ptr<CLGNetworkType> __setstate__(py::tuple&&) { return CLGNetworkType::get(); }

private:
    CLGNetworkType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

class CLGNetwork : public clone_inherit<CLGNetwork, BayesianNetwork> {
public:
    CLGNetwork(const std::vector<std::string>& nodes) : clone_inherit(CLGNetworkType::get(), nodes) {}

    CLGNetwork(const std::vector<std::string>& nodes, const FactorTypeVector& node_types)
        : clone_inherit(CLGNetworkType::get(), nodes, node_types) {}

    CLGNetwork(const ArcStringVector& arcs) : clone_inherit(CLGNetworkType::get(), arcs) {}

    CLGNetwork(const ArcStringVector& arcs, const FactorTypeVector& node_types)
        : clone_inherit(CLGNetworkType::get(), arcs, node_types) {}

    CLGNetwork(const std::vector<std::string>& nodes, const ArcStringVector& arcs)
        : clone_inherit(CLGNetworkType::get(), nodes, arcs) {}

    CLGNetwork(const std::vector<std::string>& nodes, const ArcStringVector& arcs, const FactorTypeVector& node_types)
        : clone_inherit(CLGNetworkType::get(), nodes, arcs, node_types) {}

    CLGNetwork(const Dag& graph) : clone_inherit(CLGNetworkType::get(), graph) {}

    CLGNetwork(const Dag& graph, const FactorTypeVector& node_types)
        : clone_inherit(CLGNetworkType::get(), graph, node_types) {}

    CLGNetwork(Dag&& graph) : clone_inherit(CLGNetworkType::get(), std::move(graph)) {}

    CLGNetwork(Dag&& graph, const FactorTypeVector& node_types)
        : clone_inherit(CLGNetworkType::get(), std::move(graph), node_types) {}

    std::string ToString() const override { return "CLGNetwork"; }
};

class ConditionalCLGNetwork : public clone_inherit<ConditionalCLGNetwork, ConditionalBayesianNetwork> {
public:
    ConditionalCLGNetwork(const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes)
        : clone_inherit(CLGNetworkType::get(), nodes, interface_nodes) {}

    ConditionalCLGNetwork(const std::vector<std::string>& nodes,
                          const std::vector<std::string>& interface_nodes,
                          const FactorTypeVector& node_types)
        : clone_inherit(CLGNetworkType::get(), nodes, interface_nodes, node_types) {}

    ConditionalCLGNetwork(const std::vector<std::string>& nodes,
                          const std::vector<std::string>& interface_nodes,
                          const ArcStringVector& arcs)
        : clone_inherit(CLGNetworkType::get(), nodes, interface_nodes, arcs) {}

    ConditionalCLGNetwork(const std::vector<std::string>& nodes,
                          const std::vector<std::string>& interface_nodes,
                          const ArcStringVector& arcs,
                          const FactorTypeVector& node_types)
        : clone_inherit(CLGNetworkType::get(), nodes, interface_nodes, arcs, node_types) {}

    ConditionalCLGNetwork(const ConditionalDag& graph) : clone_inherit(CLGNetworkType::get(), graph) {}

    ConditionalCLGNetwork(const ConditionalDag& graph, const FactorTypeVector& node_types)
        : clone_inherit(CLGNetworkType::get(), graph, node_types) {}

    ConditionalCLGNetwork(ConditionalDag&& graph) : clone_inherit(CLGNetworkType::get(), std::move(graph)) {}

    ConditionalCLGNetwork(ConditionalDag&& graph, const FactorTypeVector& node_types)
        : clone_inherit(CLGNetworkType::get(), std::move(graph), node_types) {}

    std::string ToString() const override { return "ConditionalCLGNetwork"; }
};

class DynamicCLGNetwork : public clone_inherit<DynamicCLGNetwork, DynamicBayesianNetwork> {
public:
    DynamicCLGNetwork(const std::vector<std::string>& variables, int markovian_order)
        : clone_inherit(CLGNetworkType::get(), variables, markovian_order) {}

    DynamicCLGNetwork(const std::vector<std::string>& variables,
                      int markovian_order,
                      std::shared_ptr<BayesianNetworkBase> static_bn,
                      std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn)
        : clone_inherit(variables, markovian_order, static_bn, transition_bn) {
        if (static_bn->type_ref() != CLGNetworkType::get_ref())
            throw std::invalid_argument("Bayesian networks are not Gaussian.");
    }

    std::string ToString() const override { return "DynamicCLGNetwork"; }
};

}  // namespace models

#endif  // PYBNESIAN_MODELS_CLGNETWORK_HPP