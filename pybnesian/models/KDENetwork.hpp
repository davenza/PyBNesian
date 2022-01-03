#ifndef PYBNESIAN_MODELS_KDENETWORK_HPP
#define PYBNESIAN_MODELS_KDENETWORK_HPP

#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <factors/continuous/CKDE.hpp>

using factors::continuous::CKDEType;

namespace models {

class KDENetworkType : public BayesianNetworkType {
public:
    KDENetworkType(const KDENetworkType&) = delete;
    void operator=(const KDENetworkType&) = delete;

    static std::shared_ptr<KDENetworkType> get() {
        static std::shared_ptr<KDENetworkType> singleton = std::shared_ptr<KDENetworkType>(new KDENetworkType);
        return singleton;
    }

    static KDENetworkType& get_ref() {
        static KDENetworkType& ref = *KDENetworkType::get();
        return ref;
    }

    std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const override;
    std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override;

    bool is_homogeneous() const override { return true; }

    std::shared_ptr<FactorType> default_node_type() const override { return CKDEType::get(); }
    std::vector<std::shared_ptr<FactorType>> data_default_node_type(
        const std::shared_ptr<DataType>& dt) const override {
        switch (dt->id()) {
            case Type::DOUBLE:
            case Type::FLOAT:
                return {CKDEType::get()};
            default:
                throw std::invalid_argument("Data type [" + dt->ToString() + "] not compatible with KDENetworkType");
        }
    }

    std::string ToString() const override { return "KDENetworkType"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<KDENetworkType> __setstate__(py::tuple&) { return KDENetworkType::get(); }

    static std::shared_ptr<KDENetworkType> __setstate__(py::tuple&&) { return KDENetworkType::get(); }

private:
    KDENetworkType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

class KDENetwork : public clone_inherit<KDENetwork, BayesianNetwork> {
public:
    KDENetwork(const std::vector<std::string>& nodes) : clone_inherit(KDENetworkType::get(), nodes) {}

    KDENetwork(const ArcStringVector& arcs) : clone_inherit(KDENetworkType::get(), arcs) {}

    KDENetwork(const std::vector<std::string>& nodes, const ArcStringVector& arcs)
        : clone_inherit(KDENetworkType::get(), nodes, arcs) {}

    KDENetwork(const Dag& graph) : clone_inherit(KDENetworkType::get(), graph) {}
    KDENetwork(Dag&& graph) : clone_inherit(KDENetworkType::get(), std::move(graph)) {}

    std::string ToString() const override { return "KDENetwork"; }
};

class ConditionalKDENetwork : public clone_inherit<ConditionalKDENetwork, ConditionalBayesianNetwork> {
public:
    ConditionalKDENetwork(const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes)
        : clone_inherit(KDENetworkType::get(), nodes, interface_nodes) {}

    ConditionalKDENetwork(const std::vector<std::string>& nodes,
                          const std::vector<std::string>& interface_nodes,
                          const ArcStringVector& arcs)
        : clone_inherit(KDENetworkType::get(), nodes, interface_nodes, arcs) {}

    ConditionalKDENetwork(const ConditionalDag& graph) : clone_inherit(KDENetworkType::get(), graph) {}

    ConditionalKDENetwork(ConditionalDag&& graph) : clone_inherit(KDENetworkType::get(), std::move(graph)) {}

    std::string ToString() const override { return "ConditionalKDENetwork"; }
};

class DynamicKDENetwork : public clone_inherit<DynamicKDENetwork, DynamicBayesianNetwork> {
public:
    DynamicKDENetwork(const std::vector<std::string>& variables, int markovian_order)
        : clone_inherit(KDENetworkType::get(), variables, markovian_order) {}

    DynamicKDENetwork(const std::vector<std::string>& variables,
                      int markovian_order,
                      std::shared_ptr<BayesianNetworkBase> static_bn,
                      std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn)
        : clone_inherit(variables, markovian_order, static_bn, transition_bn) {
        if (static_bn->type_ref() != KDENetworkType::get_ref())
            throw std::invalid_argument("Bayesian networks are not KDE networks.");
    }

    std::string ToString() const override { return "DynamicKDENetwork"; }
};

}  // namespace models

#endif  // PYBNESIAN_MODELS_KDENETWORK_HPP