#ifndef PYBNESIAN_MODELS_DISCRETEBN_HPP
#define PYBNESIAN_MODELS_DISCRETEBN_HPP

#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <util/virtual_clone.hpp>

using factors::discrete::DiscreteFactorType, factors::discrete::DiscreteFactor;
using models::DynamicBayesianNetwork;
using util::clone_inherit;

namespace models {

class DiscreteBNType : public BayesianNetworkType {
public:
    DiscreteBNType(const DiscreteBNType&) = delete;
    void operator=(const DiscreteBNType&) = delete;

    static std::shared_ptr<DiscreteBNType> get() {
        static std::shared_ptr<DiscreteBNType> singleton = std::shared_ptr<DiscreteBNType>(new DiscreteBNType);
        return singleton;
    }

    static DiscreteBNType& get_ref() {
        static DiscreteBNType& ref = *DiscreteBNType::get();
        return ref;
    }

    std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const override;
    std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override;

    bool is_homogeneous() const override { return true; }

    std::shared_ptr<FactorType> default_node_type() const override { return DiscreteFactorType::get(); }
    std::vector<std::shared_ptr<FactorType>> data_default_node_type(
        const std::shared_ptr<DataType>& dt) const override {
        switch (dt->id()) {
            case Type::DICTIONARY:
                return {DiscreteFactorType::get()};
            default:
                throw std::invalid_argument("Data type [" + dt->ToString() +
                                            "] not compatible with DiscreteFactorType");
        }
    }

    std::string ToString() const override { return "DiscreteNetworkType"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<DiscreteBNType> __setstate__(py::tuple&) { return DiscreteBNType::get(); }

    static std::shared_ptr<DiscreteBNType> __setstate__(py::tuple&&) { return DiscreteBNType::get(); }

private:
    DiscreteBNType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

class DiscreteBN : public clone_inherit<DiscreteBN, BayesianNetwork> {
public:
    DiscreteBN(const std::vector<std::string>& nodes) : clone_inherit(DiscreteBNType::get(), nodes) {}

    DiscreteBN(const ArcStringVector& arcs) : clone_inherit(DiscreteBNType::get(), arcs) {}

    DiscreteBN(const std::vector<std::string>& nodes, const ArcStringVector& arcs)
        : clone_inherit(DiscreteBNType::get(), nodes, arcs) {}

    DiscreteBN(const Dag& graph) : clone_inherit(DiscreteBNType::get(), graph) {}
    DiscreteBN(Dag&& graph) : clone_inherit(DiscreteBNType::get(), std::move(graph)) {}

    std::string ToString() const override { return "DiscreteNetwork"; }
};

class ConditionalDiscreteBN : public clone_inherit<ConditionalDiscreteBN, ConditionalBayesianNetwork> {
public:
    ConditionalDiscreteBN(const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes)
        : clone_inherit(DiscreteBNType::get(), nodes, interface_nodes) {}

    ConditionalDiscreteBN(const std::vector<std::string>& nodes,
                          const std::vector<std::string>& interface_nodes,
                          const ArcStringVector& arcs)
        : clone_inherit(DiscreteBNType::get(), nodes, interface_nodes, arcs) {}

    ConditionalDiscreteBN(const ConditionalDag& graph) : clone_inherit(DiscreteBNType::get(), graph) {}
    ConditionalDiscreteBN(ConditionalDag&& graph) : clone_inherit(DiscreteBNType::get(), std::move(graph)) {}

    std::string ToString() const override { return "ConditionalDiscreteNetwork"; }
};

class DynamicDiscreteBN : public clone_inherit<DynamicDiscreteBN, DynamicBayesianNetwork> {
public:
    DynamicDiscreteBN(const std::vector<std::string>& variables, int markovian_order)
        : clone_inherit(DiscreteBNType::get(), variables, markovian_order) {}

    DynamicDiscreteBN(const std::vector<std::string>& variables,
                      int markovian_order,
                      std::shared_ptr<BayesianNetworkBase> static_bn,
                      std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn)
        : clone_inherit(variables, markovian_order, static_bn, transition_bn) {
        if (static_bn->type_ref() != DiscreteBNType::get_ref())
            throw std::invalid_argument("Bayesian networks are not discrete.");
    }

    std::string ToString() const override { return "DynamicDiscreteNetwork"; }
};

}  // namespace models

#endif  // PYBNESIAN_MODELS_DISCRETEBN_HPP
