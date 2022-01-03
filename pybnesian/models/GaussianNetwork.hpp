#ifndef PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP
#define PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP

#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>

using factors::continuous::LinearGaussianCPDType, factors::continuous::LinearGaussianCPD;

namespace models {

class GaussianNetworkType : public BayesianNetworkType {
public:
    GaussianNetworkType(const GaussianNetworkType&) = delete;
    void operator=(const GaussianNetworkType&) = delete;

    static std::shared_ptr<GaussianNetworkType> get() {
        static std::shared_ptr<GaussianNetworkType> singleton =
            std::shared_ptr<GaussianNetworkType>(new GaussianNetworkType);
        return singleton;
    }

    static GaussianNetworkType& get_ref() {
        static GaussianNetworkType& ref = *GaussianNetworkType::get();
        return ref;
    }

    std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const override;
    std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override;

    bool is_homogeneous() const override { return true; }

    std::shared_ptr<FactorType> default_node_type() const override { return LinearGaussianCPDType::get(); }
    std::vector<std::shared_ptr<FactorType>> data_default_node_type(
        const std::shared_ptr<DataType>& dt) const override {
        switch (dt->id()) {
            case Type::DOUBLE:
            case Type::FLOAT:
                return {LinearGaussianCPDType::get()};
            default:
                throw std::invalid_argument("Data type [" + dt->ToString() +
                                            "] not compatible with GaussianNetworkType");
        }
    }

    std::string ToString() const override { return "GaussianNetworkType"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<GaussianNetworkType> __setstate__(py::tuple&) { return GaussianNetworkType::get(); }

    static std::shared_ptr<GaussianNetworkType> __setstate__(py::tuple&&) { return GaussianNetworkType::get(); }

private:
    GaussianNetworkType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

class GaussianNetwork : public clone_inherit<GaussianNetwork, BayesianNetwork> {
public:
    GaussianNetwork(const std::vector<std::string>& nodes) : clone_inherit(GaussianNetworkType::get(), nodes) {}

    GaussianNetwork(const ArcStringVector& arcs) : clone_inherit(GaussianNetworkType::get(), arcs) {}

    GaussianNetwork(const std::vector<std::string>& nodes, const ArcStringVector& arcs)
        : clone_inherit(GaussianNetworkType::get(), nodes, arcs) {}

    GaussianNetwork(const Dag& graph) : clone_inherit(GaussianNetworkType::get(), graph) {}
    GaussianNetwork(Dag&& graph) : clone_inherit(GaussianNetworkType::get(), std::move(graph)) {}

    std::string ToString() const override { return "GaussianNetwork"; }
};

class ConditionalGaussianNetwork : public clone_inherit<ConditionalGaussianNetwork, ConditionalBayesianNetwork> {
public:
    ConditionalGaussianNetwork(const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes)
        : clone_inherit(GaussianNetworkType::get(), nodes, interface_nodes) {}

    ConditionalGaussianNetwork(const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes,
                               const ArcStringVector& arcs)
        : clone_inherit(GaussianNetworkType::get(), nodes, interface_nodes, arcs) {}

    ConditionalGaussianNetwork(const ConditionalDag& graph) : clone_inherit(GaussianNetworkType::get(), graph) {}
    ConditionalGaussianNetwork(ConditionalDag&& graph) : clone_inherit(GaussianNetworkType::get(), std::move(graph)) {}

    std::string ToString() const override { return "ConditionalGaussianNetwork"; }
};

class DynamicGaussianNetwork : public clone_inherit<DynamicGaussianNetwork, DynamicBayesianNetwork> {
public:
    DynamicGaussianNetwork(const std::vector<std::string>& variables, int markovian_order)
        : clone_inherit(GaussianNetworkType::get(), variables, markovian_order) {}

    DynamicGaussianNetwork(const std::vector<std::string>& variables,
                           int markovian_order,
                           std::shared_ptr<BayesianNetworkBase> static_bn,
                           std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn)
        : clone_inherit(variables, markovian_order, static_bn, transition_bn) {
        if (static_bn->type_ref() != GaussianNetworkType::get_ref())
            throw std::invalid_argument("Bayesian networks are not Gaussian.");
    }

    std::string ToString() const override { return "DynamicGaussianNetwork"; }
};

}  // namespace models

#endif  // PYBNESIAN_MODELS_GAUSSIANNETWORK_HPP
