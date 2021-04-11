#ifndef PYBNESIAN_MODELS_HETEROGENEOUS_HPP
#define PYBNESIAN_MODELS_HETEROGENEOUS_HPP

#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <util/hash_utils.hpp>

namespace models {

class HeterogeneousBNType : public BayesianNetworkType {
public:
    HeterogeneousBNType(const HeterogeneousBNType&) = delete;
    void operator=(const HeterogeneousBNType&) = delete;

    HeterogeneousBNType(std::shared_ptr<FactorType> default_ft) : m_default_ftype(default_ft) {
        if (default_ft == nullptr) throw std::invalid_argument("Default factor_type can not be null.");

        auto obj = py::cast(this);

        m_hash = reinterpret_cast<std::uintptr_t>(obj.get_type().ptr());
        util::hash_combine(m_hash, default_ft->hash());
    }

    bool is_homogeneous() const override { return false; }

    std::shared_ptr<FactorType> default_node_type() const override { return m_default_ftype; }

    std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const override;
    std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override;

    std::string ToString() const override { return "HeterogeneousBNType(" + m_default_ftype->ToString() + ")"; }

    py::tuple __getstate__() const override { return py::make_tuple(m_default_ftype); }

    static std::shared_ptr<HeterogeneousBNType> __setstate__(py::tuple& t) {
        auto default_ftype = t[0].cast<std::shared_ptr<FactorType>>();
        return std::make_shared<HeterogeneousBNType>(default_ftype);
    }

    static std::shared_ptr<HeterogeneousBNType> __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    std::shared_ptr<FactorType> m_default_ftype;
};

class HeterogeneousBN : public clone_inherit<HeterogeneousBN, BayesianNetwork> {
public:
    HeterogeneousBN(std::shared_ptr<FactorType> ft, const std::vector<std::string>& nodes)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes) {}

    HeterogeneousBN(std::shared_ptr<FactorType> ft, const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), arcs) {}

    HeterogeneousBN(std::shared_ptr<FactorType> ft, const std::vector<std::string>& nodes, const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes, arcs) {}

    HeterogeneousBN(std::shared_ptr<FactorType> ft, const Dag& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), graph) {}
    HeterogeneousBN(std::shared_ptr<FactorType> ft, Dag&& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), std::move(graph)) {}

    std::string ToString() const override { return "HeterogeneousBN"; }
};

class ConditionalHeterogeneousBN : public clone_inherit<ConditionalHeterogeneousBN, ConditionalBayesianNetwork> {
public:
    ConditionalHeterogeneousBN(std::shared_ptr<FactorType> ft,
                               const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes, interface_nodes) {}

    ConditionalHeterogeneousBN(std::shared_ptr<FactorType> ft,
                               const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes,
                               const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes, interface_nodes, arcs) {}

    ConditionalHeterogeneousBN(std::shared_ptr<FactorType> ft, const ConditionalDag& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), graph) {}
    ConditionalHeterogeneousBN(std::shared_ptr<FactorType> ft, ConditionalDag&& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), std::move(graph)) {}

    std::string ToString() const override { return "ConditionalHeterogeneousBN"; }
};

class DynamicHeterogeneousBN : public clone_inherit<DynamicHeterogeneousBN, DynamicBayesianNetwork> {
public:
    DynamicHeterogeneousBN(std::shared_ptr<FactorType> ft,
                           const std::vector<std::string>& variables,
                           int markovian_order)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), variables, markovian_order) {}

    DynamicHeterogeneousBN(const std::vector<std::string>& variables,
                           int markovian_order,
                           std::shared_ptr<BayesianNetworkBase> static_bn,
                           std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn)
        : clone_inherit(variables, markovian_order, static_bn, transition_bn) {
        auto ptr_type = &static_bn->type_ref();
        auto heterogeneous = dynamic_cast<HeterogeneousBNType*>(ptr_type);

        if (!heterogeneous) {
            throw std::invalid_argument("Bayesian networks are not HeterogeneousBNType.");
        }
    }

    std::string ToString() const override { return "DynamicHeterogeneousBN"; }
};

}  // namespace models

#endif  // PYBNESIAN_MODELS_HETEROGENEOUS_HPP