#ifndef PYBNESIAN_MODELS_HOMOGENEOUSBN_HPP
#define PYBNESIAN_MODELS_HOMOGENEOUSBN_HPP

#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <util/hash_utils.hpp>

namespace models {

class HomogeneousBNType : public BayesianNetworkType {
public:
    HomogeneousBNType(const HomogeneousBNType&) = delete;
    void operator=(const HomogeneousBNType&) = delete;

    HomogeneousBNType(HomogeneousBNType&&) = default;
    HomogeneousBNType& operator=(HomogeneousBNType&&) = default;

    HomogeneousBNType(std::shared_ptr<FactorType> ft) : m_ftype(ft) {
        if (ft == nullptr) throw std::invalid_argument("factor_type cannot be null.");

        auto obj = py::cast(this);

        m_hash = reinterpret_cast<std::uintptr_t>(obj.get_type().ptr());
        util::hash_combine(m_hash, ft->hash());
    }

    bool is_homogeneous() const override { return true; }

    std::shared_ptr<FactorType> default_node_type() const override { return m_ftype; }
    std::vector<std::shared_ptr<FactorType>> data_default_node_type(const std::shared_ptr<DataType>&) const override {
        return {m_ftype};
    }

    std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const override;
    std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override;

    std::string ToString() const override { return "HomogeneousBNType(" + m_ftype->ToString() + ")"; }

    py::tuple __getstate__() const override { return py::make_tuple(m_ftype); }

    static std::shared_ptr<HomogeneousBNType> __setstate__(py::tuple& t) {
        auto ft = t[0].cast<std::shared_ptr<FactorType>>();
        FactorType::keep_python_alive(ft);
        return std::make_shared<HomogeneousBNType>(ft);
    }

    static std::shared_ptr<HomogeneousBNType> __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    std::shared_ptr<FactorType> m_ftype;
};

class HomogeneousBN : public clone_inherit<HomogeneousBN, BayesianNetwork> {
public:
    HomogeneousBN(std::shared_ptr<FactorType> ft, const std::vector<std::string>& nodes)
        : clone_inherit(std::make_shared<HomogeneousBNType>(ft), nodes) {}

    HomogeneousBN(std::shared_ptr<FactorType> ft, const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HomogeneousBNType>(ft), arcs) {}

    HomogeneousBN(std::shared_ptr<FactorType> ft, const std::vector<std::string>& nodes, const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HomogeneousBNType>(ft), nodes, arcs) {}

    HomogeneousBN(std::shared_ptr<FactorType> ft, const Dag& graph)
        : clone_inherit(std::make_shared<HomogeneousBNType>(ft), graph) {}

    HomogeneousBN(std::shared_ptr<FactorType> ft, Dag&& graph)
        : clone_inherit(std::make_shared<HomogeneousBNType>(ft), std::move(graph)) {}

    std::string ToString() const override { return "HomogeneousBN"; }
};

class ConditionalHomogeneousBN : public clone_inherit<ConditionalHomogeneousBN, ConditionalBayesianNetwork> {
public:
    ConditionalHomogeneousBN(std::shared_ptr<FactorType> ft,
                             const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes)
        : clone_inherit(std::make_shared<HomogeneousBNType>(ft), nodes, interface_nodes) {}

    ConditionalHomogeneousBN(std::shared_ptr<FactorType> ft,
                             const std::vector<std::string>& nodes,
                             const std::vector<std::string>& interface_nodes,
                             const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HomogeneousBNType>(ft), nodes, interface_nodes, arcs) {}

    ConditionalHomogeneousBN(std::shared_ptr<FactorType> ft, const ConditionalDag& graph)
        : clone_inherit(std::make_shared<HomogeneousBNType>(ft), graph) {}

    ConditionalHomogeneousBN(std::shared_ptr<FactorType> ft, ConditionalDag&& graph)
        : clone_inherit(std::make_shared<HomogeneousBNType>(ft), std::move(graph)) {}

    std::string ToString() const override { return "ConditionalHomogeneousBN"; }
};

class DynamicHomogeneousBN : public clone_inherit<DynamicHomogeneousBN, DynamicBayesianNetwork> {
public:
    DynamicHomogeneousBN(std::shared_ptr<FactorType> ft, const std::vector<std::string>& variables, int markovian_order)
        : clone_inherit(std::make_shared<HomogeneousBNType>(ft), variables, markovian_order) {}

    DynamicHomogeneousBN(const std::vector<std::string>& variables,
                         int markovian_order,
                         std::shared_ptr<BayesianNetworkBase> static_bn,
                         std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn)
        : clone_inherit(variables, markovian_order, static_bn, transition_bn) {
        auto ptr_type = &static_bn->type_ref();
        auto homogeneous = dynamic_cast<HomogeneousBNType*>(ptr_type);

        if (!homogeneous) {
            throw std::invalid_argument("Bayesian networks are not HomogeneousBNType.");
        }
    }

    std::string ToString() const override { return "DynamicHomogeneousBN"; }
};

template <typename DerivedBN>
std::shared_ptr<DerivedBN> __homogeneous_setstate__(py::tuple& t) {
    using DagType = typename DerivedBN::DagClass;
    if (t.size() != 5) throw std::runtime_error("Not valid BayesianNetwork.");

    auto dag = t[0].cast<DagType>();
    auto type = t[1].cast<std::shared_ptr<BayesianNetworkType>>();
    auto bn = std::make_shared<DerivedBN>(type->default_node_type(), std::move(dag));

    if (t[3].cast<bool>()) {
        auto cpds = t[4].cast<std::vector<std::shared_ptr<Factor>>>();
        Factor::keep_vector_python_alive(cpds);
        bn->add_cpds(cpds);
    }

    return bn;
}

}  // namespace models

#endif  // PYBNESIAN_MODELS_HOMOGENEOUSBN_HPP