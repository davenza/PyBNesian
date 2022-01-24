#ifndef PYBNESIAN_MODELS_HETEROGENEOUS_HPP
#define PYBNESIAN_MODELS_HETEROGENEOUS_HPP

#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <util/hash_utils.hpp>

namespace models {

class DataTypeHash {
public:
    size_t operator()(const std::shared_ptr<DataType>& dt) const { return dt->id(); }
};

class DataTypeEqualTo {
public:
    bool operator()(const std::shared_ptr<DataType>& lhs, const std::shared_ptr<DataType>& rhs) const {
        return lhs->id() == rhs->id();
    }
};

using MapDataToFactor = std::
    unordered_map<std::shared_ptr<DataType>, std::vector<std::shared_ptr<FactorType>>, DataTypeHash, DataTypeEqualTo>;

MapDataToFactor& keep_MapDataToFactor_alive(MapDataToFactor& m);
MapDataToFactor keep_MapDataToFactor_alive(const MapDataToFactor& m);

class HeterogeneousBNType : public BayesianNetworkType {
public:
    HeterogeneousBNType(const HeterogeneousBNType&) = delete;
    void operator=(const HeterogeneousBNType&) = delete;

    HeterogeneousBNType(HeterogeneousBNType&&) = default;
    HeterogeneousBNType& operator=(HeterogeneousBNType&&) = default;

    HeterogeneousBNType(std::vector<std::shared_ptr<FactorType>> default_ft)
        : m_default_ftype(default_ft), m_default_ftypes(), m_single_default(true) {
        if (default_ft.empty()) throw std::invalid_argument("Default factor_type cannot be empty.");
        auto obj = py::cast(this);

        m_hash = reinterpret_cast<std::uintptr_t>(obj.get_type().ptr());

        for (const auto& f : m_default_ftype) {
            if (f) {
                util::hash_combine(m_hash, f->hash());
            } else {
                throw std::invalid_argument("Default factor_type cannot contain null FactorType.");
            }
        }
    }

    HeterogeneousBNType(MapDataToFactor default_fts)
        : m_default_ftype(), m_default_ftypes(default_fts), m_single_default(false) {
        for (auto it = m_default_ftypes.begin(), end = m_default_ftypes.end(); it != end;) {
            if (it->second.empty()) {
                it = m_default_ftypes.erase(it);
            } else {
                ++it;
            }
        }

        if (m_default_ftypes.empty()) throw std::invalid_argument("Default factor_type cannot be empty.");

        auto obj = py::cast(this);
        m_hash = reinterpret_cast<std::uintptr_t>(obj.get_type().ptr());

        // Based on the hashing of a frozenset
        // https://stackoverflow.com/questions/20832279/python-frozenset-hashing-algorithm-implementation
        // https://github.com/python/cpython/blob/main/Objects/setobject.c
        for (const auto& ft : m_default_ftypes) {
            if (ft.first == nullptr) throw std::invalid_argument("Default factor_types cannot contain null DataType.");

            auto nt_hash = ft.second.size();
            for (const auto& f : ft.second) {
                if (f) {
                    util::hash_combine(nt_hash, f->hash());
                } else {
                    throw std::invalid_argument("Default factor_type cannot contain null FactorType.");
                }
            }

            auto partial_hash = ft.first->Hash();
            util::hash_combine(partial_hash, nt_hash);

            m_hash ^= ((partial_hash ^ 89869747UL) ^ (partial_hash << 16)) * 3644798167UL;
        }

        /* Factor in the number of active entries */
        m_hash ^= (m_default_ftypes.size() + 1) * 1927868237UL;
    }

    bool is_homogeneous() const override { return false; }

    std::shared_ptr<FactorType> default_node_type() const override {
        throw std::runtime_error("default_node_type() for HeterogeneousBN is not defined.");
    }

    std::vector<std::shared_ptr<FactorType>> data_default_node_type(
        const std::shared_ptr<DataType>& dt) const override {
        if (m_single_default) {
            return m_default_ftype;
        } else {
            auto it = m_default_ftypes.find(dt);

            if (it == m_default_ftypes.end()) {
                throw std::invalid_argument("Not valid FactorType for DataType " + dt->ToString());
            } else {
                return it->second;
            }
        }
    }

    bool single_default() const { return m_single_default; }

    MapDataToFactor default_node_types() const {
        if (m_single_default) {
            MapDataToFactor ret;
            ret.insert({nullptr, m_default_ftype});
            return ret;
        } else {
            return m_default_ftypes;
        }
    }

    std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const override;
    std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override;

    std::string ToString() const override {
        std::stringstream ss;
        ss << "HeterogeneousBNType(";
        if (m_single_default) {
            ss << "[" << m_default_ftype[0]->ToString();

            for (auto it = ++m_default_ftype.begin(), end = m_default_ftype.end(); it != end; ++it) {
                ss << ", " << (*it)->ToString();
            }

            ss << "])";
        } else {
            ss << "{";

            auto beg = m_default_ftypes.begin();
            ss << beg->first->ToString() << ": [" << beg->second[0]->ToString();
            for (auto it = ++beg->second.begin(), end = beg->second.end(); it != end; ++it) {
                ss << ", " << (*it)->ToString();
            }

            ss << "]";

            for (auto dt_it = ++m_default_ftypes.begin(), dt_end = m_default_ftypes.end(); dt_it != dt_end; ++dt_it) {
                ss << ", " << dt_it->first->ToString() << ": [" << dt_it->second[0]->ToString();
                for (auto it = ++dt_it->second.begin(), end = dt_it->second.end(); it != end; ++it) {
                    ss << ", " << (*it)->ToString();
                }
                ss << "]";
            }

            ss << "})";
        }

        return ss.str();
    }

    py::tuple __getstate__() const override {
        return py::make_tuple(m_default_ftype, m_default_ftypes, m_single_default);
    }

    static std::shared_ptr<HeterogeneousBNType> __setstate__(py::tuple& t) {
        auto single_default = t[2].cast<bool>();

        if (single_default) {
            auto node_types = t[0].cast<std::vector<std::shared_ptr<FactorType>>>();
            FactorType::keep_vector_python_alive(node_types);
            return std::make_shared<HeterogeneousBNType>(node_types);
        } else {
            auto node_types = t[1].cast<MapDataToFactor>();
            models::keep_MapDataToFactor_alive(node_types);
            return std::make_shared<HeterogeneousBNType>(node_types);
        }
    }

    static std::shared_ptr<HeterogeneousBNType> __setstate__(py::tuple&& t) { return __setstate__(t); }

private:
    std::vector<std::shared_ptr<FactorType>> m_default_ftype;
    MapDataToFactor m_default_ftypes;
    bool m_single_default;
};

class HeterogeneousBN : public clone_inherit<HeterogeneousBN, BayesianNetwork> {
public:
    HeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft, const std::vector<std::string>& nodes)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes) {}

    HeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                    const std::vector<std::string>& nodes,
                    const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes, node_types) {}

    HeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft, const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), arcs) {}

    HeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                    const ArcStringVector& arcs,
                    const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), arcs, node_types) {}

    HeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                    const std::vector<std::string>& nodes,
                    const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes, arcs) {}

    HeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                    const std::vector<std::string>& nodes,
                    const ArcStringVector& arcs,
                    const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes, arcs, node_types) {}

    HeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft, const Dag& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), graph) {}

    HeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft, const Dag& graph, const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), graph, node_types) {}

    HeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft, Dag&& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), std::move(graph)) {}

    HeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft, Dag&& graph, const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), std::move(graph), node_types) {}

    HeterogeneousBN(MapDataToFactor fts, const std::vector<std::string>& nodes)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), nodes) {}

    HeterogeneousBN(MapDataToFactor fts, const std::vector<std::string>& nodes, const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), nodes, node_types) {}

    HeterogeneousBN(MapDataToFactor fts, const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), arcs) {}

    HeterogeneousBN(MapDataToFactor fts, const ArcStringVector& arcs, const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), arcs, node_types) {}

    HeterogeneousBN(MapDataToFactor fts, const std::vector<std::string>& nodes, const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), nodes, arcs) {}

    HeterogeneousBN(MapDataToFactor fts,
                    const std::vector<std::string>& nodes,
                    const ArcStringVector& arcs,
                    const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), nodes, arcs, node_types) {}

    HeterogeneousBN(MapDataToFactor fts, const Dag& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), graph) {}

    HeterogeneousBN(MapDataToFactor fts, const Dag& graph, const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), graph, node_types) {}

    HeterogeneousBN(MapDataToFactor fts, Dag&& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), std::move(graph)) {}

    HeterogeneousBN(MapDataToFactor fts, Dag&& graph, const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), std::move(graph), node_types) {}

    std::string ToString() const override { return "HeterogeneousBN"; }
};

class ConditionalHeterogeneousBN : public clone_inherit<ConditionalHeterogeneousBN, ConditionalBayesianNetwork> {
public:
    ConditionalHeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                               const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes, interface_nodes) {}

    ConditionalHeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                               const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes,
                               const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes, interface_nodes, node_types) {}

    ConditionalHeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                               const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes,
                               const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes, interface_nodes, arcs) {}

    ConditionalHeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                               const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes,
                               const ArcStringVector& arcs,
                               const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), nodes, interface_nodes, arcs, node_types) {}

    ConditionalHeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft, const ConditionalDag& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), graph) {}

    ConditionalHeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                               const ConditionalDag& graph,
                               const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), graph, node_types) {}

    ConditionalHeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft, ConditionalDag&& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), std::move(graph)) {}

    ConditionalHeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                               ConditionalDag&& graph,
                               const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), std::move(graph), node_types) {}

    ConditionalHeterogeneousBN(MapDataToFactor fts,
                               const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), nodes, interface_nodes) {}

    ConditionalHeterogeneousBN(MapDataToFactor fts,
                               const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes,
                               const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), nodes, interface_nodes, node_types) {}

    ConditionalHeterogeneousBN(MapDataToFactor fts,
                               const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes,
                               const ArcStringVector& arcs)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), nodes, interface_nodes, arcs) {}

    ConditionalHeterogeneousBN(MapDataToFactor fts,
                               const std::vector<std::string>& nodes,
                               const std::vector<std::string>& interface_nodes,
                               const ArcStringVector& arcs,
                               const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), nodes, interface_nodes, arcs, node_types) {}

    ConditionalHeterogeneousBN(MapDataToFactor fts, const ConditionalDag& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), graph) {}

    ConditionalHeterogeneousBN(MapDataToFactor fts, const ConditionalDag& graph, const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), graph, node_types) {}

    ConditionalHeterogeneousBN(MapDataToFactor fts, ConditionalDag&& graph)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), std::move(graph)) {}

    ConditionalHeterogeneousBN(MapDataToFactor fts, ConditionalDag&& graph, const FactorTypeVector& node_types)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), std::move(graph), node_types) {}

    std::string ToString() const override { return "ConditionalHeterogeneousBN"; }
};

class DynamicHeterogeneousBN : public clone_inherit<DynamicHeterogeneousBN, DynamicBayesianNetwork> {
public:
    DynamicHeterogeneousBN(std::vector<std::shared_ptr<FactorType>> ft,
                           const std::vector<std::string>& variables,
                           int markovian_order)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(ft), variables, markovian_order) {}

    DynamicHeterogeneousBN(MapDataToFactor fts, const std::vector<std::string>& variables, int markovian_order)
        : clone_inherit(std::make_shared<HeterogeneousBNType>(fts), variables, markovian_order) {}

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

template <typename DerivedBN>
std::shared_ptr<DerivedBN> __heterogeneous_setstate__(py::tuple& t) {
    using DagType = typename DerivedBN::DagClass;
    if (t.size() != 5) throw std::runtime_error("Not valid BayesianNetwork.");

    auto dag = t[0].cast<DagType>();
    auto type = t[1].cast<std::shared_ptr<BayesianNetworkType>>();

    auto node_types = t[2].cast<FactorTypeVector>();
    util::keep_FactorTypeVector_python_alive(node_types);
    auto dwn_type = std::static_pointer_cast<HeterogeneousBNType>(type);

    std::shared_ptr<DerivedBN> bn = [&node_types, &dwn_type, &dag]() {
        if (node_types.empty()) {
            if (dwn_type->single_default()) {
                auto nts = dwn_type->default_node_types();
                auto nt = nts.find(nullptr);
                return std::make_shared<DerivedBN>(nt->second, std::move(dag));
            } else {
                return std::make_shared<DerivedBN>(dwn_type->default_node_types(), std::move(dag));
            }
        }

        if (dwn_type->single_default()) {
            if constexpr (std::is_constructible_v<DerivedBN,
                                                  std::vector<std::shared_ptr<FactorType>>,
                                                  DagType&&,
                                                  FactorTypeVector>) {
                auto nts = dwn_type->default_node_types();
                auto nt = nts.find(nullptr);
                return std::make_shared<DerivedBN>(nt->second, std::move(dag), node_types);
            } else {
                throw std::runtime_error("Invalid node types array for heterogeneous Bayesian network.");
            }
        } else {
            if constexpr (std::is_constructible_v<DerivedBN, const MapDataToFactor&, DagType&&, FactorTypeVector>) {
                return std::make_shared<DerivedBN>(dwn_type->default_node_types(), std::move(dag), node_types);
            } else {
                throw std::runtime_error("Invalid node types array for heterogeneous Bayesian network.");
            }
        }
    }();

    if (t[3].cast<bool>()) {
        auto cpds = t[4].cast<std::vector<std::shared_ptr<Factor>>>();
        Factor::keep_vector_python_alive(cpds);
        bn->add_cpds(cpds);
    }

    return bn;
}

}  // namespace models

#endif  // PYBNESIAN_MODELS_HETEROGENEOUS_HPP