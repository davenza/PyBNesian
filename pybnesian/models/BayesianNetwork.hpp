#ifndef PYBNESIAN_MODELS_BAYESIANNETWORK_HPP
#define PYBNESIAN_MODELS_BAYESIANNETWORK_HPP

#include <random>
#include <dataset/dataset.hpp>
#include <factors/factors.hpp>
#include <factors/arguments.hpp>
#include <factors/unknown_factor.hpp>
#include <graph/generic_graph.hpp>
#include <util/parameter_traits.hpp>
#include <util/virtual_clone.hpp>

using arrow::DataType;
using dataset::DataFrame;
using factors::Factor, factors::Arguments, factors::UnknownFactorType;
using graph::DagBase, graph::ConditionalDagBase, graph::Dag, graph::ConditionalDag;
using util::ArcStringVector, util::FactorTypeVector;

using util::abstract_class, util::clone_inherit, util::clone_inherit_condition;

using Field_ptr = std::shared_ptr<arrow::Field>;
using Array_ptr = std::shared_ptr<arrow::Array>;

namespace models {

class ConditionalBayesianNetworkBase;
class BayesianNetworkType;

class BayesianNetworkBase : public std::enable_shared_from_this<BayesianNetworkBase> {
public:
    virtual ~BayesianNetworkBase() = default;
    virtual bool is_python_derived() const { return false; }
    virtual const DagBase& graph() const = 0;
    virtual DagBase& graph() = 0;
    virtual int num_nodes() const = 0;
    virtual int num_arcs() const = 0;
    virtual const std::vector<std::string>& nodes() const = 0;
    virtual ArcStringVector arcs() const = 0;
    virtual const std::unordered_map<std::string, int>& indices() const = 0;
    virtual int index(const std::string& node) const = 0;
    virtual int collapsed_index(const std::string& node) const = 0;
    virtual int index_from_collapsed(int collapsed_index) const = 0;
    virtual int collapsed_from_index(int index) const = 0;
    virtual const std::unordered_map<std::string, int>& collapsed_indices() const = 0;
    virtual bool is_valid(const std::string& name) const = 0;
    virtual bool contains_node(const std::string& name) const = 0;
    virtual int add_node(const std::string& node) = 0;
    virtual void remove_node(const std::string& node) = 0;
    virtual const std::string& name(int node_index) const = 0;
    virtual const std::string& collapsed_name(int collapsed_index) const = 0;
    virtual int num_parents(const std::string& node) const = 0;
    virtual int num_children(const std::string& node) const = 0;
    virtual std::vector<std::string> parents(const std::string& node) const = 0;
    virtual std::vector<std::string> children(const std::string& node) const = 0;
    virtual bool has_arc(const std::string& source, const std::string& target) const = 0;
    virtual bool has_path(const std::string& source, const std::string& target) const = 0;
    virtual void add_arc(const std::string& source, const std::string& target) = 0;
    virtual void add_arc_unsafe(const std::string& source, const std::string& target) = 0;
    virtual void remove_arc(const std::string& source, const std::string& target) = 0;
    virtual void flip_arc(const std::string& source, const std::string& target) = 0;
    virtual void flip_arc_unsafe(const std::string& source, const std::string& target) = 0;
    virtual bool can_add_arc(const std::string& source, const std::string& target) const = 0;
    virtual bool can_flip_arc(const std::string& source, const std::string& target) const = 0;

    virtual void check_blacklist(const ArcStringVector& arc_blacklist) const {
        for (const auto& arc : arc_blacklist) {
            if (has_arc(arc.first, arc.second)) {
                throw std::invalid_argument("Arc " + arc.first + " -> " + arc.second +
                                            " in blacklist,"
                                            " but it is present in the Bayesian Network.");
            }
        }
    }

    virtual void force_whitelist(const ArcStringVector& arc_whitelist) = 0;
    virtual bool fitted() const = 0;
    virtual std::shared_ptr<Factor> cpd(const std::string& node) = 0;
    virtual const std::shared_ptr<Factor> cpd(const std::string& node) const = 0;
    virtual void add_cpds(const std::vector<std::shared_ptr<Factor>>& cpds) = 0;
    virtual void fit(const DataFrame& df, const Arguments& construction_args = Arguments()) = 0;
    virtual VectorXd logl(const DataFrame& df) const = 0;
    virtual double slogl(const DataFrame& df) const = 0;
    virtual std::shared_ptr<BayesianNetworkType> type() const = 0;
    virtual BayesianNetworkType& type_ref() const = 0;
    virtual DataFrame sample(int n, unsigned int seed = std::random_device{}(), bool ordered = false) const = 0;
    virtual std::shared_ptr<ConditionalBayesianNetworkBase> conditional_bn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const = 0;
    virtual std::shared_ptr<ConditionalBayesianNetworkBase> conditional_bn() const = 0;
    virtual std::shared_ptr<BayesianNetworkBase> unconditional_bn() const = 0;
    virtual void save(std::string name, bool include_cpd = false) const = 0;
    virtual bool include_cpd() const = 0;
    virtual void set_include_cpd(bool include_cpd) const = 0;
    virtual py::tuple __getstate__() const = 0;

    virtual std::shared_ptr<FactorType> node_type(const std::string& node) const = 0;
    virtual std::unordered_map<std::string, std::shared_ptr<FactorType>> node_types() const = 0;
    virtual std::shared_ptr<FactorType> underlying_node_type(const DataFrame&, const std::string& node) const = 0;
    virtual bool has_unknown_node_types() const = 0;
    virtual void set_node_type(const std::string& node, const std::shared_ptr<FactorType>& new_type) = 0;
    virtual void set_unknown_node_types(const DataFrame& df,
                                        const FactorTypeVector& type_blacklist = FactorTypeVector()) = 0;
    virtual void force_type_whitelist(const FactorTypeVector& type_whitelist) = 0;
    virtual std::string ToString() const = 0;

    std::shared_ptr<BayesianNetworkBase> clone() const {
        if (is_python_derived()) {
            auto self = py::cast(this);

            // Clone with pickle because it conserves the Python derived type and includes the extra info.
            auto bytes = py::module_::import("pickle").attr("dumps")(self);
            auto cloned = py::module_::import("pickle").attr("loads")(bytes);

            auto keep_python_state_alive = std::make_shared<py::object>(cloned);
            auto ptr = cloned.cast<BayesianNetworkBase*>();
            return std::shared_ptr<BayesianNetworkBase>(keep_python_state_alive, ptr);
        } else {
            return std::shared_ptr<BayesianNetworkBase>(clone_impl());
        }
    }

    static std::shared_ptr<BayesianNetworkBase>& keep_python_alive(std::shared_ptr<BayesianNetworkBase>& m) {
        if (m && m->is_python_derived()) {
            auto o = py::cast(m);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<BayesianNetworkBase*>();
            m = std::shared_ptr<BayesianNetworkBase>(keep_python_state_alive, ptr);
        }

        return m;
    }

    static std::shared_ptr<BayesianNetworkBase> keep_python_alive(const std::shared_ptr<BayesianNetworkBase>& m) {
        if (m && m->is_python_derived()) {
            auto o = py::cast(m);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<BayesianNetworkBase*>();
            return std::shared_ptr<BayesianNetworkBase>(keep_python_state_alive, ptr);
        }

        return m;
    }

private:
    virtual BayesianNetworkBase* clone_impl() const = 0;
};

class ConditionalBayesianNetworkBase : public BayesianNetworkBase {
public:
    virtual ~ConditionalBayesianNetworkBase() = default;
    virtual const ConditionalDagBase& graph() const = 0;
    virtual ConditionalDagBase& graph() = 0;
    virtual int num_interface_nodes() const = 0;
    virtual int num_joint_nodes() const = 0;
    virtual const std::vector<std::string>& interface_nodes() const = 0;
    virtual const std::vector<std::string>& joint_nodes() const = 0;
    virtual ArcStringVector interface_arcs() const = 0;
    virtual int interface_collapsed_index(const std::string& name) const = 0;
    virtual int joint_collapsed_index(const std::string& name) const = 0;
    virtual const std::unordered_map<std::string, int>& interface_collapsed_indices() const = 0;
    virtual const std::unordered_map<std::string, int>& joint_collapsed_indices() const = 0;
    virtual int index_from_interface_collapsed(int interface_collapsed_index) const = 0;
    virtual int index_from_joint_collapsed(int joint_collapsed_index) const = 0;
    virtual int interface_collapsed_from_index(int index) const = 0;
    virtual int joint_collapsed_from_index(int index) const = 0;
    virtual const std::string& interface_collapsed_name(int interface_collapsed_index) const = 0;
    virtual const std::string& joint_collapsed_name(int joint_collapsed_index) const = 0;
    virtual bool contains_interface_node(const std::string& name) const = 0;
    virtual bool contains_joint_node(const std::string& name) const = 0;
    virtual int add_interface_node(const std::string& node) = 0;
    virtual void remove_interface_node(const std::string& node) = 0;
    virtual bool is_interface(const std::string& name) const = 0;
    virtual void set_interface(const std::string& name) = 0;
    virtual void set_node(const std::string& name) = 0;
    using BayesianNetworkBase::sample;
    virtual DataFrame sample(const DataFrame& evidence,
                             unsigned int seed = std::random_device{}(),
                             bool concat_evidence = false,
                             bool ordered = false) const = 0;

    std::shared_ptr<ConditionalBayesianNetworkBase> clone() const {
        if (is_python_derived()) {
            auto self = py::cast(this);

            // Clone with pickle because it conserves the Python derived type and includes the extra info.
            auto bytes = py::module_::import("pickle").attr("dumps")(self);
            auto cloned = py::module_::import("pickle").attr("loads")(bytes);

            auto keep_python_state_alive = std::make_shared<py::object>(cloned);
            auto ptr = cloned.cast<ConditionalBayesianNetworkBase*>();
            return std::shared_ptr<ConditionalBayesianNetworkBase>(keep_python_state_alive, ptr);
        } else {
            return std::shared_ptr<ConditionalBayesianNetworkBase>(clone_impl());
        }
    }

    static std::shared_ptr<ConditionalBayesianNetworkBase>& keep_python_alive(
        std::shared_ptr<ConditionalBayesianNetworkBase>& m) {
        if (m && m->is_python_derived()) {
            auto o = py::cast(m);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<ConditionalBayesianNetworkBase*>();
            m = std::shared_ptr<ConditionalBayesianNetworkBase>(keep_python_state_alive, ptr);
        }

        return m;
    }

    static std::shared_ptr<ConditionalBayesianNetworkBase> keep_python_alive(
        const std::shared_ptr<ConditionalBayesianNetworkBase>& m) {
        if (m && m->is_python_derived()) {
            auto o = py::cast(m);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<ConditionalBayesianNetworkBase*>();
            return std::shared_ptr<ConditionalBayesianNetworkBase>(keep_python_state_alive, ptr);
        }

        return m;
    }

private:
    virtual ConditionalBayesianNetworkBase* clone_impl() const = 0;
};

class BayesianNetworkType {
public:
    virtual ~BayesianNetworkType() {}

    virtual bool is_python_derived() const { return false; }

    virtual std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const = 0;
    virtual std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const = 0;

    static std::shared_ptr<BayesianNetworkType>& keep_python_alive(std::shared_ptr<BayesianNetworkType>& s) {
        if (s && s->is_python_derived()) {
            auto o = py::cast(s);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<BayesianNetworkType*>();
            s = std::shared_ptr<BayesianNetworkType>(keep_python_state_alive, ptr);
        }

        return s;
    }

    static std::shared_ptr<BayesianNetworkType> keep_python_alive(const std::shared_ptr<BayesianNetworkType>& s) {
        if (s && s->is_python_derived()) {
            auto o = py::cast(s);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<BayesianNetworkType*>();
            return std::shared_ptr<BayesianNetworkType>(keep_python_state_alive, ptr);
        }

        return s;
    }

    virtual bool is_homogeneous() const = 0;

    virtual std::shared_ptr<FactorType> default_node_type() const = 0;
    virtual std::vector<std::shared_ptr<FactorType>> data_default_node_type(
        const std::shared_ptr<DataType>& dt) const = 0;

    virtual bool compatible_node_type(const BayesianNetworkBase&,
                                      const std::string&,
                                      const std::shared_ptr<FactorType>&) const {
        return true;
    }
    virtual bool compatible_node_type(const ConditionalBayesianNetworkBase&,
                                      const std::string&,
                                      const std::shared_ptr<FactorType>&) const {
        return true;
    }

    virtual bool can_have_arc(const BayesianNetworkBase&, const std::string&, const std::string&) const { return true; }
    virtual bool can_have_arc(const ConditionalBayesianNetworkBase&, const std::string&, const std::string&) const {
        return true;
    }

    virtual std::vector<std::shared_ptr<FactorType>> alternative_node_type(const BayesianNetworkBase&,
                                                                           const std::string&) const {
        return std::vector<std::shared_ptr<FactorType>>();
    }

    virtual std::vector<std::shared_ptr<FactorType>> alternative_node_type(const ConditionalBayesianNetworkBase&,
                                                                           const std::string&) const {
        return std::vector<std::shared_ptr<FactorType>>();
    }

    virtual bool operator==(const BayesianNetworkType& other) const { return this->hash() == other.hash(); }
    virtual bool operator!=(const BayesianNetworkType& o) const { return !(*this == o); }
    virtual bool operator==(BayesianNetworkType&& other) const { return this->hash() == other.hash(); }
    virtual bool operator!=(BayesianNetworkType&& o) const { return !(*this == o); }

    virtual std::string ToString() const = 0;

    virtual std::size_t hash() const { return m_hash; }

    virtual py::tuple __getstate__() const = 0;

protected:
    mutable std::uintptr_t m_hash;
};

void requires_continuous_data(const DataFrame& df);
void requires_discrete_data(const DataFrame& df);

template <typename DagType>
class BNGeneric;

using BayesianNetwork = BNGeneric<Dag>;
class ConditionalBayesianNetwork;

template <typename DagType>
class BNGeneric : public clone_inherit_condition<graph::is_unconditional_graph_v<DagType>,
                                                 BNGeneric<DagType>,
                                                 std::conditional_t<graph::is_unconditional_graph_v<DagType>,
                                                                    BayesianNetworkBase,
                                                                    ConditionalBayesianNetworkBase>> {
private:
    void initialize_no_types() {
        if (m_type == nullptr) throw std::runtime_error("Type of Bayesian network must be non-null.");

        if (!m_type->is_homogeneous()) {
            m_node_types.resize(g.num_raw_nodes());
            std::fill(m_node_types.begin(), m_node_types.end(), UnknownFactorType::get());
        }
    }

    void initialize_types(const FactorTypeVector& node_types) {
        if (m_type == nullptr) throw std::runtime_error("Type of Bayesian network must be non-null.");

        if (m_type->is_homogeneous()) {
            for (const auto& p : node_types) {
                // Check also null
                if (*p.second != *m_type->default_node_type())
                    throw std::invalid_argument("Wrong factor type \"" + p.second->ToString() + "\" for node \"" +
                                                p.first + "\" in Bayesian network type \"" + m_type->ToString() +
                                                "\".");
            }
        } else {
            m_node_types.resize(g.num_raw_nodes());
            std::fill(m_node_types.begin(), m_node_types.end(), UnknownFactorType::get());

            for (const auto& p : node_types) {
                auto index = check_index(p.first);
                m_node_types[index] = p.second;
            }

            for (const auto& p : node_types) {
                if (!m_type->compatible_node_type(*this, p.first, p.second)) {
                    throw std::invalid_argument("Node type " + p.second->ToString() +
                                                " not compatible with"
                                                " Bayesian network " +
                                                m_type->ToString());
                }
            }
        }
    }

public:
    using BaseClass = std::
        conditional_t<graph::is_unconditional_graph_v<DagType>, BayesianNetworkBase, ConditionalBayesianNetworkBase>;
    using DagClass = DagType;

    // /////////////////////////////////////
    // Unconditional BN constructors
    // /////////////////////////////////////
    template <typename D = DagType, graph::enable_if_unconditional_graph_t<D, int> = 0>
    BNGeneric(std::shared_ptr<BayesianNetworkType> type, const std::vector<std::string>& nodes)
        : g(nodes), m_type(type), m_cpds(), m_node_types() {
        initialize_no_types();
    }

    template <typename D = DagType, graph::enable_if_unconditional_graph_t<D, int> = 0>
    BNGeneric(std::shared_ptr<BayesianNetworkType> type,
              const std::vector<std::string>& nodes,
              const FactorTypeVector& node_types)
        : g(nodes), m_type(type), m_cpds(), m_node_types() {
        initialize_types(node_types);
    }

    template <typename D = DagType, graph::enable_if_unconditional_graph_t<D, int> = 0>
    BNGeneric(std::shared_ptr<BayesianNetworkType> type, const ArcStringVector& arcs)
        : g(arcs), m_type(type), m_cpds(), m_node_types() {
        initialize_no_types();
    }

    template <typename D = DagType, graph::enable_if_unconditional_graph_t<D, int> = 0>
    BNGeneric(std::shared_ptr<BayesianNetworkType> type,
              const ArcStringVector& arcs,
              const FactorTypeVector& node_types)
        : g(arcs), m_type(type), m_cpds(), m_node_types() {
        initialize_types(node_types);
    }

    template <typename D = DagType, graph::enable_if_unconditional_graph_t<D, int> = 0>
    BNGeneric(std::shared_ptr<BayesianNetworkType> type,
              const std::vector<std::string>& nodes,
              const ArcStringVector& arcs)
        : g(nodes, arcs), m_type(type), m_cpds(), m_node_types() {
        initialize_no_types();
    }

    template <typename D = DagType, graph::enable_if_unconditional_graph_t<D, int> = 0>
    BNGeneric(std::shared_ptr<BayesianNetworkType> type,
              const std::vector<std::string>& nodes,
              const ArcStringVector& arcs,
              const FactorTypeVector& node_types)
        : g(nodes, arcs), m_type(type), m_cpds(), m_node_types() {
        initialize_types(node_types);
    }

    // /////////////////////////////////////
    // Conditional BN constructors
    // /////////////////////////////////////
    template <typename D = DagType, graph::enable_if_conditional_graph_t<D, int> = 0>
    BNGeneric(std::shared_ptr<BayesianNetworkType> type,
              const std::vector<std::string>& nodes,
              const std::vector<std::string>& interface_nodes)
        : g(nodes, interface_nodes), m_type(type), m_cpds(), m_node_types() {
        initialize_no_types();
    }

    template <typename D = DagType, graph::enable_if_conditional_graph_t<D, int> = 0>
    BNGeneric(std::shared_ptr<BayesianNetworkType> type,
              const std::vector<std::string>& nodes,
              const std::vector<std::string>& interface_nodes,
              const FactorTypeVector& node_types)
        : g(nodes, interface_nodes), m_type(type), m_cpds(), m_node_types() {
        initialize_types(node_types);
    }

    template <typename D = DagType, graph::enable_if_conditional_graph_t<D, int> = 0>
    BNGeneric(std::shared_ptr<BayesianNetworkType> type,
              const std::vector<std::string>& nodes,
              const std::vector<std::string>& interface_nodes,
              const ArcStringVector& arcs)
        : g(nodes, interface_nodes, arcs), m_type(type), m_cpds(), m_node_types() {
        initialize_no_types();
    }

    template <typename D = DagType, graph::enable_if_conditional_graph_t<D, int> = 0>
    BNGeneric(std::shared_ptr<BayesianNetworkType> type,
              const std::vector<std::string>& nodes,
              const std::vector<std::string>& interface_nodes,
              const ArcStringVector& arcs,
              const FactorTypeVector& node_types)
        : g(nodes, interface_nodes, arcs), m_type(type), m_cpds(), m_node_types() {
        initialize_types(node_types);
    }

    BNGeneric(std::shared_ptr<BayesianNetworkType> type, const DagType& graph) : g(graph), m_type(type), m_cpds() {
        initialize_no_types();
    }

    BNGeneric(std::shared_ptr<BayesianNetworkType> type, const DagType& graph, const FactorTypeVector& node_types)
        : g(graph), m_type(type), m_cpds(), m_node_types() {
        initialize_types(node_types);
    }

    BNGeneric(std::shared_ptr<BayesianNetworkType> type, DagType&& graph)
        : g(std::move(graph)), m_type(type), m_cpds(), m_node_types() {
        initialize_no_types();
    }

    BNGeneric(std::shared_ptr<BayesianNetworkType> type, DagType&& graph, const FactorTypeVector& node_types)
        : g(std::move(graph)), m_type(type), m_cpds(), m_node_types() {
        initialize_types(node_types);
    }

    const DagType& graph() const override { return g; }
    DagType& graph() override { return g; }

    int num_nodes() const override { return g.num_nodes(); }

    int num_raw_nodes() const { return g.num_raw_nodes(); }

    int num_arcs() const override { return g.num_arcs(); }

    const std::vector<std::string>& nodes() const override { return g.nodes(); }

    ArcStringVector arcs() const override { return g.arcs(); }

    const std::unordered_map<std::string, int>& indices() const override { return g.indices(); }

    int index(const std::string& node) const override { return g.index(node); }

    int check_index(int idx) const { return g.check_index(idx); }

    int check_index(const std::string& name) const { return g.check_index(name); }

    int collapsed_index(const std::string& node) const override { return g.collapsed_index(node); }

    int index_from_collapsed(int collapsed_index) const override { return g.index_from_collapsed(collapsed_index); }

    int collapsed_from_index(int index) const override { return g.collapsed_from_index(index); }

    const std::unordered_map<std::string, int>& collapsed_indices() const override { return g.collapsed_indices(); }

    bool is_valid(const std::string& name) const override { return g.is_valid(g.index(name)); }

    bool contains_node(const std::string& name) const override { return g.contains_node(name); }

    int add_node(const std::string& node) override {
        int idx = g.add_node(node);

        if (idx == (g.num_raw_nodes() - 1)) {
            if (!m_cpds.empty()) m_cpds.resize(idx + 1);
            if (!m_type->is_homogeneous()) {
                m_node_types.resize(idx + 1);
                m_node_types[idx] = UnknownFactorType::get();
            }
        }

        return idx;
    }

    void remove_node(const std::string& node) override {
        if (!m_cpds.empty()) {
            m_cpds[g.index(node)] = nullptr;
        }

        if (!m_type->is_homogeneous()) {
            m_node_types[g.index(node)] = UnknownFactorType::get();
        }

        g.remove_node(node);
    }

    const std::string& name(int node_index) const override { return g.name(node_index); }

    const std::string& collapsed_name(int collapsed_index) const override { return g.collapsed_name(collapsed_index); }

    int num_parents(const std::string& node) const override { return g.num_parents(node); }

    int num_children(const std::string& node) const override { return g.num_children(node); }

    std::vector<std::string> parents(const std::string& node) const override { return g.parents(node); }

    std::vector<std::string> children(const std::string& node) const override { return g.children(node); }

    bool has_arc(const std::string& source, const std::string& target) const override {
        return g.has_arc(source, target);
    }

    bool has_path(const std::string& source, const std::string& target) const override {
        return g.has_path(source, target);
    }

    void add_arc(const std::string& source, const std::string& target) override {
        if (can_add_arc(source, target)) {
            add_arc_unsafe(source, target);
        } else {
            throw std::invalid_argument("Cannot add arc " + source + " -> " + target + ".");
        }
    }

    void add_arc_unsafe(const std::string& source, const std::string& target) override { g.add_arc(source, target); }

    void remove_arc(const std::string& source, const std::string& target) override { g.remove_arc(source, target); }

    void flip_arc(const std::string& source, const std::string& target) override {
        if (can_flip_arc(source, target)) {
            flip_arc_unsafe(source, target);
        } else {
            throw std::invalid_argument("Cannot flip arc " + source + " -> " + target + ".");
        }
    }

    void flip_arc_unsafe(const std::string& source, const std::string& target) override { g.flip_arc(source, target); }

    bool can_add_arc(const std::string& source, const std::string& target) const override {
        return g.can_add_arc(source, target) && m_type->can_have_arc(*this, source, target);
    }

    bool can_flip_arc(const std::string& source, const std::string& target) const override {
        return g.can_flip_arc(source, target) && m_type->can_have_arc(*this, target, source);
    }

    void force_whitelist(const ArcStringVector& arc_whitelist) override {
        for (const auto& arc : arc_whitelist) {
            if (!has_arc(arc.first, arc.second)) {
                if (has_arc(arc.second, arc.first)) {
                    throw std::invalid_argument("Arc " + arc.first + " -> " + arc.second +
                                                " in whitelist,"
                                                " but arc " +
                                                arc.second + " -> " + arc.first +
                                                " is present"
                                                " in the Bayesian Network.");
                } else if (can_add_arc(arc.first, arc.second)) {
                    add_arc_unsafe(arc.first, arc.second);
                } else {
                    throw std::invalid_argument("Arc " + arc.first + " -> " + arc.second +
                                                " not allowed in this Bayesian network.");
                }
            }
        }

        g.topological_sort();
    }

    virtual bool can_have_cpd(const std::string& name) const { return is_valid(name); }

    bool fitted() const override;

    std::shared_ptr<Factor> cpd(const std::string& node) override {
        auto idx = check_index(node);
        if (!m_cpds.empty() && m_cpds[idx])
            return m_cpds[idx];
        else
            throw std::invalid_argument("CPD of variable \"" + node +
                                        "\" not added. Call add_cpds() or fit() to add the CPD.");
    }

    const std::shared_ptr<Factor> cpd(const std::string& node) const override {
        auto idx = check_index(node);
        if (!m_cpds.empty() && m_cpds[idx])
            return m_cpds[idx];
        else
            throw std::invalid_argument("CPD of variable \"" + node +
                                        "\" not added. Call add_cpds() or fit() to add the CPD.");
    }

    virtual void check_compatible_cpd(const Factor& cpd) const;
    virtual bool must_construct_cpd(const Factor& cpd,
                                    const FactorType& model_node_type,
                                    const std::vector<std::string>& model_parents) const;
    void add_cpds(const std::vector<std::shared_ptr<Factor>>& cpds) override;
    void fit(const DataFrame& df, const Arguments& construction_args = Arguments()) override;
    VectorXd logl(const DataFrame& df) const override;
    double slogl(const DataFrame& df) const override;

    std::shared_ptr<BayesianNetworkType> type() const override { return m_type; }
    BayesianNetworkType& type_ref() const override { return *m_type; }

    using BaseClass::sample;
    DataFrame sample(int n, unsigned int seed = std::random_device{}(), bool ordered = false) const override;

    std::shared_ptr<ConditionalBayesianNetworkBase> conditional_bn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override;
    std::shared_ptr<ConditionalBayesianNetworkBase> conditional_bn() const override;
    std::shared_ptr<BayesianNetworkBase> unconditional_bn() const override;

    void save(std::string name, bool include_cpd = false) const override;
    virtual py::tuple __getstate__() const override;

    template <typename DerivedBN>
    friend std::shared_ptr<DerivedBN> __derived_bn_setstate__(py::tuple& t);

    bool include_cpd() const override { return m_include_cpd; }

    void set_include_cpd(bool include_cpd) const override { m_include_cpd = include_cpd; }

    std::shared_ptr<FactorType> node_type(const std::string& node) const override {
        if (m_type->is_homogeneous()) {
            return m_type->default_node_type();
        } else {
            auto node_index = check_index(node);
            return m_node_types[node_index];
        }
    }

    std::shared_ptr<FactorType> underlying_node_type(const DataFrame& df, const std::string& node) const override {
        if (m_type->is_homogeneous()) {
            return m_type->default_node_type();
        } else {
            auto node_index = check_index(node);

            if (*m_node_types[node_index] != UnknownFactorType::get_ref()) {
                return m_node_types[node_index];
            } else {
                auto nt = m_type->data_default_node_type(df.col(node)->type());
                if (nt.empty()) {
                    throw std::invalid_argument("There is no underlying FactorType for node " + node +
                                                " as there is no valid FactorType for DataType " +
                                                df.col(node)->type()->ToString());
                } else {
                    return nt[0];
                }
            }
        }
    }

    std::unordered_map<std::string, std::shared_ptr<FactorType>> node_types() const override {
        std::unordered_map<std::string, std::shared_ptr<FactorType>> res;

        if (m_type->is_homogeneous()) {
            auto type = m_type->default_node_type();
            for (const auto& n : this->nodes()) {
                res.insert({n, type});
            }
        } else {
            for (const auto& n : this->nodes()) {
                auto type = m_node_types[index(n)];
                res.insert({n, type});
            }
        }

        return res;
    }

    void set_node_type(const std::string& node, const std::shared_ptr<FactorType>& new_type) override {
        if (m_type->is_homogeneous()) {
            if (*new_type != *m_type->default_node_type())
                throw std::invalid_argument("Wrong factor type \"" + new_type->ToString() + "\" for node \"" + node +
                                            "\" in Bayesian network type \"" + m_type->ToString() + "\".");
        } else {
            if (*new_type != UnknownFactorType::get_ref() && !m_type->compatible_node_type(*this, node, new_type)) {
                throw std::invalid_argument("Wrong factor type \"" + new_type->ToString() + "\" for node \"" + node +
                                            "\" in Bayesian network type \"" + m_type->ToString() + "\".");
            }

            auto node_index = check_index(node);
            m_node_types[node_index] = new_type;

            if (!m_cpds.empty() && m_cpds[node_index] && *m_node_types[node_index] != m_cpds[node_index]->type_ref())
                m_cpds[node_index] = nullptr;
        }
    }

    void set_unknown_node_types(const DataFrame& df,
                                const FactorTypeVector& type_blacklist = FactorTypeVector()) override {
        if (m_type->is_homogeneous()) return;

        util::FactorTypeSet blacklist_set{type_blacklist.begin(), type_blacklist.end()};

        FactorTypeVector new_types;
        for (const auto& nn : nodes()) {
            if (*node_type(nn) == UnknownFactorType::get_ref()) {
                auto default_node_types = m_type->data_default_node_type(df.col(nn)->type());

                bool found_new_type = false;
                for (const auto& type : default_node_types) {
                    auto new_pair = std::make_pair(nn, type);
                    if (blacklist_set.count(new_pair) == 0) {
                        new_types.push_back(std::move(new_pair));
                        found_new_type = true;
                        break;
                    }
                }

                if (!found_new_type) {
                    throw std::invalid_argument("A valid FactorType for node " + nn + " could not be inferred.");
                }
            }
        }

        force_type_whitelist(new_types);
    }

    bool has_unknown_node_types() const override {
        if (m_type->is_homogeneous()) return false;

        for (const auto& nn : nodes()) {
            auto node_index = check_index(nn);
            if (*m_node_types[node_index] == UnknownFactorType::get_ref()) return true;
        }

        return false;
    }

    void force_type_whitelist(const FactorTypeVector& type_whitelist) override {
        if (m_type->is_homogeneous()) {
            for (const auto& p : type_whitelist) {
                // Check also null
                if (*p.second != *m_type->default_node_type())
                    throw std::invalid_argument("Wrong factor type \"" + p.second->ToString() + "\" for node \"" +
                                                p.first + "\" in Bayesian network type \"" + m_type->ToString() +
                                                "\".");
            }
        } else {
            FactorTypeVector old_data;
            for (const auto& p : type_whitelist) {
                auto index = check_index(p.first);
                old_data.push_back({p.first, m_node_types[index]});
                m_node_types[index] = p.second;
            }

            for (const auto& p : type_whitelist) {
                // Check also null
                if (*p.second != UnknownFactorType::get_ref() &&
                    !m_type->compatible_node_type(*this, p.first, p.second)) {
                    for (const auto& old : old_data) {
                        m_node_types[index(old.first)] = old.second;
                    }

                    throw std::invalid_argument("Wrong factor type \"" + p.second->ToString() + "\" for node \"" +
                                                p.first + "\" in Bayesian network type \"" + m_type->ToString() +
                                                "\".");
                }
            }

            if (!m_cpds.empty()) {
                for (const auto& p : type_whitelist) {
                    auto node_index = index(p.first);
                    if (m_cpds[node_index] && *m_node_types[node_index] != m_cpds[node_index]->type_ref()) {
                        m_cpds[node_index] = nullptr;
                    }
                }
            }
        }
    }

    std::string ToString() const override {
        if constexpr (graph::is_unconditional_graph_v<DagType>)
            return "BayesianNetwork[" + m_type->ToString() + "]";
        else if constexpr (graph::is_conditional_graph_v<DagType>)
            return "ConditionalBayesianNetwork[" + m_type->ToString() + "]";
        else
            static_assert(util::always_false<DagType>, "Wrong data type for BNGeneric");
    }

protected:
    void check_fitted() const;
    DagType g;
    std::shared_ptr<BayesianNetworkType> m_type;
    std::vector<std::shared_ptr<Factor>> m_cpds;
    std::vector<std::shared_ptr<FactorType>> m_node_types;
    // This is necessary because __getstate__() do not admit parameters.
    mutable bool m_include_cpd;
};

template <typename DagType>
bool BNGeneric<DagType>::fitted() const {
    if (m_cpds.empty()) {
        return false;
    } else {
        for (const auto& nn : nodes()) {
            auto i = check_index(nn);
            if (!m_cpds[i] || !m_cpds[i]->fitted() ||
                (!m_type->is_homogeneous() && *m_cpds[i]->type() != *m_node_types[i])) {
                return false;
            }
        }

        return true;
    }
}

template <typename DagType>
void BNGeneric<DagType>::check_fitted() const {
    if (m_cpds.empty()) {
        throw py::value_error("Model not fitted.");
    } else {
        bool all_fitted = true;
        std::string err;
        for (const auto& nn : nodes()) {
            auto i = check_index(nn);
            if (!m_cpds[i] || !m_cpds[i]->fitted() ||
                (!m_type->is_homogeneous() && m_cpds[i]->type_ref() != *m_node_types[i])) {
                if (all_fitted) {
                    err += "Some CPDs are not fitted:\n";
                    all_fitted = false;
                }
                err += m_cpds[i]->ToString() + "\n";
            }
        }

        if (!all_fitted) throw py::value_error(err);
    }
}

template <typename DagType>
void BNGeneric<DagType>::check_compatible_cpd(const Factor& cpd) const {
    if (!contains_node(cpd.variable())) {
        throw std::invalid_argument("CPD defined on variable which is not present in the model:\n" + cpd.ToString());
    }

    const auto& evidence = cpd.evidence();

    for (const auto& ev : evidence) {
        if constexpr (graph::is_unconditional_graph_v<DagType>) {
            if (!contains_node(ev)) {
                throw std::invalid_argument("Evidence variable " + ev + " is not present in the model:\n" +
                                            cpd.ToString());
            }
        } else if constexpr (graph::is_conditional_graph_v<DagType>) {
            if (!this->contains_joint_node(ev)) {
                throw std::invalid_argument("Evidence variable " + ev + " is not present in the model:\n" +
                                            cpd.ToString());
            }
        } else {
            static_assert(util::always_false<DagType>, "Wrong BN Type");
        }
    }

    auto pa = parents(cpd.variable());
    if (pa.size() != evidence.size()) {
        std::string err = "CPD do not have the model's parent set as evidence:\n" + cpd.ToString() +
                          "\nParents: " + g.parents_to_string(cpd.variable());

        throw std::invalid_argument(err);
    }

    std::unordered_set<std::string> evidence_set(evidence.begin(), evidence.end());
    for (const auto& parent : pa) {
        if (evidence_set.find(parent) == evidence_set.end()) {
            std::string err = "CPD do not have the model's parent set as evidence:\n" + cpd.ToString() +
                              "\nParents: " + g.parents_to_string(cpd.variable());
            throw std::invalid_argument(err);
        }
    }

    auto node_type_ = node_type(cpd.variable());
    auto cpd_type = cpd.type();
    if (*node_type_ != UnknownFactorType::get_ref() && *cpd_type != *node_type_) {
        throw std::invalid_argument("Factor " + cpd.ToString() + " is of type " + cpd_type->ToString() +
                                    "."
                                    " Bayesian network expects type " +
                                    node_type_->ToString());
    }
}

template <typename DagType>
void BNGeneric<DagType>::add_cpds(const std::vector<std::shared_ptr<Factor>>& cpds) {
    for (const auto& cpd : cpds) {
        check_compatible_cpd(*cpd);
    }

    FactorTypeVector new_factor_types;
    for (const auto& cpd : cpds) {
        if (*node_type(cpd->variable()) == UnknownFactorType::get_ref()) {
            new_factor_types.push_back({cpd->variable(), cpd->type()});
        }
    }

    force_type_whitelist(new_factor_types);

    if (m_cpds.empty()) {
        m_cpds.resize(num_raw_nodes());
    }

    for (const auto& cpd : cpds) {
        if (can_have_cpd(cpd->variable())) {
            auto idx = index(cpd->variable());
            m_cpds[idx] = cpd;
        } else {
            throw std::invalid_argument("CPD for node " + cpd->variable() + " not valid for Bayesian network.");
        }
    }
}

template <typename DagType>
bool BNGeneric<DagType>::must_construct_cpd(const Factor& cpd,
                                            const FactorType& model_node_type,
                                            const std::vector<std::string>& model_parents) const {
    const auto& cpd_evidence = cpd.evidence();

    if (cpd.type_ref() != model_node_type) return true;

    if (cpd_evidence.size() != model_parents.size()) return true;

    if (std::is_permutation(cpd_evidence.begin(), cpd_evidence.end(), model_parents.begin(), model_parents.end())) {
        return false;
    } else {
        return true;
    }
}

template <typename DagType>
void BNGeneric<DagType>::fit(const DataFrame& df, const Arguments& construction_args) {
    if (m_cpds.empty()) {
        m_cpds.resize(num_raw_nodes());
    }

    FactorTypeVector new_factor_types;
    if (!m_type->is_homogeneous()) {
        for (const auto& nn : nodes()) {
            auto node_type_ = node_type(nn);

            if (*node_type_ == UnknownFactorType::get_ref()) {
                new_factor_types.push_back({nn, underlying_node_type(df, nn)});
            }
        }
    }

    force_type_whitelist(new_factor_types);

    for (const auto& nn : nodes()) {
        auto i = check_index(nn);

        auto p = parents(nn);
        df.raise_has_columns(nn, p);

        auto node_type_ = node_type(nn);

        if (!m_cpds[i] || must_construct_cpd(*m_cpds[i], *node_type_, p)) {
            auto [args, kwargs] = construction_args.args(nn, node_type_);
            m_cpds[i] = node_type_->new_factor(*this, nn, p, args, kwargs);
            m_cpds[i]->fit(df);
        } else if (!m_cpds[i]->fitted()) {
            m_cpds[i]->fit(df);
        }
    }
}

template <typename DagType>
VectorXd BNGeneric<DagType>::logl(const DataFrame& df) const {
    check_fitted();

    const auto& nn = nodes();
    VectorXd accum = m_cpds[index(nn[0])]->logl(df);

    for (int i = 1, i_end = nn.size(); i < i_end; ++i) {
        accum += m_cpds[index(nn[i])]->logl(df);
    }

    return accum;
}

template <typename DagType>
double BNGeneric<DagType>::slogl(const DataFrame& df) const {
    check_fitted();

    double accum = 0;

    for (const auto& nn : nodes()) {
        auto i = index(nn);
        accum += m_cpds[i]->slogl(df);
    }

    return accum;
}

template <typename DagType>
DataFrame BNGeneric<DagType>::sample(int n, unsigned int seed, bool ordered) const {
    if (n < 0) {
        throw std::invalid_argument("n should be a non-negative number");
    }

    if constexpr (graph::is_conditional_graph_v<DagType>) {
        if (this->num_interface_nodes() > 0)
            throw std::runtime_error(
                "ConditionalBayesianNetwork cannot sample "
                "if evidence is not provided for the interface nodes.");
    }

    check_fitted();

    DataFrame parents(n);

    auto top_sort = g.topological_sort();
    for (size_t i = 0; i < top_sort.size(); ++i) {
        auto idx = index(top_sort[i]);
        auto array = m_cpds[idx]->sample(n, parents, seed + i);

        auto res = parents->AddColumn(i, top_sort[i], array);
        parents = DataFrame(std::move(res).ValueOrDie());
    }

    if (ordered) {
        std::vector<Field_ptr> fields;
        std::vector<Array_ptr> columns;

        auto schema = parents->schema();
        for (auto& name : nodes()) {
            fields.push_back(schema->GetFieldByName(name));
            columns.push_back(parents->GetColumnByName(name));
        }

        auto new_schema = std::make_shared<arrow::Schema>(fields);
        auto new_rb = arrow::RecordBatch::Make(new_schema, n, columns);
        return DataFrame(new_rb);
    } else {
        return parents;
    }
}

template <typename DagType>
std::shared_ptr<ConditionalBayesianNetworkBase> BNGeneric<DagType>::conditional_bn(
    const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const {
    auto new_dag = g.conditional_graph(nodes, interface_nodes);

    auto new_bn = std::make_shared<ConditionalBayesianNetwork>(m_type, std::move(new_dag));

    if (!m_cpds.empty()) {
        std::vector<std::shared_ptr<Factor>> cpds;
        cpds.reserve(nodes.size());

        for (const auto& name : new_bn->nodes()) {
            if (can_have_cpd(name)) {
                cpds.push_back(cpd(name));
            }
        }

        new_bn->add_cpds(cpds);
    }

    return new_bn;
}

template <typename DagType>
std::shared_ptr<ConditionalBayesianNetworkBase> BNGeneric<DagType>::conditional_bn() const {
    if constexpr (graph::is_conditional_graph_v<DagType>) {
        return this->clone();
    } else {
        std::vector<std::string> interface;
        return conditional_bn(this->nodes(), interface);
    }
}

template <typename DagType>
std::shared_ptr<BayesianNetworkBase> BNGeneric<DagType>::unconditional_bn() const {
    if constexpr (graph::is_unconditional_graph_v<DagType>) {
        return this->clone();
    } else {
        auto new_dag = g.unconditional_graph();

        auto new_bn = std::make_shared<BNGeneric<Dag>>(m_type, std::move(new_dag));

        if (!m_cpds.empty()) {
            std::vector<std::shared_ptr<Factor>> cpds;
            cpds.reserve(num_nodes());

            for (const auto& nn : nodes()) {
                auto i = index(nn);
                cpds.push_back(m_cpds[i]);
            }

            new_bn->add_cpds(cpds);
        }

        return new_bn;
    }
}

template <typename DagType>
void BNGeneric<DagType>::save(std::string name, bool include_cpd) const {
    m_include_cpd = include_cpd;
    auto open = py::module_::import("io").attr("open");

    if (name.size() < 7 || name.substr(name.size() - 7) != ".pickle") name += ".pickle";

    auto file = open(name, "wb");
    py::module_::import("pickle").attr("dump")(py::cast(this), file, 2);
    file.attr("close")();
}

template <typename DagType>
py::tuple BNGeneric<DagType>::__getstate__() const {
    std::vector<std::shared_ptr<Factor>> cpds;

    if (m_include_cpd && !m_cpds.empty()) {
        for (const auto& nn : nodes()) {
            auto i = index(nn);
            if (m_cpds[i]) {
                try {
                    check_compatible_cpd(*m_cpds[i]);
                    cpds.push_back(m_cpds[i]);
                } catch (std::exception&) {
                }
            }
        }
    }

    FactorTypeVector node_types;
    if (!m_type->is_homogeneous()) {
        node_types.reserve(g.num_nodes());

        for (const auto& nn : nodes()) {
            auto i = index(nn);
            if (*m_node_types[i] != UnknownFactorType::get_ref())
                node_types.push_back(std::make_pair(nn, m_node_types[i]));
        }
    }

    return py::make_tuple(g, m_type, node_types, m_include_cpd, cpds);
}

template <typename BNType>
void __nonderived_bn_setstate__(py::object& self, py::tuple& t) {
    using DagType = typename BNType::DagClass;
    if (t.size() != 5) throw std::runtime_error("Not valid BayesianNetwork.");

    auto pybntype = py::type::of<BNType>();

    auto dag = t[0].cast<DagType>();
    auto type = t[1].cast<std::shared_ptr<BayesianNetworkType>>();

    // Initialize the C++ side:
    // The Python constructor ensures that the python objects are kept alive.
    if (type->is_homogeneous()) {
        pybntype.attr("__init__")(self, type, std::move(dag));
    } else {
        auto node_types = t[2].cast<FactorTypeVector>();
        if (node_types.empty()) pybntype.attr("__init__")(self, type, std::move(dag));

        // BNType is BayesianNetwork or ConditionalBayesianNetwork, so this constructor always exists.
        pybntype.attr("__init__")(self, type, std::move(dag), node_types);
    }

    auto cpp_self = self.cast<std::shared_ptr<BNType>>();

    if (t[3].cast<bool>()) {
        auto cpds = t[4].cast<std::vector<std::shared_ptr<Factor>>>();
        Factor::keep_vector_python_alive(cpds);
        cpp_self->add_cpds(cpds);
    }
}

template <typename DerivedBN>
std::shared_ptr<DerivedBN> __derived_bn_setstate__(py::tuple& t) {
    using DagType = typename DerivedBN::DagClass;
    if (t.size() != 5) throw std::runtime_error("Not valid BayesianNetwork.");

    auto dag = t[0].cast<DagType>();
    auto type = t[1].cast<std::shared_ptr<BayesianNetworkType>>();

    std::shared_ptr<DerivedBN> bn = [&t, &type, &dag]() {
        if (type->is_homogeneous()) {
            return std::make_shared<DerivedBN>(std::move(dag));
        } else {
            // Only C++ FactorType are allowed in these Bayesian networks, so no need to keep Python alive.
            auto node_types = t[2].cast<FactorTypeVector>();
            if (node_types.empty()) return std::make_shared<DerivedBN>(std::move(dag));

            if constexpr (std::is_constructible_v<DerivedBN, DagType&&, FactorTypeVector>) {
                return std::make_shared<DerivedBN>(std::move(dag), node_types);
            } else {
                auto m = std::make_shared<DerivedBN>(std::move(dag));
                m->initialize_types(node_types);
                return m;
            }
        }
    }();

    if (t[3].cast<bool>()) {
        auto cpds = t[4].cast<std::vector<std::shared_ptr<Factor>>>();
        // Keep Python factors alive. Only needed because someone could create a Python Factor which
        // implements a C++-implemented FactorType.
        Factor::keep_vector_python_alive(cpds);
        bn->add_cpds(cpds);
    }

    return bn;
}

class ConditionalBayesianNetwork : public clone_inherit<ConditionalBayesianNetwork, BNGeneric<ConditionalDag>> {
public:
    using clone_inherit<ConditionalBayesianNetwork, BNGeneric<ConditionalDag>>::clone_inherit;

    int num_interface_nodes() const override { return this->g.num_interface_nodes(); }

    int num_joint_nodes() const override { return this->g.num_joint_nodes(); }

    const std::vector<std::string>& interface_nodes() const override { return this->g.interface_nodes(); }

    const std::vector<std::string>& joint_nodes() const override { return this->g.joint_nodes(); }

    ArcStringVector interface_arcs() const override { return this->g.interface_arcs(); }

    int interface_collapsed_index(const std::string& name) const override {
        return this->g.interface_collapsed_index(name);
    }

    int joint_collapsed_index(const std::string& name) const override { return this->g.joint_collapsed_index(name); }

    const std::unordered_map<std::string, int>& interface_collapsed_indices() const override {
        return this->g.interface_collapsed_indices();
    }

    const std::unordered_map<std::string, int>& joint_collapsed_indices() const override {
        return this->g.joint_collapsed_indices();
    }

    int index_from_interface_collapsed(int interface_collapsed_index) const override {
        return this->g.index_from_interface_collapsed(interface_collapsed_index);
    }

    int index_from_joint_collapsed(int joint_collapsed_index) const override {
        return this->g.index_from_joint_collapsed(joint_collapsed_index);
    }

    int interface_collapsed_from_index(int index) const override {
        return this->g.interface_collapsed_from_index(index);
    }

    int joint_collapsed_from_index(int index) const override { return this->g.joint_collapsed_from_index(index); }

    const std::string& interface_collapsed_name(int interface_collapsed_index) const override {
        return this->g.interface_collapsed_name(interface_collapsed_index);
    }

    const std::string& joint_collapsed_name(int joint_collapsed_index) const override {
        return this->g.joint_collapsed_name(joint_collapsed_index);
    }

    bool contains_interface_node(const std::string& name) const override {
        return this->g.contains_interface_node(name);
    }

    bool contains_joint_node(const std::string& name) const override { return this->g.contains_joint_node(name); }

    int add_interface_node(const std::string& node) override { return this->g.add_interface_node(node); }

    void remove_interface_node(const std::string& node) override { this->g.remove_interface_node(node); }

    bool is_interface(const std::string& name) const override { return this->g.is_interface(name); }

    void set_interface(const std::string& name) override {
        this->g.set_interface(name);
        if (!this->m_cpds.empty()) {
            this->m_cpds[this->index(name)] = nullptr;
        }
    }

    void set_node(const std::string& name) override {
        this->g.set_node(name);
        if (!this->m_cpds.empty()) this->m_cpds[this->index(name)] = nullptr;
    }

    bool can_have_cpd(const std::string& name) const override { return this->is_valid(name) && !is_interface(name); }

    using BNGeneric<ConditionalDag>::sample;
    DataFrame sample(const DataFrame& evidence,
                     unsigned int seed = std::random_device{}(),
                     bool concat_evidence = false,
                     bool ordered = false) const override;
};

}  // namespace models

#endif  // PYBNESIAN_MODELS_BAYESIANNETWORK_HPP
