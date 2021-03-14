#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <models/KDENetwork.hpp>
#include <models/DiscreteBN.hpp>
#include <util/util_types.hpp>

using models::BayesianNetworkType, models::GaussianNetworkType, models::SemiparametricBNType, models::KDENetworkType,
    models::DiscreteBNType, models::BayesianNetworkBase, models::BNGeneric, models::BayesianNetwork,
    models::GaussianNetwork, models::SemiparametricBN, models::KDENetwork, models::DiscreteBN;

using models::ConditionalBayesianNetworkBase, models::ConditionalBayesianNetwork, models::ConditionalGaussianNetwork,
    models::ConditionalSemiparametricBN, models::ConditionalKDENetwork, models::ConditionalDiscreteBN;

using models::DynamicBayesianNetworkBase, models::DynamicBayesianNetwork, models::DynamicGaussianNetwork,
    models::DynamicSemiparametricBN, models::DynamicDiscreteBN, models::DynamicKDENetwork;

using util::random_seed_arg;

class PyBayesianNetworkType : public BayesianNetworkType {
public:
    using BayesianNetworkType::BayesianNetworkType;

    PyBayesianNetworkType() { m_hash = reinterpret_cast<std::uintptr_t>(nullptr); }

    bool is_python_derived() const override { return true; }

    bool is_homogeneous() const override { PYBIND11_OVERRIDE_PURE(bool, BayesianNetworkType, is_homogeneous, ); }

    std::shared_ptr<FactorType> default_node_type() const override {
        py::gil_scoped_acquire gil;

        py::function override = py::get_override(static_cast<const BayesianNetworkType*>(this), "default_node_type");
        if (override) {
            auto o = override();
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<FactorType*>();

            return std::shared_ptr<FactorType>(keep_python_state_alive, ptr);
        }

        py::pybind11_fail("Tried to call pure virtual function \"BayesianNetworkType::default_node_type\"");
    }

    bool compatible_node_type(const BayesianNetworkBase& m, const std::string& variable) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "compatible_node_type");
        if (override) {
            auto o = override(m.shared_from_this(), variable);
            return o.cast<bool>();
        }

        return BayesianNetworkType::compatible_node_type(m, variable);
    }

    bool compatible_node_type(const ConditionalBayesianNetworkBase& m, const std::string& variable) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "compatible_node_type");
        if (override) {
            auto o = override(m.shared_from_this(), variable);
            return o.cast<bool>();
        }

        return BayesianNetworkType::compatible_node_type(m, variable);
    }

    bool can_add_arc(const BayesianNetworkBase& m,
                     const std::string& source,
                     const std::string& target) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "can_add_arc");
        if (override) {
            auto o = override(m.shared_from_this(), source, target);
            return o.cast<bool>();
        }

        return BayesianNetworkType::can_add_arc(m, source, target);
    }

    bool can_flip_arc(const BayesianNetworkBase& m,
                      const std::string& source,
                      const std::string& target) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "can_flip_arc");
        if (override) {
            auto o = override(m.shared_from_this(), source, target);
            return o.cast<bool>();
        }

        return BayesianNetworkType::can_flip_arc(m, source, target);
    }

    bool can_add_arc(const ConditionalBayesianNetworkBase& m,
                     const std::string& source,
                     const std::string& target) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "can_add_arc");
        if (override) {
            auto o = override(m.shared_from_this(), source, target);
            return o.cast<bool>();
        }

        return BayesianNetworkType::can_add_arc(m, source, target);
    }

    bool can_flip_arc(const ConditionalBayesianNetworkBase& m,
                      const std::string& source,
                      const std::string& target) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "can_add_arc");
        if (override) {
            auto o = override(m.shared_from_this(), source, target);
            return o.cast<bool>();
        }

        return BayesianNetworkType::can_add_arc(m, source, target);
    }

    std::string ToString() const override { PYBIND11_OVERRIDE_PURE(std::string, BayesianNetworkType, ToString, ); }

    std::size_t hash() const override {
        if (m_hash == reinterpret_cast<std::uintptr_t>(nullptr)) {
            py::object o = py::cast(this);
            py::handle ttype = o.get_type();

            // Get the pointer of the Python derived type class.
            // !!!! We have to do this here because in the constructor,
            // "this" is just a FactorType instead of the derived Python class !!!!!!!!!!!!!!!
            m_hash = reinterpret_cast<std::uintptr_t>(ttype.ptr());
        }

        return m_hash;
    }

    py::tuple __getstate__() const override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(static_cast<const BayesianNetworkType*>(this), "__getstate_extra__");
        if (override) {
            return py::make_tuple(true, override());
        } else {
            return py::make_tuple(false, py::make_tuple());
        }
    }

    static void __setstate__(py::object& self, py::tuple& t) {
        // Call trampoline constructor
        py::gil_scoped_acquire gil;
        auto bntype = py::type::of<BayesianNetworkType>();
        bntype.attr("__init__")(self);

        auto ptr = self.cast<const BayesianNetworkType*>();

        auto extra_info = t[0].cast<bool>();
        if (extra_info) {
            py::function override = py::get_override(ptr, "__setstate_extra__");
            if (override) {
                override(t[1]);
            } else {
                py::pybind11_fail("Tried to call function \"BayesianNetworkType::__setstate_extra__\"");
            }
        }
    }
};

template <typename Base = BayesianNetworkBase>
class PyBayesianNetworkBase : public Base {
public:
    using Base::Base;

    int num_nodes() const override { PYBIND11_OVERRIDE_PURE(int, Base, num_nodes, ); }
    int num_arcs() const override { PYBIND11_OVERRIDE_PURE(int, Base, num_arcs, ); }

    const std::vector<std::string>& nodes() const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<std::string>&, Base, nodes, );
    }

    ArcStringVector arcs() const override { PYBIND11_OVERRIDE_PURE(ArcStringVector, Base, arcs, ); }

    const std::unordered_map<std::string, int>& indices() const override {
        PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(const std::unordered_map<std::string, int>&), Base, indices, );
    }

    int index(const std::string& node) const override { PYBIND11_OVERRIDE_PURE(int, Base, index, node); }

    int collapsed_index(const std::string& node) const override {
        PYBIND11_OVERRIDE_PURE(int, Base, collapsed_index, node);
    }

    int index_from_collapsed(int collapsed_index) const override {
        PYBIND11_OVERRIDE_PURE(int, Base, index_from_collapsed, collapsed_index);
    }

    int collapsed_from_index(int index) const override {
        PYBIND11_OVERRIDE_PURE(int, Base, collapsed_from_index, index);
    }

    const std::unordered_map<std::string, int>& collapsed_indices() const override {
        PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(const std::unordered_map<std::string, int>&), Base, collapsed_indices, );
    }

    bool is_valid(const std::string& name) const override { PYBIND11_OVERRIDE_PURE(bool, Base, is_valid, name); }

    bool contains_node(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(bool, Base, contains_node, name);
    }

    int add_node(const std::string& node) override { PYBIND11_OVERRIDE_PURE(int, Base, add_node, node); }

    void remove_node(const std::string& node) override { PYBIND11_OVERRIDE_PURE(void, Base, remove_node, node); }

    const std::string& name(int node_index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, Base, name, node_index);
    }

    const std::string& collapsed_name(int collapsed_index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, Base, collapsed_name, collapsed_index);
    }

    int num_parents(const std::string& node) const override { PYBIND11_OVERRIDE_PURE(int, Base, num_parents, node); }

    int num_children(const std::string& node) const override { PYBIND11_OVERRIDE_PURE(int, Base, num_children, node); }

    std::vector<std::string> parents(const std::string& node) const override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::string>, Base, parents, node);
    }

    std::vector<std::string> children(const std::string& node) const override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::string>, Base, children, node);
    }

    bool has_arc(const std::string& source, const std::string& target) const override {
        PYBIND11_OVERRIDE_PURE(bool, Base, has_arc, source, target);
    }

    bool has_path(const std::string& source, const std::string& target) const override {
        PYBIND11_OVERRIDE_PURE(bool, Base, has_path, source, target);
    }

    void add_arc(const std::string& source, const std::string& target) override {
        PYBIND11_OVERRIDE_PURE(void, Base, add_arc, source, target);
    }

    void add_arc_unsafe(const std::string& source, const std::string& target) override {
        PYBIND11_OVERRIDE_PURE(void, Base, add_arc_unsafe, source, target);
    }

    void remove_arc(const std::string& source, const std::string& target) override {
        PYBIND11_OVERRIDE_PURE(void, Base, remove_arc, source, target);
    }

    void flip_arc(const std::string& source, const std::string& target) override {
        PYBIND11_OVERRIDE_PURE(void, Base, flip_arc, source, target);
    }

    void flip_arc_unsafe(const std::string& source, const std::string& target) override {
        PYBIND11_OVERRIDE_PURE(void, Base, flip_arc_unsafe, source, target);
    }

    bool can_add_arc(const std::string& source, const std::string& target) const override {
        PYBIND11_OVERRIDE_PURE(bool, Base, can_add_arc, source, target);
    }

    bool can_flip_arc(const std::string& source, const std::string& target) const override {
        PYBIND11_OVERRIDE_PURE(bool, Base, can_flip_arc, source, target);
    }

    void check_blacklist(const ArcStringVector& arc_blacklist) const override {
        PYBIND11_OVERRIDE_PURE(void, Base, check_blacklist, arc_blacklist);
    }

    void force_whitelist(const ArcStringVector& arc_whitelist) override {
        PYBIND11_OVERRIDE_PURE(void, Base, force_whitelist, arc_whitelist);
    }

    bool fitted() const override { PYBIND11_OVERRIDE_PURE(bool, Base, fitted, ); }

    std::shared_ptr<Factor> cpd(const std::string& node) override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<Factor>, Base, cpd, node);
    }

    const std::shared_ptr<Factor> cpd(const std::string& node) const override {
        PYBIND11_OVERRIDE_PURE(const std::shared_ptr<Factor>, Base, cpd, node);
    }

    void add_cpds(const std::vector<std::shared_ptr<Factor>>& cpds) override {
        PYBIND11_OVERRIDE_PURE(void, Base, add_cpds, cpds);
    }

    void fit(const DataFrame& df) override { PYBIND11_OVERRIDE_PURE(void, Base, fit, df); }

    VectorXd logl(const DataFrame& df) const override { PYBIND11_OVERRIDE_PURE(VectorXd, Base, logl, df); }

    double slogl(const DataFrame& df) const override { PYBIND11_OVERRIDE_PURE(double, Base, slogl, df); }

    std::shared_ptr<BayesianNetworkType> type() const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<BayesianNetworkType>, Base, type, );
    }

    BayesianNetworkType& type_ref() const override { return *type(); }

    using Base::sample;
    DataFrame sample(int n, unsigned int seed, bool ordered) const override {
        PYBIND11_OVERRIDE_PURE(DataFrame, Base, sample, n, seed, ordered);
    }

    std::shared_ptr<ConditionalBayesianNetworkBase> conditional_bn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override {
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<ConditionalBayesianNetworkBase>, Base, conditional_bn, nodes, interface_nodes);
    }

    std::shared_ptr<ConditionalBayesianNetworkBase> conditional_bn() const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<ConditionalBayesianNetworkBase>, Base, conditional_bn, );
    }

    std::shared_ptr<BayesianNetworkBase> unconditional_bn() const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<BayesianNetworkBase>, Base, unconditional_bn, );
    }

    void save(std::string name, bool include_cpd = false) const override {
        PYBIND11_OVERRIDE_PURE(void, Base, save, name, include_cpd);
    }

    py::tuple __getstate__() const override { PYBIND11_OVERRIDE_PURE(py::tuple, Base, __getstate__, ); }

    std::shared_ptr<FactorType> node_type(const std::string& node) const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<FactorType>, Base, node_type, node);
    }

    std::unordered_map<std::string, std::shared_ptr<FactorType>> node_types() const override {
        PYBIND11_OVERRIDE_PURE(
            PYBIND11_TYPE(std::unordered_map<std::string, std::shared_ptr<FactorType>>), Base, node_types, );
    }

    void set_node_type(const std::string& node, const std::shared_ptr<FactorType>& new_type) override {
        PYBIND11_OVERRIDE_PURE(void, Base, set_node_type, node, new_type);
    }

    void force_type_whitelist(const FactorTypeVector& type_whitelist) override {
        PYBIND11_OVERRIDE_PURE(void, Base, force_type_whitelist, type_whitelist);
    }

    std::string ToString() const override { PYBIND11_OVERRIDE_PURE(std::string, Base, ToString, ); }
};

template <typename Base = ConditionalBayesianNetworkBase>
class PyConditionalBayesianNetworkBase : public PyBayesianNetworkBase<Base> {
public:
    using PyBayesianNetworkBase<Base>::PyBayesianNetworkBase;

    int num_interface_nodes() const override { PYBIND11_OVERRIDE_PURE(int, Base, num_interface_nodes, ); }

    int num_total_nodes() const override { PYBIND11_OVERRIDE_PURE(int, Base, num_total_nodes, ); }

    const std::vector<std::string>& interface_nodes() const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<std::string>&, Base, interface_nodes, );
    }

    const std::vector<std::string>& all_nodes() const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<std::string>&, Base, all_nodes, );
    }

    int interface_collapsed_index(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(int, Base, interface_collapsed_index, name);
    }

    int joint_collapsed_index(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(int, Base, joint_collapsed_index, name);
    }

    const std::unordered_map<std::string, int>& interface_collapsed_indices() const override {
        PYBIND11_OVERRIDE_PURE(
            PYBIND11_TYPE(const std::unordered_map<std::string, int>&), Base, interface_collapsed_indices, );
    }

    const std::unordered_map<std::string, int>& joint_collapsed_indices() const override {
        PYBIND11_OVERRIDE_PURE(
            PYBIND11_TYPE(const std::unordered_map<std::string, int>&), Base, joint_collapsed_indices, );
    }

    int index_from_interface_collapsed(int interface_collapsed_index) const override {
        PYBIND11_OVERRIDE_PURE(int, Base, index_from_interface_collapsed, interface_collapsed_index);
    }

    int index_from_joint_collapsed(int joint_collapsed_index) const override {
        PYBIND11_OVERRIDE_PURE(int, Base, index_from_joint_collapsed, joint_collapsed_index);
    }

    int interface_collapsed_from_index(int index) const override {
        PYBIND11_OVERRIDE_PURE(int, Base, interface_collapsed_from_index, index);
    }

    int joint_collapsed_from_index(int index) const override {
        PYBIND11_OVERRIDE_PURE(int, Base, joint_collapsed_from_index, index);
    }

    const std::string& interface_collapsed_name(int interface_collapsed_index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, Base, interface_collapsed_name, interface_collapsed_index);
    }

    const std::string& joint_collapsed_name(int joint_collapsed_index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, Base, joint_collapsed_name, joint_collapsed_index);
    }

    bool contains_interface_node(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(bool, Base, contains_interface_node, name);
    }

    bool contains_total_node(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(bool, Base, contains_total_node, name);
    }

    int add_interface_node(const std::string& node) override {
        PYBIND11_OVERRIDE_PURE(int, Base, add_interface_node, node);
    }

    void remove_interface_node(const std::string& node) override {
        PYBIND11_OVERRIDE_PURE(void, Base, remove_interface_node, node);
    }

    bool is_interface(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(bool, Base, is_interface, name);
    }

    void set_interface(const std::string& name) override { PYBIND11_OVERRIDE_PURE(void, Base, set_interface, name); }

    void set_node(const std::string& name) override { PYBIND11_OVERRIDE_PURE(void, Base, set_node, name); }

    using BayesianNetworkBase::sample;
    DataFrame sample(const DataFrame& evidence, unsigned int seed, bool concat_evidence, bool ordered) const override {
        PYBIND11_OVERRIDE_PURE(DataFrame, Base, sample, evidence, seed, concat_evidence, ordered);
    }
};

std::shared_ptr<BayesianNetworkType> keep_python_instance_alive(std::shared_ptr<BayesianNetworkType> type) {
    py::object o = py::cast(type);
    auto keep_python_state_alive = std::make_shared<py::object>(o);
    auto ptr = o.cast<BayesianNetworkType*>();

    return std::shared_ptr<BayesianNetworkType>(keep_python_state_alive, ptr);
}

template <typename Base = BayesianNetwork>
class PyBayesianNetwork : public PyBayesianNetworkBase<Base> {
public:
    using PyBayesianNetworkBase<Base>::PyBayesianNetworkBase;

    int num_nodes() const override { PYBIND11_OVERRIDE(int, Base, num_nodes, ); }
    int num_arcs() const override { PYBIND11_OVERRIDE(int, Base, num_arcs, ); }

    const std::vector<std::string>& nodes() const override {
        PYBIND11_OVERRIDE(const std::vector<std::string>&, Base, nodes, );
    }

    ArcStringVector arcs() const override { PYBIND11_OVERRIDE(ArcStringVector, Base, arcs, ); }

    const std::unordered_map<std::string, int>& indices() const override {
        PYBIND11_OVERRIDE(PYBIND11_TYPE(const std::unordered_map<std::string, int>&), Base, indices, );
    }

    int index(const std::string& node) const override { PYBIND11_OVERRIDE(int, Base, index, node); }

    int collapsed_index(const std::string& node) const override { PYBIND11_OVERRIDE(int, Base, collapsed_index, node); }

    int index_from_collapsed(int collapsed_index) const override {
        PYBIND11_OVERRIDE(int, Base, index_from_collapsed, collapsed_index);
    }

    int collapsed_from_index(int index) const override { PYBIND11_OVERRIDE(int, Base, collapsed_from_index, index); }

    const std::unordered_map<std::string, int>& collapsed_indices() const override {
        PYBIND11_OVERRIDE(PYBIND11_TYPE(const std::unordered_map<std::string, int>&), Base, collapsed_indices, );
    }

    bool is_valid(const std::string& name) const override { PYBIND11_OVERRIDE(bool, Base, is_valid, name); }

    bool contains_node(const std::string& name) const override { PYBIND11_OVERRIDE(bool, Base, contains_node, name); }

    int add_node(const std::string& node) override { PYBIND11_OVERRIDE(int, Base, add_node, node); }

    void remove_node(const std::string& node) override { PYBIND11_OVERRIDE(void, Base, remove_node, node); }

    const std::string& name(int node_index) const override {
        PYBIND11_OVERRIDE(const std::string&, Base, name, node_index);
    }

    const std::string& collapsed_name(int collapsed_index) const override {
        PYBIND11_OVERRIDE(const std::string&, Base, collapsed_name, collapsed_index);
    }

    int num_parents(const std::string& node) const override { PYBIND11_OVERRIDE(int, Base, num_parents, node); }

    int num_children(const std::string& node) const override { PYBIND11_OVERRIDE(int, Base, num_children, node); }

    std::vector<std::string> parents(const std::string& node) const override {
        PYBIND11_OVERRIDE(std::vector<std::string>, Base, parents, node);
    }

    std::vector<std::string> children(const std::string& node) const override {
        PYBIND11_OVERRIDE(std::vector<std::string>, Base, children, node);
    }

    bool has_arc(const std::string& source, const std::string& target) const override {
        PYBIND11_OVERRIDE(bool, Base, has_arc, source, target);
    }

    bool has_path(const std::string& source, const std::string& target) const override {
        PYBIND11_OVERRIDE(bool, Base, has_path, source, target);
    }

    void add_arc(const std::string& source, const std::string& target) override {
        PYBIND11_OVERRIDE(void, Base, add_arc, source, target);
    }

    void add_arc_unsafe(const std::string& source, const std::string& target) override {
        PYBIND11_OVERRIDE(void, Base, add_arc_unsafe, source, target);
    }

    void remove_arc(const std::string& source, const std::string& target) override {
        PYBIND11_OVERRIDE(void, Base, remove_arc, source, target);
    }

    void flip_arc(const std::string& source, const std::string& target) override {
        PYBIND11_OVERRIDE(void, Base, flip_arc, source, target);
    }

    void flip_arc_unsafe(const std::string& source, const std::string& target) override {
        PYBIND11_OVERRIDE(void, Base, flip_arc_unsafe, source, target);
    }

    bool can_add_arc(const std::string& source, const std::string& target) const override {
        PYBIND11_OVERRIDE(bool, Base, can_add_arc, source, target);
    }

    bool can_flip_arc(const std::string& source, const std::string& target) const override {
        PYBIND11_OVERRIDE(bool, Base, can_flip_arc, source, target);
    }

    void check_blacklist(const ArcStringVector& arc_blacklist) const override {
        PYBIND11_OVERRIDE(void, Base, check_blacklist, arc_blacklist);
    }

    void force_whitelist(const ArcStringVector& arc_whitelist) override {
        PYBIND11_OVERRIDE(void, Base, force_whitelist, arc_whitelist);
    }

    bool can_have_cpd(const std::string& name) const override { PYBIND11_OVERRIDE(bool, Base, can_have_cpd, name); }

    bool fitted() const override { PYBIND11_OVERRIDE(bool, Base, fitted, ); }

    std::shared_ptr<Factor> cpd(const std::string& node) override {
        PYBIND11_OVERRIDE(std::shared_ptr<Factor>, Base, cpd, node);
    }

    const std::shared_ptr<Factor> cpd(const std::string& node) const override {
        PYBIND11_OVERRIDE(const std::shared_ptr<Factor>, Base, cpd, node);
    }

    void check_compatible_cpd(const Factor& cpd) const override {
        PYBIND11_OVERRIDE(void, Base, check_compatible_cpd, cpd);
    }

    bool must_construct_cpd(const Factor& cpd,
                            const FactorType& model_node_type,
                            const std::vector<std::string>& model_parents) const override {
        PYBIND11_OVERRIDE(bool, Base, must_construct_cpd, cpd, model_node_type, model_parents);
    }

    void add_cpds(const std::vector<std::shared_ptr<Factor>>& cpds) override {
        PYBIND11_OVERRIDE(void, Base, add_cpds, cpds);
    }

    void fit(const DataFrame& df) override { PYBIND11_OVERRIDE(void, Base, fit, df); }

    VectorXd logl(const DataFrame& df) const override { PYBIND11_OVERRIDE(VectorXd, Base, logl, df); }

    double slogl(const DataFrame& df) const override { PYBIND11_OVERRIDE(double, Base, slogl, df); }

    std::shared_ptr<BayesianNetworkType> type() const override {
        PYBIND11_OVERRIDE(std::shared_ptr<BayesianNetworkType>, Base, type, );
    }

    BayesianNetworkType& type_ref() const override { return *type(); }

    using Base::sample;
    DataFrame sample(int n, unsigned int seed, bool ordered) const override {
        PYBIND11_OVERRIDE(DataFrame, Base, sample, n, seed, ordered);
    }

    std::shared_ptr<ConditionalBayesianNetworkBase> conditional_bn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override {
        PYBIND11_OVERRIDE(
            std::shared_ptr<ConditionalBayesianNetworkBase>, Base, conditional_bn, nodes, interface_nodes);
    }

    std::shared_ptr<ConditionalBayesianNetworkBase> conditional_bn() const override {
        PYBIND11_OVERRIDE(std::shared_ptr<ConditionalBayesianNetworkBase>, Base, conditional_bn, );
    }

    std::shared_ptr<BayesianNetworkBase> unconditional_bn() const override {
        PYBIND11_OVERRIDE(std::shared_ptr<BayesianNetworkBase>, Base, unconditional_bn, );
    }

    void save(std::string name, bool include_cpd = false) const override {
        PYBIND11_OVERRIDE(void, Base, save, name, include_cpd);
    }

    py::tuple __getstate__() const override {
        auto t = Base::__getstate__();
        py::gil_scoped_acquire gil;

        py::function override = py::get_override(static_cast<const Base*>(this), "__getstate_extra__");

        if (override) {
            auto extra = override();
            return py::make_tuple(t, true, py::make_tuple(extra));
        } else {
            return py::make_tuple(t, false, py::make_tuple());
        }
    }

    static void __setstate__(py::object& self, py::tuple& t) {
        if (t.size() != 3) throw std::runtime_error("Not valid BayesianNetwork.");

        using DagType = typename Base::DagClass;
        py::gil_scoped_acquire gil;

        auto pybntype = py::type::of<Base>();

        auto bn_base = t[0].cast<py::tuple>();
        auto dag = bn_base[0].cast<DagType>();
        auto type = bn_base[1].cast<std::shared_ptr<BayesianNetworkType>>();

        if (type->is_homogeneous()) {
            pybntype.attr("__init__")(self, type, std::move(dag));
        } else {
            auto node_types = bn_base[2].cast<FactorTypeVector>();
            if (node_types.empty()) pybntype.attr("__init__")(self, type, std::move(dag));

            if constexpr (std::is_constructible_v<Base,
                                                  std::shared_ptr<BayesianNetworkType>,
                                                  DagType&&,
                                                  FactorTypeVector>) {
                pybntype.attr("__init__")(self, type, std::move(dag), node_types);
            } else {
                throw std::runtime_error("Invalid node types array for non-homogeneous Bayesian network.");
            }
        }

        auto self_cpp = self.cast<Base*>();

        if (bn_base[3].cast<bool>()) {
            auto cpds = bn_base[4].cast<std::vector<std::shared_ptr<Factor>>>();

            self_cpp->add_cpds(cpds);
        }

        if (t[1].cast<bool>()) {
            auto extra = t[2].cast<py::tuple>();

            py::gil_scoped_acquire gil;

            py::function override = py::get_override(self_cpp, "__setstate_extra__");
            if (override) {
                override(extra[0]);
            } else {
                py::pybind11_fail("Tried to call \"BayesianNetwork::__setstate_extra__\"");
            }
        }
    }

    std::shared_ptr<FactorType> node_type(const std::string& node) const override {
        PYBIND11_OVERRIDE(std::shared_ptr<FactorType>, Base, node_type, node);
    }

    std::unordered_map<std::string, std::shared_ptr<FactorType>> node_types() const override {
        PYBIND11_OVERRIDE(
            PYBIND11_TYPE(std::unordered_map<std::string, std::shared_ptr<FactorType>>), Base, node_types, );
    }

    void set_node_type(const std::string& node, const std::shared_ptr<FactorType>& new_type) override {
        PYBIND11_OVERRIDE(void, Base, set_node_type, node, new_type);
    }

    void force_type_whitelist(const FactorTypeVector& type_whitelist) override {
        PYBIND11_OVERRIDE(void, Base, force_type_whitelist, type_whitelist);
    }

    std::string ToString() const override { PYBIND11_OVERRIDE(std::string, Base, ToString, ); }
};

template <typename Base = ConditionalBayesianNetwork>
class PyConditionalBayesianNetwork : public PyBayesianNetwork<Base> {
public:
    using PyBayesianNetwork<Base>::PyBayesianNetwork;

    int num_interface_nodes() const override { PYBIND11_OVERRIDE(int, Base, num_interface_nodes, ); }

    int num_total_nodes() const override { PYBIND11_OVERRIDE(int, Base, num_total_nodes, ); }

    const std::vector<std::string>& interface_nodes() const override {
        PYBIND11_OVERRIDE(const std::vector<std::string>&, Base, interface_nodes, );
    }

    const std::vector<std::string>& all_nodes() const override {
        PYBIND11_OVERRIDE(const std::vector<std::string>&, Base, all_nodes, );
    }

    int interface_collapsed_index(const std::string& name) const override {
        PYBIND11_OVERRIDE(int, Base, interface_collapsed_index, name);
    }

    int joint_collapsed_index(const std::string& name) const override {
        PYBIND11_OVERRIDE(int, Base, joint_collapsed_index, name);
    }

    const std::unordered_map<std::string, int>& interface_collapsed_indices() const override {
        PYBIND11_OVERRIDE(
            PYBIND11_TYPE(const std::unordered_map<std::string, int>&), Base, interface_collapsed_indices, );
    }

    const std::unordered_map<std::string, int>& joint_collapsed_indices() const override {
        PYBIND11_OVERRIDE(PYBIND11_TYPE(const std::unordered_map<std::string, int>&), Base, joint_collapsed_indices, );
    }

    int index_from_interface_collapsed(int interface_collapsed_index) const override {
        PYBIND11_OVERRIDE(int, Base, index_from_interface_collapsed, interface_collapsed_index);
    }

    int index_from_joint_collapsed(int joint_collapsed_index) const override {
        PYBIND11_OVERRIDE(int, Base, index_from_joint_collapsed, joint_collapsed_index);
    }

    int interface_collapsed_from_index(int index) const override {
        PYBIND11_OVERRIDE(int, Base, interface_collapsed_from_index, index);
    }

    int joint_collapsed_from_index(int index) const override {
        PYBIND11_OVERRIDE(int, Base, joint_collapsed_from_index, index);
    }

    const std::string& interface_collapsed_name(int interface_collapsed_index) const override {
        PYBIND11_OVERRIDE(const std::string&, Base, interface_collapsed_name, interface_collapsed_index);
    }

    const std::string& joint_collapsed_name(int joint_collapsed_index) const override {
        PYBIND11_OVERRIDE(const std::string&, Base, joint_collapsed_name, joint_collapsed_index);
    }

    bool contains_interface_node(const std::string& name) const override {
        PYBIND11_OVERRIDE(bool, Base, contains_interface_node, name);
    }

    bool contains_total_node(const std::string& name) const override {
        PYBIND11_OVERRIDE(bool, Base, contains_total_node, name);
    }

    int add_interface_node(const std::string& node) override { PYBIND11_OVERRIDE(int, Base, add_interface_node, node); }

    void remove_interface_node(const std::string& node) override {
        PYBIND11_OVERRIDE(void, Base, remove_interface_node, node);
    }

    bool is_interface(const std::string& name) const override { PYBIND11_OVERRIDE(bool, Base, is_interface, name); }

    void set_interface(const std::string& name) override { PYBIND11_OVERRIDE(void, Base, set_interface, name); }

    void set_node(const std::string& name) override { PYBIND11_OVERRIDE(void, Base, set_node, name); }

    using BayesianNetworkBase::sample;
    DataFrame sample(const DataFrame& evidence, unsigned int seed, bool concat_evidence, bool ordered) const override {
        PYBIND11_OVERRIDE(DataFrame, Base, sample, evidence, seed, concat_evidence, ordered);
    }
};

template <typename Base = DynamicBayesianNetworkBase>
class PyDynamicBayesianNetworkBase : public Base {
public:
    using Base::Base;

    BayesianNetworkBase& static_bn() override { PYBIND11_OVERRIDE_PURE(BayesianNetworkBase&, Base, static_bn, ); }

    const BayesianNetworkBase& static_bn() const override {
        PYBIND11_OVERRIDE_PURE(const BayesianNetworkBase&, Base, static_bn, );
    }

    ConditionalBayesianNetworkBase& transition_bn() override {
        PYBIND11_OVERRIDE_PURE(ConditionalBayesianNetworkBase&, Base, transition_bn, );
    }

    const ConditionalBayesianNetworkBase& transition_bn() const override {
        PYBIND11_OVERRIDE_PURE(const ConditionalBayesianNetworkBase&, Base, transition_bn, );
    }

    int markovian_order() const override { PYBIND11_OVERRIDE_PURE(int, Base, markovian_order, ); }

    int num_variables() const override { PYBIND11_OVERRIDE_PURE(int, Base, num_variables, ); }

    const std::vector<std::string>& variables() const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<std::string>&, Base, variables, );
    }

    bool contains_variable(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(bool, Base, contains_variable, name);
    }

    void add_variable(const std::string& name) override { PYBIND11_OVERRIDE_PURE(void, Base, add_variable, name); }

    void remove_variable(const std::string& name) override {
        PYBIND11_OVERRIDE_PURE(void, Base, remove_variable, name);
    }

    bool fitted() const override { PYBIND11_OVERRIDE_PURE(bool, Base, fitted, ); }

    void fit(const DataFrame& df) override { PYBIND11_OVERRIDE_PURE(void, Base, fit, df); }

    VectorXd logl(const DataFrame& df) const override { PYBIND11_OVERRIDE_PURE(VectorXd, Base, logl, df); }

    double slogl(const DataFrame& df) const override { PYBIND11_OVERRIDE_PURE(double, Base, slogl, df); }

    std::shared_ptr<BayesianNetworkType> type() const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<BayesianNetworkType>, Base, type, );
    }

    BayesianNetworkType& type_ref() const override { return *type(); }

    DataFrame sample(int n, unsigned int seed) const override {
        PYBIND11_OVERRIDE_PURE(DataFrame, Base, sample, n, seed);
    }

    void save(std::string name, bool include_cpd = false) const override {
        PYBIND11_OVERRIDE_PURE(void, Base, save, name, include_cpd);
    }

    std::string ToString() const override { PYBIND11_OVERRIDE_PURE(std::string, Base, ToString, ); }
};

template <typename Base = DynamicBayesianNetwork>
class PyDynamicBayesianNetwork : public PyDynamicBayesianNetworkBase<Base> {
public:
    using PyDynamicBayesianNetworkBase<Base>::PyDynamicBayesianNetworkBase;

    BayesianNetworkBase& static_bn() override { PYBIND11_OVERRIDE(BayesianNetworkBase&, Base, static_bn, ); }

    const BayesianNetworkBase& static_bn() const override {
        PYBIND11_OVERRIDE(const BayesianNetworkBase&, Base, static_bn, );
    }

    ConditionalBayesianNetworkBase& transition_bn() override {
        PYBIND11_OVERRIDE(ConditionalBayesianNetworkBase&, Base, transition_bn, );
    }

    const ConditionalBayesianNetworkBase& transition_bn() const override {
        PYBIND11_OVERRIDE(const ConditionalBayesianNetworkBase&, Base, transition_bn, );
    }

    int markovian_order() const override { PYBIND11_OVERRIDE(int, Base, markovian_order, ); }

    int num_variables() const override { PYBIND11_OVERRIDE(int, Base, num_variables, ); }

    const std::vector<std::string>& variables() const override {
        PYBIND11_OVERRIDE(const std::vector<std::string>&, Base, variables, );
    }

    bool contains_variable(const std::string& name) const override {
        PYBIND11_OVERRIDE(bool, Base, contains_variable, name);
    }

    void add_variable(const std::string& name) override { PYBIND11_OVERRIDE(void, Base, add_variable, name); }

    void remove_variable(const std::string& name) override { PYBIND11_OVERRIDE(void, Base, remove_variable, name); }

    bool fitted() const override { PYBIND11_OVERRIDE(bool, Base, fitted, ); }

    void fit(const DataFrame& df) override { PYBIND11_OVERRIDE(void, Base, fit, df); }

    VectorXd logl(const DataFrame& df) const override { PYBIND11_OVERRIDE(VectorXd, Base, logl, df); }

    double slogl(const DataFrame& df) const override { PYBIND11_OVERRIDE(double, Base, slogl, df); }

    std::shared_ptr<BayesianNetworkType> type() const override {
        PYBIND11_OVERRIDE(std::shared_ptr<BayesianNetworkType>, Base, type, );
    }

    BayesianNetworkType& type_ref() const override { return *type(); }

    DataFrame sample(int n, unsigned int seed) const override { PYBIND11_OVERRIDE(DataFrame, Base, sample, n, seed); }

    void save(std::string name, bool include_cpd = false) const override {
        PYBIND11_OVERRIDE(void, Base, save, name, include_cpd);
    }

    std::string ToString() const override { PYBIND11_OVERRIDE(std::string, Base, ToString, ); }

    py::tuple __getstate__() const override {
        auto t = Base::__getstate__();
        py::gil_scoped_acquire gil;

        py::function override = py::get_override(static_cast<const Base*>(this), "__getstate_extra__");

        if (override) {
            auto extra = override();
            return py::make_tuple(t, true, py::make_tuple(extra));
        } else {
            return py::make_tuple(t, false, py::make_tuple());
        }
    }

    static void __setstate__(py::object& self, py::tuple& t) {
        if (t.size() != 3) throw std::runtime_error("Not valid DynamicBayesianNetwork.");

        py::gil_scoped_acquire gil;

        auto bn_base = t[0].cast<py::tuple>();

        models::__nonderived_dbn_setstate__(self, bn_base);

        auto self_cpp = self.cast<Base*>();

        if (t[1].cast<bool>()) {
            auto extra = t[2].cast<py::tuple>();

            py::function override = py::get_override(self_cpp, "__setstate_extra__");
            if (override) {
                override(extra[0]);
            } else {
                py::pybind11_fail("Tried to call \"DynamicBayesianNetwork::__setstate_extra__\"");
            }
        }
    }
};

template <typename CppClass, typename Class>
void register_BayesianNetwork_methods(Class& m) {
    m.def_property_readonly("fitted", &CppClass::fitted)
        .def("num_nodes", &CppClass::num_nodes)
        .def("num_arcs", &CppClass::num_arcs)
        .def("nodes", &CppClass::nodes, py::return_value_policy::reference_internal)
        .def("arcs", &CppClass::arcs, py::return_value_policy::take_ownership)
        .def("indices", &CppClass::indices, py::return_value_policy::reference_internal)
        .def("index", &CppClass::index)
        .def("collapsed_index", &CppClass::collapsed_index)
        .def("index_from_collapsed", &CppClass::index_from_collapsed)
        .def("collapsed_from_index", &CppClass::collapsed_from_index)
        .def("collapsed_indices", &CppClass::collapsed_indices, py::return_value_policy::reference_internal)
        .def("is_valid", &CppClass::is_valid)
        .def("contains_node", &CppClass::contains_node)
        .def("add_node", &CppClass::add_node)
        .def("remove_node", &CppClass::remove_node)
        .def("name", &CppClass::name)
        .def("collapsed_name", &CppClass::collapsed_name)
        .def("num_parents", &CppClass::num_parents)
        .def("num_children", &CppClass::num_children)
        .def("parents", &CppClass::parents, py::return_value_policy::take_ownership)
        .def("children", &CppClass::children, py::return_value_policy::take_ownership)
        .def("has_arc", &CppClass::has_arc)
        .def("has_path", &CppClass::has_path)
        .def("add_arc", &CppClass::add_arc)
        .def("add_arc_unsafe", &CppClass::add_arc_unsafe)
        .def("remove_arc", &CppClass::remove_arc)
        .def("flip_arc", &CppClass::flip_arc)
        .def("flip_arc_unsafe", &CppClass::flip_arc_unsafe)
        .def("can_add_arc", &CppClass::can_add_arc)
        .def("can_flip_arc", &CppClass::can_flip_arc)
        .def("force_whitelist", &CppClass::force_whitelist)
        .def("cpd", py::overload_cast<const std::string&>(&CppClass::cpd))
        .def("add_cpds", &CppClass::add_cpds)
        .def("fit", &CppClass::fit)
        .def("logl", &CppClass::logl, py::return_value_policy::take_ownership)
        .def("slogl", &CppClass::slogl)
        .def("type", &CppClass::type)
        .def(
            "sample",
            [](const CppClass& self, int n, bool ordered) { return self.sample(n, std::random_device{}(), ordered); },
            py::return_value_policy::move,
            py::arg("n"),
            py::arg("ordered") = false)
        .def(
            "sample",
            [](const CppClass& self, int n, std::optional<unsigned int> seed, bool ordered) {
                return self.sample(n, random_seed_arg(seed), ordered);
            },
            py::return_value_policy::move,
            py::arg("n"),
            py::arg("seed") = std::nullopt,
            py::arg("ordered") = false)
        .def("conditional_bn",
             py::overload_cast<const std::vector<std::string>&, const std::vector<std::string>&>(
                 &CppClass::conditional_bn, py::const_))
        .def("conditional_bn", py::overload_cast<>(&CppClass::conditional_bn, py::const_))
        .def("unconditional_bn", &CppClass::unconditional_bn)
        .def("save", &CppClass::save)
        .def("node_type", &CppClass::node_type)
        .def("node_types", &CppClass::node_types)
        .def("set_node_type", &CppClass::set_node_type)
        .def("force_type_whitelist", &CppClass::force_type_whitelist)
        .def("clone", &CppClass::clone)
        .def("ToString", &CppClass::ToString);
}

template <typename CppClass, typename Class>
void register_ConditionalBayesianNetwork_methods(Class& m) {
    m.def("num_interface_nodes", &CppClass::num_interface_nodes)
        .def("num_total_nodes", &CppClass::num_total_nodes)
        .def("interface_nodes", &CppClass::interface_nodes, py::return_value_policy::reference_internal)
        .def("all_nodes", &CppClass::all_nodes, py::return_value_policy::reference_internal)
        .def("interface_collapsed_index", &CppClass::interface_collapsed_index)
        .def("joint_collapsed_index", &CppClass::joint_collapsed_index)
        .def("interface_collapsed_indices",
             &CppClass::interface_collapsed_indices,
             py::return_value_policy::reference_internal)
        .def("joint_collapsed_indices", &CppClass::joint_collapsed_indices, py::return_value_policy::reference_internal)
        .def("index_from_interface_collapsed", &CppClass::index_from_interface_collapsed)
        .def("index_from_joint_collapsed", &CppClass::index_from_joint_collapsed)
        .def("interface_collapsed_from_index", &CppClass::interface_collapsed_from_index)
        .def("joint_collapsed_from_index", &CppClass::joint_collapsed_from_index)
        .def("interface_collapsed_name", &CppClass::interface_collapsed_name)
        .def("joint_collapsed_name", &CppClass::joint_collapsed_name)
        .def("contains_interface_node", &CppClass::contains_interface_node)
        .def("contains_total_node", &CppClass::contains_total_node)
        .def("add_interface_node", &CppClass::add_interface_node)
        .def("remove_interface_node", &CppClass::remove_interface_node)
        .def("is_interface", &CppClass::is_interface)
        .def("set_interface", &CppClass::set_interface)
        .def("set_node", &CppClass::set_node)
        .def(
            "sample",
            [](const CppClass& self, const DataFrame& evidence, bool concat_evidence, bool ordered) {
                return self.sample(evidence, std::random_device{}(), concat_evidence, ordered);
            },
            py::return_value_policy::move,
            py::arg("evidence"),
            py::arg("concat_evidence") = false,
            py::arg("ordered") = false)
        .def(
            "sample",
            [](const CppClass& self,
               const DataFrame& evidence,
               std::optional<unsigned int> seed,
               bool concat_evidence,
               bool ordered) { return self.sample(evidence, random_seed_arg(seed), concat_evidence, ordered); },
            py::return_value_policy::move,
            py::arg("evidence"),
            py::arg("seed") = std::nullopt,
            py::arg("concat_evidence") = false,
            py::arg("ordered") = false)
        .def("clone", &CppClass::clone);
}

template <typename CppClass, typename Class>
void register_BNGeneric_methods(Class& m) {
    m.def_property("include_cpd", &CppClass::include_cpd, &CppClass::set_include_cpd)
        .def("graph", py::overload_cast<>(&CppClass::graph))
        .def("num_raw_nodes", &CppClass::num_raw_nodes)
        .def("can_have_cpd", &CppClass::can_have_cpd)
        .def("check_compatible_cpd", &CppClass::check_compatible_cpd);
}

template <typename CppClass, typename Class>
void register_DynamicBayesianNetwork_methods(Class& m) {
    m.def_property_readonly("fitted", &CppClass::fitted)
        .def("static_bn", py::overload_cast<>(&CppClass::static_bn), py::return_value_policy::reference_internal)
        .def(
            "transition_bn", py::overload_cast<>(&CppClass::transition_bn), py::return_value_policy::reference_internal)
        .def("markovian_order", &CppClass::markovian_order)
        .def("num_variables", &CppClass::num_variables)
        .def("variables", &CppClass::variables)
        .def("contains_variable", &CppClass::contains_variable)
        .def("add_variable", &CppClass::add_variable)
        .def("remove_variable", &CppClass::remove_variable)
        .def("fit", &CppClass::fit)
        .def("logl", &CppClass::logl)
        .def("slogl", &CppClass::slogl)
        .def("type", &CppClass::type)
        .def("sample", &CppClass::sample)
        .def("save", &CppClass::save)
        .def("ToString", &CppClass::ToString);
}

template <typename DerivedBN>
py::class_<DerivedBN, BayesianNetwork, std::shared_ptr<DerivedBN>> register_DerivedBayesianNetwork(
    py::module& m, const char* derivedbn_name) {
    return py::class_<DerivedBN, BayesianNetwork, std::shared_ptr<DerivedBN>>(m, derivedbn_name)
        .def(py::init<const std::vector<std::string>&>())
        .def(py::init<const ArcStringVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcStringVector&>())
        .def(py::init<const Dag&>())
        .def(py::pickle([](const DerivedBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__derived_bn_setstate__<DerivedBN>(t); }));
}

template <typename DerivedBN>
py::class_<DerivedBN, ConditionalBayesianNetwork, std::shared_ptr<DerivedBN>>
register_DerivedConditionalBayesianNetwork(py::module& m, const char* derivedbn_name) {
    return py::class_<DerivedBN, ConditionalBayesianNetwork, std::shared_ptr<DerivedBN>>(m, derivedbn_name)
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>())
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const ArcStringVector&>())
        .def(py::init<const ConditionalDag&>())
        .def(py::pickle([](const DerivedBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__derived_bn_setstate__<DerivedBN>(t); }));
}

template <typename DerivedBN>
py::class_<DerivedBN, DynamicBayesianNetwork, std::shared_ptr<DerivedBN>> register_DerivedDynamicBayesianNetwork(
    py::module& m, const char* derivedbn_name) {
    return py::class_<DerivedBN, DynamicBayesianNetwork, std::shared_ptr<DerivedBN>>(m, derivedbn_name)
        .def(py::init<const std::vector<std::string>&, int>())
        .def(py::init<const std::vector<std::string>&,
                      int,
                      std::shared_ptr<BayesianNetworkBase>,
                      std::shared_ptr<ConditionalBayesianNetworkBase>>())
        .def(py::pickle([](const DerivedBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__derived_dbn_setstate__<DerivedBN>(t); }));
}

void pybindings_models(py::module& root) {
    auto models = root.def_submodule("models", "Models submodule.");

    py::class_<BayesianNetworkType, PyBayesianNetworkType, std::shared_ptr<BayesianNetworkType>>(models,
                                                                                                 "BayesianNetworkType")
        .def(py::init<>())
        .def("is_homogeneous", &BayesianNetworkType::is_homogeneous)
        .def("default_node_type", &BayesianNetworkType::default_node_type)
        .def("compatible_node_type",
             py::overload_cast<const ConditionalBayesianNetworkBase&, const std::string&>(
                 &BayesianNetworkType::compatible_node_type, py::const_))
        .def("compatible_node_type",
             py::overload_cast<const BayesianNetworkBase&, const std::string&>(
                 &BayesianNetworkType::compatible_node_type, py::const_))
        .def("can_add_arc",
             py::overload_cast<const ConditionalBayesianNetworkBase&, const std::string&, const std::string&>(
                 &BayesianNetworkType::can_add_arc, py::const_))
        .def("can_add_arc",
             py::overload_cast<const BayesianNetworkBase&, const std::string&, const std::string&>(
                 &BayesianNetworkType::can_add_arc, py::const_))
        .def("can_flip_arc",
             py::overload_cast<const ConditionalBayesianNetworkBase&, const std::string&, const std::string&>(
                 &BayesianNetworkType::can_flip_arc, py::const_))
        .def("can_flip_arc",
             py::overload_cast<const BayesianNetworkBase&, const std::string&, const std::string&>(
                 &BayesianNetworkType::can_flip_arc, py::const_))
        // The equality operator do not compile in GCC, so it is implemented with lambdas:
        // https://github.com/pybind/pybind11/issues/1487
        .def(
            "__eq__",
            [](const BayesianNetworkType& self, const BayesianNetworkType& other) { return self == other; },
            py::is_operator())
        .def(
            "__ne__",
            [](const BayesianNetworkType& self, const BayesianNetworkType& other) { return self != other; },
            py::is_operator())
        // .def(py::self == py::self)
        // .def(py::self != py::self)
        .def("ToString", &BayesianNetworkType::ToString)
        .def("__getstate__", [](const BayesianNetworkType& self) { return self.__getstate__(); })
        .def("__setstate__", [](py::object& self, py::tuple& t) { PyBayesianNetworkType::__setstate__(self, t); });

    py::class_<GaussianNetworkType, BayesianNetworkType, std::shared_ptr<GaussianNetworkType>>(models,
                                                                                               "GaussianNetworkType")
        .def(py::init(&GaussianNetworkType::get))
        .def(py::pickle([](const GaussianNetworkType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return GaussianNetworkType::get(); }));

    py::class_<SemiparametricBNType, BayesianNetworkType, std::shared_ptr<SemiparametricBNType>>(models,
                                                                                                 "SemiparametricBNType")
        .def(py::init(&SemiparametricBNType::get))
        .def(py::pickle([](const SemiparametricBNType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return SemiparametricBNType::get(); }));

    py::class_<KDENetworkType, BayesianNetworkType, std::shared_ptr<KDENetworkType>>(models, "KDENetworkType")
        .def(py::init(&KDENetworkType::get))
        .def(py::pickle([](const KDENetworkType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return KDENetworkType::get(); }));

    py::class_<DiscreteBNType, BayesianNetworkType, std::shared_ptr<DiscreteBNType>>(models, "DiscreteBNType")
        .def(py::init(&DiscreteBNType::get))
        .def(py::pickle([](const DiscreteBNType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return DiscreteBNType::get(); }));

    py::class_<BayesianNetworkBase, PyBayesianNetworkBase<>, std::shared_ptr<BayesianNetworkBase>> bn_base(
        models, "BayesianNetworkBase");
    register_BayesianNetwork_methods<BayesianNetworkBase>(bn_base);

    py::class_<ConditionalBayesianNetworkBase,
               BayesianNetworkBase,
               PyConditionalBayesianNetworkBase<>,
               std::shared_ptr<ConditionalBayesianNetworkBase>>
        cbn_base(models, "ConditionalBayesianNetworkBase");
    register_ConditionalBayesianNetwork_methods<ConditionalBayesianNetworkBase>(cbn_base);

    py::class_<DynamicBayesianNetworkBase, PyDynamicBayesianNetworkBase<>, std::shared_ptr<DynamicBayesianNetworkBase>>
        dbn_base(models, "DynamicBayesianNetworkBase");
    register_DynamicBayesianNetwork_methods<DynamicBayesianNetworkBase>(dbn_base);

    /** Replicate here the constructors because the Python derived object is destroyed if
     * goes out of scope. It will be probably fixed in:
     * https://github.com/pybind/pybind11/pull/2839
     * See also:
     * https://github.com/pybind/pybind11/issues/1333.
     */
    py::class_<BayesianNetwork, BayesianNetworkBase, PyBayesianNetwork<>, std::shared_ptr<BayesianNetwork>> bn(
        models, "BayesianNetwork");
    bn.def(py::init(
               [](std::shared_ptr<BayesianNetworkType> t, const std::vector<std::string>& nodes) {
                   return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), nodes);
               },
               [](std::shared_ptr<BayesianNetworkType> t, const std::vector<std::string>& nodes) {
                   return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t), nodes);
               }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const FactorTypeVector& node_types) {
                return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), nodes, node_types);
            },
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const FactorTypeVector& node_types) {
                return std::make_shared<PyBayesianNetwork<>>(
                    BayesianNetworkType::keep_python_alive(t), nodes, node_types);
            }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t, const ArcStringVector& arcs) {
                return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), arcs);
            },
            [](std::shared_ptr<BayesianNetworkType> t, const ArcStringVector& arcs) {
                return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t), arcs);
            }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t,
               const ArcStringVector& arcs,
               const FactorTypeVector& node_types) {
                return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), arcs, node_types);
            },
            [](std::shared_ptr<BayesianNetworkType> t,
               const ArcStringVector& arcs,
               const FactorTypeVector& node_types) {
                return std::make_shared<PyBayesianNetwork<>>(
                    BayesianNetworkType::keep_python_alive(t), arcs, node_types);
            }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const ArcStringVector& arcs) {
                return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), nodes, arcs);
            },
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const ArcStringVector& arcs) {
                return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t), nodes, arcs);
            }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const ArcStringVector& arcs,
               const FactorTypeVector& node_types) {
                return std::make_shared<BayesianNetwork>(
                    BayesianNetworkType::keep_python_alive(t), nodes, arcs, node_types);
            },
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const ArcStringVector& arcs,
               const FactorTypeVector& node_types) {
                return std::make_shared<PyBayesianNetwork<>>(
                    BayesianNetworkType::keep_python_alive(t), nodes, arcs, node_types);
            }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t, const Dag& graph) {
                return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), graph);
            },
            [](std::shared_ptr<BayesianNetworkType> t, const Dag& graph) {
                return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t), graph);
            }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t, const Dag& graph, const FactorTypeVector& node_types) {
                return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), graph, node_types);
            },
            [](std::shared_ptr<BayesianNetworkType> t, const Dag& graph, const FactorTypeVector& node_types) {
                return std::make_shared<PyBayesianNetwork<>>(
                    BayesianNetworkType::keep_python_alive(t), graph, node_types);
            }))
        .def("__getstate__", [](const BayesianNetwork& self) { return self.__getstate__(); })
        .def("__setstate__", [](py::object& self, py::tuple& t) {
            // If is a C++ BayesianNetwork
            if (self.get_type().ptr() == py::type::of<BayesianNetwork>().ptr()) {
                models::__nonderived_bn_setstate__<BayesianNetwork>(self, t);
            } else {
                PyBayesianNetwork<>::__setstate__(self, t);
            }
        });

    register_BNGeneric_methods<BayesianNetwork>(bn);

    py::class_<BNGeneric<ConditionalDag>, ConditionalBayesianNetworkBase, std::shared_ptr<BNGeneric<ConditionalDag>>>
        cbn_generic(models, "BNGeneric<ConditionalDag>");
    register_BNGeneric_methods<BNGeneric<ConditionalDag>>(cbn_generic);

    py::class_<ConditionalBayesianNetwork,
               BNGeneric<ConditionalDag>,
               PyConditionalBayesianNetwork<>,
               std::shared_ptr<ConditionalBayesianNetwork>>
        cbn(models, "ConditionalBayesianNetwork");
    cbn.def(py::init(
                [](std::shared_ptr<BayesianNetworkType> t,
                   const std::vector<std::string>& nodes,
                   const std::vector<std::string>& interface_nodes) {
                    return std::make_shared<ConditionalBayesianNetwork>(
                        BayesianNetworkType::keep_python_alive(t), nodes, interface_nodes);
                },
                [](std::shared_ptr<BayesianNetworkType> t,
                   const std::vector<std::string>& nodes,
                   const std::vector<std::string>& interface_nodes) {
                    return std::make_shared<PyConditionalBayesianNetwork<>>(
                        BayesianNetworkType::keep_python_alive(t), nodes, interface_nodes);
                }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const std::vector<std::string>& interface_nodes,
               const FactorTypeVector& node_types) {
                return std::make_shared<ConditionalBayesianNetwork>(
                    BayesianNetworkType::keep_python_alive(t), nodes, interface_nodes, node_types);
            },
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const std::vector<std::string>& interface_nodes,
               const FactorTypeVector& node_types) {
                return std::make_shared<PyConditionalBayesianNetwork<>>(
                    BayesianNetworkType::keep_python_alive(t), nodes, interface_nodes, node_types);
            }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const std::vector<std::string>& interface_nodes,
               const ArcStringVector& arcs) {
                return std::make_shared<ConditionalBayesianNetwork>(
                    BayesianNetworkType::keep_python_alive(t), nodes, interface_nodes, arcs);
            },
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const std::vector<std::string>& interface_nodes,
               const ArcStringVector& arcs) {
                return std::make_shared<PyConditionalBayesianNetwork<>>(
                    BayesianNetworkType::keep_python_alive(t), nodes, interface_nodes, arcs);
            }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const std::vector<std::string>& interface_nodes,
               const ArcStringVector& arcs,
               const FactorTypeVector& node_types) {
                return std::make_shared<ConditionalBayesianNetwork>(
                    BayesianNetworkType::keep_python_alive(t), nodes, interface_nodes, arcs, node_types);
            },
            [](std::shared_ptr<BayesianNetworkType> t,
               const std::vector<std::string>& nodes,
               const std::vector<std::string>& interface_nodes,
               const ArcStringVector& arcs,
               const FactorTypeVector& node_types) {
                return std::make_shared<PyConditionalBayesianNetwork<>>(
                    BayesianNetworkType::keep_python_alive(t), nodes, interface_nodes, arcs, node_types);
            }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t, const ConditionalDag& graph) {
                return std::make_shared<ConditionalBayesianNetwork>(BayesianNetworkType::keep_python_alive(t), graph);
            },
            [](std::shared_ptr<BayesianNetworkType> t, const ConditionalDag& graph) {
                return std::make_shared<PyConditionalBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t),
                                                                        graph);
            }))
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t,
               const ConditionalDag& graph,
               const FactorTypeVector& node_types) {
                return std::make_shared<ConditionalBayesianNetwork>(
                    BayesianNetworkType::keep_python_alive(t), graph, node_types);
            },
            [](std::shared_ptr<BayesianNetworkType> t,
               const ConditionalDag& graph,
               const FactorTypeVector& node_types) {
                return std::make_shared<PyConditionalBayesianNetwork<>>(
                    BayesianNetworkType::keep_python_alive(t), graph, node_types);
            }))
        .def("__getstate__", [](const ConditionalBayesianNetwork& self) { return self.__getstate__(); })
        .def("__setstate__", [](py::object& self, py::tuple& t) {
            // If is a C++ type
            if (self.get_type().ptr() == py::type::of<ConditionalBayesianNetwork>().ptr()) {
                models::__nonderived_bn_setstate__<ConditionalBayesianNetwork>(self, t);
            } else {
                PyBayesianNetwork<ConditionalBayesianNetwork>::__setstate__(self, t);
            }
        });

    py::class_<DynamicBayesianNetwork,
               DynamicBayesianNetworkBase,
               PyDynamicBayesianNetwork<>,
               std::shared_ptr<DynamicBayesianNetwork>>
        dbn(models, "DynamicBayesianNetwork");
    dbn
        .def(py::init(
            [](std::shared_ptr<BayesianNetworkType> t, const std::vector<std::string>& variables, int markovian_order) {
                return std::make_shared<DynamicBayesianNetwork>(
                    BayesianNetworkType::keep_python_alive(t), variables, markovian_order);
            },
            [](std::shared_ptr<BayesianNetworkType> t, const std::vector<std::string>& variables, int markovian_order) {
                return std::make_shared<PyDynamicBayesianNetwork<>>(
                    BayesianNetworkType::keep_python_alive(t), variables, markovian_order);
            }))
        .def(py::init<const std::vector<std::string>&,
                      int,
                      std::shared_ptr<BayesianNetworkBase>,
                      std::shared_ptr<ConditionalBayesianNetworkBase>>())
        .def_property("include_cpd", &DynamicBayesianNetwork::include_cpd, &DynamicBayesianNetwork::set_include_cpd)
        .def("__getstate__", [](const DynamicBayesianNetwork& self) { return self.__getstate__(); })
        .def("__setstate__", [](py::object& self, py::tuple& t) {
            if (self.get_type().ptr() == py::type::of<DynamicBayesianNetwork>().ptr()) {
                models::__nonderived_dbn_setstate__(self, t);
            } else {
                PyDynamicBayesianNetwork<DynamicBayesianNetwork>::__setstate__(self, t);
            }
        });

    register_DerivedBayesianNetwork<GaussianNetwork>(models, "GaussianNetwork");
    auto spbn = register_DerivedBayesianNetwork<SemiparametricBN>(models, "SemiparametricBN");
    spbn.def(py::init<const std::vector<std::string>&, FactorTypeVector&>())
        .def(py::init<const ArcStringVector&, FactorTypeVector&>())
        .def(py::init<const std::vector<std::string>&, const ArcStringVector&, FactorTypeVector&>())
        .def(py::init<const Dag&, FactorTypeVector&>());
    register_DerivedBayesianNetwork<KDENetwork>(models, "KDENetwork");
    register_DerivedBayesianNetwork<DiscreteBN>(models, "DiscreteBN");

    register_DerivedConditionalBayesianNetwork<ConditionalGaussianNetwork>(models, "ConditionalGaussianNetwork");
    auto conditional_spbn =
        register_DerivedConditionalBayesianNetwork<ConditionalSemiparametricBN>(models, "ConditionalSemiparametricBN");
    conditional_spbn
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, FactorTypeVector&>())
        .def(py::init<const std::vector<std::string>&,
                      const std::vector<std::string>&,
                      const ArcStringVector&,
                      FactorTypeVector&>())
        .def(py::init<const ConditionalDag&, FactorTypeVector&>());
    register_DerivedConditionalBayesianNetwork<ConditionalKDENetwork>(models, "ConditionalKDENetwork");
    register_DerivedConditionalBayesianNetwork<ConditionalDiscreteBN>(models, "ConditionalDiscreteBN");

    register_DerivedDynamicBayesianNetwork<DynamicGaussianNetwork>(models, "DynamicGaussianNetwork");
    register_DerivedDynamicBayesianNetwork<DynamicSemiparametricBN>(models, "DynamicSemiparametricBN");
    register_DerivedDynamicBayesianNetwork<DynamicKDENetwork>(models, "DynamicKDENetwork");
    register_DerivedDynamicBayesianNetwork<DynamicDiscreteBN>(models, "DynamicDiscreteBN");
}
