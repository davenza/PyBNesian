#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>
#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>
#include <models/KDENetwork.hpp>
#include <models/DiscreteBN.hpp>
#include <models/HomogeneousBN.hpp>
#include <models/HeterogeneousBN.hpp>
#include <models/CLGNetwork.hpp>
#include <util/util_types.hpp>

using models::BayesianNetworkType, models::GaussianNetworkType, models::SemiparametricBNType, models::KDENetworkType,
    models::DiscreteBNType, models::HomogeneousBNType, models::HeterogeneousBNType, models::CLGNetworkType,
    models::BayesianNetworkBase, models::BNGeneric, models::BayesianNetwork, models::GaussianNetwork,
    models::SemiparametricBN, models::KDENetwork, models::DiscreteBN, models::HomogeneousBN, models::HeterogeneousBN,
    models::CLGNetwork;

using models::MapDataToFactor;

using models::ConditionalBayesianNetworkBase, models::ConditionalBayesianNetwork, models::ConditionalGaussianNetwork,
    models::ConditionalSemiparametricBN, models::ConditionalKDENetwork, models::ConditionalDiscreteBN,
    models::ConditionalHomogeneousBN, models::ConditionalHeterogeneousBN, models::ConditionalCLGNetwork;

using models::DynamicBayesianNetworkBase, models::DynamicBayesianNetwork, models::DynamicGaussianNetwork,
    models::DynamicSemiparametricBN, models::DynamicKDENetwork, models::DynamicDiscreteBN, models::DynamicHomogeneousBN,
    models::DynamicHeterogeneousBN, models::DynamicCLGNetwork;

using util::random_seed_arg;

class PyBayesianNetworkType : public BayesianNetworkType {
public:
    using BayesianNetworkType::BayesianNetworkType;

    PyBayesianNetworkType() { m_hash = reinterpret_cast<std::uintptr_t>(nullptr); }

    bool is_python_derived() const override { return true; }

    std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "new_bn");

        if (override) {
            auto o = override(nodes);

            if (o.is(py::none())) {
                throw std::invalid_argument("BayesianNetworkType::new_bn cannot return None.");
            }

            try {
                auto m = o.cast<std::shared_ptr<BayesianNetworkBase>>();
                BayesianNetworkBase::keep_python_alive(m);
                return m;
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of BayesianNetworkType::new_bn is not a BayesianNetworkBase.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"BayesianNetworkType::new_bn\".");
    }
    std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "new_cbn");

        if (override) {
            auto o = override(nodes, interface_nodes);

            if (o.is(py::none())) {
                throw std::invalid_argument("BayesianNetworkType::new_cbn cannot return None.");
            }

            try {
                auto m = o.cast<std::shared_ptr<ConditionalBayesianNetworkBase>>();
                ConditionalBayesianNetworkBase::keep_python_alive(m);
                return m;
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of BayesianNetworkType::new_cbn is not a ConditionalBayesianNetworkBase.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"BayesianNetworkType::new_cbn\".");
    }

    bool is_homogeneous() const override { PYBIND11_OVERRIDE_PURE(bool, BayesianNetworkType, is_homogeneous, ); }

    std::shared_ptr<FactorType> default_node_type() const override {
        py::gil_scoped_acquire gil;

        py::function override = py::get_override(static_cast<const BayesianNetworkType*>(this), "default_node_type");
        if (override) {
            auto o = override();

            if (o.is(py::none())) {
                throw std::invalid_argument("BayesianNetworkType::default_node_type cannot return None.");
            }

            try {
                auto f = o.cast<std::shared_ptr<FactorType>>();
                FactorType::keep_python_alive(f);
                return f;
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of BayesianNetworkType::default_node_type is not a FactorType.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"BayesianNetworkType::default_node_type\".");
    }

    std::vector<std::shared_ptr<FactorType>> data_default_node_type(
        const std::shared_ptr<DataType>& dt) const override {
        py::gil_scoped_acquire gil;

        py::function override =
            py::get_override(static_cast<const BayesianNetworkType*>(this), "data_default_node_type");
        if (override) {
            auto o = override(dt);

            if (o.is(py::none())) {
                throw std::invalid_argument("BayesianNetworkType::data_default_node_type cannot return None.");
            }

            try {
                auto v = o.cast<std::vector<std::shared_ptr<FactorType>>>();

                for (auto& f : v) {
                    if (f) {
                        FactorType::keep_python_alive(f);
                    } else {
                        throw std::invalid_argument("BayesianNetworkType::data_default_node_type cannot contain None.");
                    }
                }

                return v;
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of BayesianNetworkType::data_default_node_type is not a list of FactorType.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"BayesianNetworkType::data_default_node_type\"");
    }

    bool compatible_node_type(const BayesianNetworkBase& m,
                              const std::string& variable,
                              const std::shared_ptr<FactorType>& nt) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "compatible_node_type");
        if (override) {
            auto o = override(m.shared_from_this(), variable, nt);
            try {
                return o.cast<bool>();
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of BayesianNetworkType::compatible_node_type is not a boolean.");
            }
        }

        return BayesianNetworkType::compatible_node_type(m, variable, nt);
    }

    bool compatible_node_type(const ConditionalBayesianNetworkBase& m,
                              const std::string& variable,
                              const std::shared_ptr<FactorType>& nt) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "compatible_node_type");
        if (override) {
            auto o = override(m.shared_from_this(), variable, nt);
            try {
                return o.cast<bool>();
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of BayesianNetworkType::compatible_node_type is not a boolean.");
            }
        }

        return BayesianNetworkType::compatible_node_type(m, variable, nt);
    }

    bool can_have_arc(const BayesianNetworkBase& m,
                      const std::string& source,
                      const std::string& target) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "can_have_arc");
        if (override) {
            auto o = override(m.shared_from_this(), source, target);
            try {
                return o.cast<bool>();
            } catch (py::cast_error& e) {
                throw std::runtime_error("The returned object of BayesianNetworkType::can_have_arc is not a boolean.");
            }
        }

        return BayesianNetworkType::can_have_arc(m, source, target);
    }

    bool can_have_arc(const ConditionalBayesianNetworkBase& m,
                      const std::string& source,
                      const std::string& target) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override =
            pybind11::get_override(static_cast<const BayesianNetworkType*>(this), "can_have_arc");
        if (override) {
            auto o = override(m.shared_from_this(), source, target);
            try {
                return o.cast<bool>();
            } catch (py::cast_error& e) {
                throw std::runtime_error("The returned object of BayesianNetworkType::can_have_arc is not a boolean.");
            }
        }

        return BayesianNetworkType::can_have_arc(m, source, target);
    }

    std::vector<std::shared_ptr<FactorType>> alternative_node_type(const BayesianNetworkBase& model,
                                                                   const std::string& variable) const override {
        py::gil_scoped_acquire gil;

        py::function override =
            py::get_override(static_cast<const BayesianNetworkType*>(this), "alternative_node_type");
        if (override) {
            auto o = override(model.shared_from_this(), variable);

            if (o.is(py::none())) {
                return std::vector<std::shared_ptr<FactorType>>();
            }

            try {
                auto v = o.cast<std::vector<std::shared_ptr<FactorType>>>();

                for (auto& f : v) {
                    if (f) {
                        FactorType::keep_python_alive(f);
                    } else {
                        throw std::invalid_argument("BayesianNetworkType::alternative_node_type cannot contain None.");
                    }
                }

                return v;
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of BayesianNetworkType::alternative_node_type is not a list of FactorType.");
            }
        }

        return BayesianNetworkType::alternative_node_type(model, variable);
    }

    std::vector<std::shared_ptr<FactorType>> alternative_node_type(const ConditionalBayesianNetworkBase& model,
                                                                   const std::string& variable) const override {
        py::gil_scoped_acquire gil;

        py::function override =
            py::get_override(static_cast<const BayesianNetworkType*>(this), "alternative_node_type");
        if (override) {
            auto o = override(model.shared_from_this(), variable);

            if (o.is(py::none())) {
                return std::vector<std::shared_ptr<FactorType>>();
            }

            try {
                auto v = o.cast<std::vector<std::shared_ptr<FactorType>>>();

                for (auto& f : v) {
                    if (f) {
                        FactorType::keep_python_alive(f);
                    } else {
                        throw std::invalid_argument("BayesianNetworkType::alternative_node_type cannot contain None.");
                    }
                }

                return v;
            } catch (py::cast_error& e) {
                throw std::runtime_error(
                    "The returned object of BayesianNetworkType::alternative_node_type is not a list of FactorType.");
            }
        }

        return BayesianNetworkType::alternative_node_type(model, variable);
    }

    std::string ToString() const override {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, BayesianNetworkType, "__str__", ToString, );
    }

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

    bool is_python_derived() const override { return true; }

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

    void fit(const DataFrame& df, const Arguments& construction_args = Arguments()) override {
        PYBIND11_OVERRIDE_PURE(void, Base, fit, df, construction_args);
    }

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

    std::string ToString() const override { PYBIND11_OVERRIDE_PURE_NAME(std::string, Base, "__str__", ToString, ); }
};

template <typename Base = ConditionalBayesianNetworkBase>
class PyConditionalBayesianNetworkBase : public PyBayesianNetworkBase<Base> {
public:
    using PyBayesianNetworkBase<Base>::PyBayesianNetworkBase;

    int num_interface_nodes() const override { PYBIND11_OVERRIDE_PURE(int, Base, num_interface_nodes, ); }

    int num_joint_nodes() const override { PYBIND11_OVERRIDE_PURE(int, Base, num_joint_nodes, ); }

    const std::vector<std::string>& interface_nodes() const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<std::string>&, Base, interface_nodes, );
    }

    const std::vector<std::string>& joint_nodes() const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<std::string>&, Base, joint_nodes, );
    }

    ArcStringVector interface_arcs() const override { PYBIND11_OVERRIDE_PURE(ArcStringVector, Base, interface_arcs, ); }

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

    bool contains_joint_node(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(bool, Base, contains_joint_node, name);
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

    void fit(const DataFrame& df, const Arguments& construction_args = Arguments()) override {
        PYBIND11_OVERRIDE(void, Base, fit, df, construction_args);
    }

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

        // Initialize the C++ side:
        // The Python constructor ensures that the python objects are kept alive.
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
            Factor::keep_vector_python_alive(cpds);
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

    std::string ToString() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "__str__", ToString, ); }
};

template <typename Base = ConditionalBayesianNetwork>
class PyConditionalBayesianNetwork : public PyBayesianNetwork<Base> {
public:
    using PyBayesianNetwork<Base>::PyBayesianNetwork;

    int num_interface_nodes() const override { PYBIND11_OVERRIDE(int, Base, num_interface_nodes, ); }

    int num_joint_nodes() const override { PYBIND11_OVERRIDE(int, Base, num_joint_nodes, ); }

    const std::vector<std::string>& interface_nodes() const override {
        PYBIND11_OVERRIDE(const std::vector<std::string>&, Base, interface_nodes, );
    }

    const std::vector<std::string>& joint_nodes() const override {
        PYBIND11_OVERRIDE(const std::vector<std::string>&, Base, joint_nodes, );
    }

    ArcStringVector interface_arcs() const override { PYBIND11_OVERRIDE(ArcStringVector, Base, interface_arcs, ); }

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

    bool contains_joint_node(const std::string& name) const override {
        PYBIND11_OVERRIDE(bool, Base, contains_joint_node, name);
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

    void fit(const DataFrame& df, const Arguments& construction_args = Arguments()) override {
        PYBIND11_OVERRIDE_PURE(void, Base, fit, df, construction_args);
    }

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

    std::string ToString() const override { PYBIND11_OVERRIDE_PURE_NAME(std::string, Base, "__str__", ToString, ); }
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

    void fit(const DataFrame& df, const Arguments& construction_args = Arguments()) override {
        PYBIND11_OVERRIDE(void, Base, fit, df, construction_args);
    }

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

    std::string ToString() const override { PYBIND11_OVERRIDE_NAME(std::string, Base, "__str__", ToString, ); }

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
    m.def_property("include_cpd", &CppClass::include_cpd, &CppClass::set_include_cpd, R"doc(
This property indicates if the factors of the Bayesian network model should be saved when ``__getstate__`` is called.
)doc")
        .def("fitted", &CppClass::fitted, R"doc(
Checks whether the model is fitted.

:returns: True if the model is fitted, False otherwise.
)doc")
        .def("num_nodes", &CppClass::num_nodes, R"doc(
Gets the number of nodes.

:returns: Number of nodes.
)doc")
        .def("num_arcs", &CppClass::num_arcs, R"doc(
Gets the number of arcs.

:returns: Number of arcs.
)doc")
        .def("nodes", &CppClass::nodes, py::return_value_policy::reference_internal, R"doc(
Gets the nodes of the Bayesian network.

:returns: Nodes of the Bayesian network.
)doc")
        .def("arcs", &CppClass::arcs, py::return_value_policy::take_ownership, R"doc(
Gets the list of arcs.

:returns: A list of tuples (source, target) representing an arc source -> target.
)doc")
        .def("indices", &CppClass::indices, py::return_value_policy::reference_internal, R"doc(
Gets all the indices in the graph.

:returns: A dictionary with the index of each node.
)doc")
        .def("index", &CppClass::index, py::arg("node"), R"doc(
Gets the index of a node from its name.

:param node: Name of the node.
:returns: Index of the node.
)doc")
        .def("collapsed_index", &CppClass::collapsed_index, py::arg("node"), R"doc(
Gets the collapsed index of a node from its name.

:param node: Name of the node.
:returns: Collapsed index of the node.
)doc")
        .def("index_from_collapsed", &CppClass::index_from_collapsed, py::arg("collapsed_index"), R"doc(
Gets the index of a node from its collapsed index.

:param collapsed_index: Collapsed index of the node.
:returns: Index of the node.
)doc")
        .def("collapsed_from_index", &CppClass::collapsed_from_index, py::arg("index"), R"doc(
Gets the collapsed index of a node from its index.

:param index: Index of the node.
:returns: Collapsed index of the node.
)doc")
        .def("collapsed_indices", &CppClass::collapsed_indices, py::return_value_policy::reference_internal, R"doc(
Gets all the collapsed indices for the nodes in the graph.

:returns: A dictionary with the collapsed index of each node.
)doc")
        .def("is_valid", &CppClass::is_valid, py::arg("node"), R"doc(
Checks whether a node is valid (the node is not removed).

:param node: Node name.
:returns: True if the node is valid, False otherwise.
)doc")
        .def("contains_node", &CppClass::contains_node, py::arg("node"), R"doc(
Tests whether the node is in the Bayesian network or not.

:param node: Name of the node.
:returns: True if the Bayesian network contains the node, False otherwise.
)doc")
        .def("add_node", &CppClass::add_node, py::arg("node"), R"doc(
Adds a node to the Bayesian network and returns its index.

:param node: Name of the new node.
:returns: Index of the new node.
)doc")
        .def("remove_node", &CppClass::remove_node, py::arg("node"), R"doc(
Removes a node.

:param node: A node name.
)doc")
        .def("name", &CppClass::name, py::arg("index"), R"doc(
Gets the name of a node from its index.

:param index: Index of the node.
:returns: Name of the node.
)doc")
        .def("collapsed_name", &CppClass::collapsed_name, py::arg("collapsed_index"), R"doc(
Gets the name of a node from its collapsed index.

:param collapsed_index: Collapsed index of the node.
:returns: Name of the node.
)doc")
        .def("num_parents", &CppClass::num_parents, py::arg("node"), R"doc(
Gets the number of parent nodes of a node.

:param node: A node name.
:returns: Number of parent nodes.
)doc")
        .def("num_children", &CppClass::num_children, py::arg("node"), R"doc(
Gets the number of children nodes of a node.

:param node: A node name.
:returns: Number of children nodes.
)doc")
        .def("parents", &CppClass::parents, py::return_value_policy::take_ownership, py::arg("node"), R"doc(
Gets the parent nodes of a node.

:param node: A node name.
:returns: Parent node names.
)doc")
        .def("children", &CppClass::children, py::return_value_policy::take_ownership, py::arg("node"), R"doc(
Gets the children nodes of a node.

:param node: A node name.
:returns: Children node names.
)doc")
        .def("has_arc", &CppClass::has_arc, py::arg("source"), py::arg("target"), R"doc(
Checks whether an arc between the nodes ``source`` and ``target`` exists.

:param source: A node name.
:param target: A node name.
:returns: True if the arc exists, False otherwise.
)doc")
        .def("has_path", &CppClass::has_path, py::arg("n1"), py::arg("n2"), R"doc(
Checks whether there is a directed path between nodes ``n1`` and ``n2``.

:param n1: A node name.
:param n2: A node name.
:returns: True if there is an directed path between ``n1`` and ``n2``, False otherwise.
)doc")
        .def("add_arc", &CppClass::add_arc, py::arg("source"), py::arg("target"), R"doc(
Adds an arc between the nodes ``source`` and ``target``. If the arc already exists, the graph is left unaffected.

:param source: A node name.
:param target: A node name.
)doc")
        .def("remove_arc", &CppClass::remove_arc, py::arg("source"), py::arg("target"), R"doc(
Removes an arc between the nodes ``source`` and ``target``. If the arc do not exist, the graph is left unaffected.

:param source: A node name.
:param target: A node name.
)doc")
        .def("flip_arc", &CppClass::flip_arc, py::arg("source"), py::arg("target"), R"doc(
Flips (reverses) an arc between the nodes ``source`` and ``target``. If the arc do not exist, the graph is left
unaffected.

:param source: A node name.
:param target: A node name.
)doc")
        .def("can_add_arc", &CppClass::can_add_arc, py::arg("source"), py::arg("target"), R"doc(
Checks whether an arc between the nodes ``source`` and ``target`` can be added.

An arc addition can be not allowed for multiple reasons:

- It generates a cycle.
- It is a conditional BN and both source and target are interface nodes.
- It is not allowed by the :class:`BayesianNetworkType`.

:param source: A node name.
:param target: A node name.
:returns: True if the arc can be added, False otherwise.
)doc")
        .def("can_flip_arc", &CppClass::can_flip_arc, py::arg("source"), py::arg("target"), R"doc(
Checks whether an arc between the nodes ``source`` and ``target`` can be flipped.

An arc flip can be not allowed for multiple reasons:

- It generates a cycle.
- It is not allowed by the :class:`BayesianNetworkType`.

:param source: A node name.
:param target: A node name.
:returns: True if the arc can be added, False otherwise.
)doc")
        .def("force_whitelist", &CppClass::force_whitelist, py::arg("arc_whitelist"), R"doc(
Include the given whitelisted arcs. It checks the validity of the graph after including the arc whitelist.

:param arc_whitelist: List of arcs tuples (``source``, ``target``) that must be added to the graph.
)doc")
        .def("cpd", py::overload_cast<const std::string&>(&CppClass::cpd), py::arg("node"), R"doc(
Returns the conditional probability distribution (CPD) associated to ``node``. This is a
:class:`Factor <pybnesian.Factor>` type.

:param node: A node name.
:returns: The :class:`Factor <pybnesian.Factor>` associated to ``node``
:raises ValueError: If ``node`` do not have an associated :class:`Factor <pybnesian.Factor>` yet.
)doc")
        .def(
            "add_cpds",
            [](CppClass& self, const std::vector<std::shared_ptr<Factor>>& cpds) {
                self.add_cpds(Factor::keep_vector_python_alive(cpds));
            },
            py::arg("cpds"),
            R"doc(
Adds a list of CPDs to the Bayesian network. The list may be complete (for all the nodes all the Bayesian network) or
partial (just some a subset of the nodes).

:param cpds: List of :class:`Factor <pybnesian.Factor>`.
)doc")
        .def("fit", &CppClass::fit, py::arg("df"), py::arg("construction_args") = Arguments(), R"doc(
Fit all the unfitted :class:`Factor <pybnesian.Factor>` with the data ``df``.

:param df: DataFrame to fit the Bayesian network.
:param construction_args: Additional arguments provided to construct the :class:`Factor <pybnesian.Factor>`.
)doc")
        .def("logl",
             &CppClass::logl,
             py::return_value_policy::take_ownership,
             py::arg("df"),
             R"doc(
Returns the log-likelihood of each instance in the DataFrame ``df``. This returns the sum of the log-likelihood for all
the factors in the Bayesian network.

:param df: DataFrame to compute the log-likelihood.
:returns: A :class:`numpy.ndarray` vector with dtype :class:`numpy.float64`, where the i-th value is the log-likelihod
          of the i-th instance of ``df``.
)doc")
        .def("slogl", &CppClass::slogl, py::arg("df"), R"doc(
Returns the sum of the log-likelihood of each instance in the DataFrame ``df``. That is, the sum of the result of
:func:`BayesianNetworkBase.logl`.

:param df: DataFrame to compute the sum of the log-likelihood.
:returns: The sum of log-likelihood for DataFrame ``df``.
)doc")
        .def("type", &CppClass::type, R"doc(
Gets the underlying :class:`BayesianNetworkType`.

:returns: The :class:`BayesianNetworkType` of ``self``.
)doc")
        .def(
            "sample",
            [](const CppClass& self, int n, std::optional<unsigned int> seed, bool ordered) {
                return self.sample(n, random_seed_arg(seed), ordered);
            },
            py::return_value_policy::move,
            py::arg("n"),
            py::arg("seed") = std::nullopt,
            py::arg("ordered") = false,
            R"doc(
Samples ``n`` values from this BayesianNetwork. This method returns a :class:`pyarrow.RecordBatch` with ``n`` instances.

If ``ordered`` is True, it orders the columns according to the list :func:`BayesianNetworkBase.nodes`. Else, it
orders the columns according to a topological sort.

:param n: Number of instances to sample.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
:param ordered: If True, order the columns according to :func:`BayesianNetworkBase.nodes`.
:returns: A DataFrame with ``n`` instances that contains the sampled data.
)doc")
        .def("conditional_bn", py::overload_cast<>(&CppClass::conditional_bn, py::const_), R"doc(
Returns the conditional Bayesian network version of this Bayesian network.

- If ``self`` is not conditional, it returns a conditional version of the Bayesian network where the graph is
  transformed using :func:`Dag.conditional_graph <pybnesian.Dag.conditional_graph>`.
- If ``self`` is conditional, it returns a copy of ``self``.

:returns: The conditional graph transformation of ``self``.
)doc")
        .def("conditional_bn",
             py::overload_cast<const std::vector<std::string>&, const std::vector<std::string>&>(
                 &CppClass::conditional_bn, py::const_),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             R"doc(
Returns the conditional Bayesian network version of this Bayesian network.

- If ``self`` is not conditional, it returns a conditional version of the Bayesian network where the graph is
  transformed using :func:`Dag.conditional_graph <pybnesian.Dag.conditional_graph>` using the given set of nodes
  and interface nodes.
- If ``self`` is conditional, it returns a copy of ``self``.

:returns: The conditional graph transformation of ``self``.
)doc")
        .def("unconditional_bn", &CppClass::unconditional_bn, R"doc(
Returns the unconditional Bayesian network version of this Bayesian network.

- If ``self`` is not conditional, it returns a copy of ``self``.
- If ``self`` is conditional, the interface nodes are included as nodes in the returned Bayesian network.

:returns: The unconditional graph transformation of ``self``.
)doc")
        .def("save", &CppClass::save, py::arg("filename"), py::arg("include_cpd") = false, R"doc(
Saves the Bayesian network in a pickle file with the given name. If ``include_cpd`` is True, it also saves the
conditional probability distributions (CPDs) in the Bayesian network.

:param filename: File name of the saved Bayesian network.
:param include_cpd: Include the CPDs.
)doc")
        .def("node_type", &CppClass::node_type, py::arg("node"), R"doc(
Gets the corresponding :class:`FactorType <pybnesian.FactorType>` for ``node``.

:param node: A node name.
:returns: The :class:`FactorType <pybnesian.FactorType>` of ``node``.
)doc")
        .def("node_types", &CppClass::node_types, R"doc(
Gets the :class:`FactorType <pybnesian.FactorType>` for all the nodes.

:returns: The corresponding :class:`FactorType <pybnesian.FactorType>` for each node.
)doc")
        .def("underlying_node_type", &CppClass::underlying_node_type, py::arg("df"), py::arg("node"), R"doc(
Gets the underlying :class:`FactorType <pybnesian.FactorType>` for a given node type.

1) If the node has a node type different from :class:`UnknownFactorType <pybnesian.UnknownFactorType>`, it returns it.
2) Else, it returns the first default node type from
   :func:`BayesianNetworkType.data_default_node_type <pybnesian.BayesianNetworkType.data_default_node_type>`.

:param df: Data to extract the underlying node type (if 2) is required).
:param node: A node name.
:returns: The underlying :class:`FactorType <pybnesian.FactorType>` for each node.
)doc")
        .def("has_unknown_node_types", &CppClass::has_unknown_node_types, R"doc(
Checks whether there are nodes with an unknown node type (i.e.
:class:`UnknownFactorType <pybnesian.UnknownFactorType>`).

:returns: True if there are nodes with an unkown node type, False otherwise.
)doc")
        .def("set_unknown_node_types",
             &CppClass::set_unknown_node_types,
             py::arg("df"),
             py::arg("type_blacklist") = FactorTypeVector(),
             R"doc(
Changes the unknown node types (i.e. the nodes with :class:`UnknownFactorType <pybnesian.UnknownFactorType>`) to
the default node types specified by the :class:`BayesianNetworkType`. If a :class:`FactorType <pybnesian.FactorType>` is blacklisted for a given node, the next element in the :func:`BayesianNetworkType.data_default_node_type() <pybnesian.BayesianNetworkType.data_default_node_type>` list is used as the default :class:`FactorType <pybnesian.FactorType>`.

:param df: DataFrame to get the default node type for each unknown node type.
:param type_blacklist: List of type blacklist (forbidden :class:`FactorType <pybnesian.FactorType>`).
)doc")
        .def(
            "set_node_type",
            [](CppClass& self, const std::string& node, const std::shared_ptr<FactorType>& new_type) {
                self.set_node_type(node, FactorType::keep_python_alive(new_type));
            },
            py::arg("node"),
            py::arg("new_type"),
            R"doc(
Sets the ``new_type`` :class:`FactorType <pybnesian.FactorType>` for ``node``.

:param node: A node name.
:param new_type: The new :class:`FactorType <pybnesian.FactorType>` for ``node``.
)doc")
        .def(
            "force_type_whitelist",
            [](CppClass& self, const FactorTypeVector& type_whitelist) {
                self.force_type_whitelist(util::keep_FactorTypeVector_python_alive(type_whitelist));
            },
            py::arg("type_whitelist"),
            R"doc(
Forces the Bayesian network to have the given whitelisted node types.

:param type_whitelist: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                       specifies the whitelisted type for each node.
)doc")
        .def("clone", &CppClass::clone, R"doc(
Clones (copies) this Bayesian network.

:returns: A copy of ``self``.
)doc")
        .def("__str__", &CppClass::ToString);
}

template <typename CppClass, typename Class>
void register_ConditionalBayesianNetwork_methods(Class& m) {
    m.def("num_interface_nodes", &CppClass::num_interface_nodes, R"doc(
Gets the number of interface nodes.

:returns: Number of interface nodes.
)doc")
        .def("num_joint_nodes", &CppClass::num_joint_nodes, R"doc(
Gets the number of joint nodes. That is, ``num_nodes() + num_interface_nodes()``

:returns: Number of joint nodes.
)doc")
        .def("interface_nodes", &CppClass::interface_nodes, py::return_value_policy::reference_internal, R"doc(
Gets the interface nodes of the Bayesian network.

:returns: Interface nodes of the Bayesian network.
)doc")
        .def("joint_nodes", &CppClass::joint_nodes, py::return_value_policy::reference_internal, R"doc(
Gets the joint set of nodes of the Bayesian network.

:returns: Joint set of nodes of the Bayesian network.
)doc")
        .def("interface_arcs", &CppClass::interface_arcs, py::return_value_policy::reference_internal, R"doc(
Gets the arcs where the source node is an interface node.

:returns: arcs with an interface node as source node.
)doc")
        .def("interface_collapsed_indices",
             &CppClass::interface_collapsed_indices,
             py::return_value_policy::reference_internal,
             R"doc(
Gets all the interface collapsed indices for the interface nodes in the graph.

:returns: A dictionary with the interface collapsed index of each interface node.
)doc")
        .def("joint_collapsed_indices",
             &CppClass::joint_collapsed_indices,
             py::return_value_policy::reference_internal,
             R"doc(
Gets all the joint collapsed indices for the joint set of nodes in the graph.

:returns: A dictionary with the joint collapsed index of each joint node.
)doc")
        .def("interface_collapsed_index", &CppClass::interface_collapsed_index, py::arg("node"), R"doc(
Gets the interface collapsed index of an interface node from its name.

:param node: Name of the interface node.
:returns: Interface collapsed index of the interface node.
)doc")
        .def("joint_collapsed_index", &CppClass::joint_collapsed_index, py::arg("node"), R"doc(
Gets the joint collapsed index of a node from its name.

:param node: Name of the node.
:returns: Joint collapsed index of the node.
)doc")
        .def("index_from_interface_collapsed",
             &CppClass::index_from_interface_collapsed,
             py::arg("collapsed_index"),
             R"doc(
Gets the index of a node from the interface collapsed index.

:param collapsed_index: Interface collapsed index of the node.
:returns: Index of the node.
)doc")
        .def("index_from_joint_collapsed",
             &CppClass::index_from_joint_collapsed,
             py::arg("collapsed_index"),
             R"doc(
Gets the index of a node from the joint collapsed index.

:param collapsed_index: Joint collapsed index of the node.
:returns: Index of the node.
)doc")
        .def("interface_collapsed_from_index", &CppClass::interface_collapsed_from_index, py::arg("index"), R"doc(
Gets the interface collapsed index of a node from its index.

:param index: Index of the node.
:returns: Interface collapsed index of the node.
)doc")
        .def("joint_collapsed_from_index", &CppClass::joint_collapsed_from_index, py::arg("index"), R"doc(
Gets the joint collapsed index of a node from its index.

:param index: Index of the node.
:returns: Joint collapsed index of the node.
)doc")
        .def("interface_collapsed_name", &CppClass::interface_collapsed_name, py::arg("collapsed_index"), R"doc(
Gets the name of an interface node from its collapsed index.

:param collapsed_index: Collapsed index of the interface node.
:returns: Name of the interface node.
)doc")
        .def("joint_collapsed_name", &CppClass::joint_collapsed_name, py::arg("collapsed_index"), R"doc(
Gets the name of a node from its joint collapsed index.

:param collapsed_index: Joint collapsed index of the node.
:returns: Name of the node.
)doc")
        .def("contains_interface_node", &CppClass::contains_interface_node, py::arg("node"), R"doc(
Tests whether the interface node is in the Bayesian network or not.

:param node: Name of the node.
:returns: True if the Bayesian network contains the interface node, False otherwise.
)doc")
        .def("contains_joint_node", &CppClass::contains_joint_node, py::arg("node"), R"doc(
Tests whether the node is in the joint set of nodes or not.

:param node: Name of the node.
:returns: True if the node is in the joint set of nodes, False otherwise.
)doc")
        .def("add_interface_node", &CppClass::add_interface_node, py::arg("node"), R"doc(
Adds an interface node to the Bayesian network and returns its index.

:param node: Name of the new interface node.
:returns: Index of the new interface node.
)doc")
        .def("remove_interface_node", &CppClass::remove_interface_node, py::arg("node"), R"doc(
Removes an interface node.

:param node: A node name.
)doc")
        .def("is_interface", &CppClass::is_interface, py::arg("node"), R"doc(
Checks whether the ``node`` is an interface node.

:param node: A node name.
:returns: True if ``node`` is interface node, False, otherwise.
)doc")
        .def("set_interface", &CppClass::set_interface, py::arg("node"), R"doc(
Converts a normal node into an interface node.

:param node: A node name.
)doc")
        .def("set_node", &CppClass::set_node, py::arg("node"), R"doc(
Converts an interface node into a normal node.

:param node: A node name.
)doc")
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
            py::arg("ordered") = false,
            R"doc(
Samples ``n`` values from this conditional BayesianNetwork conditioned on ``evidence``. ``evidence`` must contain a
column for each interface node. This method returns a :class:`pyarrow.RecordBatch` with ``n`` instances.

If ``concat`` is True, it concatenates ``evidence`` in the result.

If ``ordered`` is True, it orders the columns according to the list :func:`BayesianNetworkBase.nodes`. Else, it
orders the columns according to a topological sort.

:param n: Number of instances to sample.
:param evidence: A DataFrame of ``n`` instances to condition the sampling.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
:param ordered: If True, order the columns according to :func:`BayesianNetworkBase.nodes`.
:returns: A DataFrame with ``n`` instances that contains the sampled data.
)doc")
        .def("clone", &CppClass::clone, R"doc(
Clones (copies) this Bayesian network.

:returns: A copy of ``self``.
)doc");
}

template <typename CppClass, typename Class>
void register_BNGeneric_methods(Class& m) {
    m.def("graph", py::overload_cast<>(&CppClass::graph), R"doc(
Gets the underlying graph of the Bayesian network.

:returns: Graph of the Bayesian network.
)doc")
        .def("can_have_cpd", &CppClass::can_have_cpd, py::arg("node"), R"doc(
Checks whether a given node name can have an associated CPD. For

:param node: A node name.
:returns: True if the given node can have a CPD, False otherwise.
)doc")
        .def("check_compatible_cpd", &CppClass::check_compatible_cpd, py::arg("cpd"), R"doc(
Checks whether the given CPD is compatible with this Bayesian network.

:param cpd: A :class:`Factor <pybnesian.Factor>`.
:returns: True if ``cpd`` is compatible with this Bayesian network, False otherwise.
)doc");
}

template <typename CppClass, typename Class>
void register_DynamicBayesianNetwork_methods(Class& m) {
    m.def("fitted", &CppClass::fitted, R"doc(
Checks whether the model is fitted.

:returns: True if the model is fitted, False otherwise.
)doc")
        .def("static_bn", py::overload_cast<>(&CppClass::static_bn), py::return_value_policy::reference_internal, R"doc(
Returns the static Bayesian network.

:returns: Static Bayesian network.
)doc")
        .def("transition_bn",
             py::overload_cast<>(&CppClass::transition_bn),
             py::return_value_policy::reference_internal,
             R"doc(
Returns the transition Bayesian network.

:returns: Transition Bayesian network.
)doc")
        .def("markovian_order", &CppClass::markovian_order, R"doc(
Gets the markovian order of the dynamic Bayesian network.

:returns: markovian order of this dynamic Bayesian network.
)doc")
        .def("num_variables", &CppClass::num_variables, R"doc(
Gets the number of variables.

:returns: Number of variables.
)doc")
        .def("variables", &CppClass::variables, R"doc(
Gets the variables of the dynamic Bayesian network.

:returns: Variables of the dynamic Bayesian network.
)doc")
        .def("contains_variable", &CppClass::contains_variable, py::arg("variable"), R"doc(
Tests whether the variable is in the dynamic Bayesian network or not.

:param variable: Name of the variable.
:returns: True if the dynamic Bayesian network contains the variable, False otherwise.
)doc")
        .def("add_variable", &CppClass::add_variable, py::arg("variable"), R"doc(
Adds a variable to the dynamic Bayesian network. It adds a node for each temporal slice in the static and transition
Bayesian networks.

:param variable: Name of the new variable.
)doc")
        .def("remove_variable", &CppClass::remove_variable, py::arg("variable"), R"doc(
Removes a variable. It removes all the corresponding nodes in the static and transition Bayesian networks.

:param variable: A variable name.
)doc")
        .def("fit", &CppClass::fit, py::arg("df"), py::arg("construction_args") = Arguments(), R"doc(
Fit all the unfitted :class:`Factor <pybnesian.Factor>` with the data ``df`` in both the static and transition
Bayesian networks.

:param df: DataFrame to fit the dynamic Bayesian network.
:param construction_args: Additional arguments provided to construct the :class:`Factor <pybnesian.Factor>`.
)doc")
        .def("logl",
             &CppClass::logl,
             py::arg("df"),
             R"doc(
Returns the log-likelihood of each instance in the DataFrame ``df``.

:param df: DataFrame to compute the log-likelihood.
:returns: A :class:`numpy.ndarray` vector with dtype :class:`numpy.float64`, where the i-th value is the log-likelihood
          of the i-th instance of ``df``.
)doc")
        .def("slogl", &CppClass::slogl, py::arg("df"), R"doc(
Returns the sum of the log-likelihood of each instance in the DataFrame ``df``. That is, the sum of the result of
:func:`DynamicBayesianNetworkBase.logl`.

:param df: DataFrame to compute the sum of the log-likelihood.
:returns: The sum of log-likelihood for DataFrame ``df``.
)doc")
        .def("type", &CppClass::type, R"doc(
Gets the underlying :class:`BayesianNetworkType`.

:returns: The :class:`BayesianNetworkType` of ``self``.
)doc")
        .def(
            "sample",
            [](const CppClass& self, int n, std::optional<unsigned int> seed) {
                return self.sample(n, random_seed_arg(seed));
            },
            py::arg("n"),
            py::arg("seed") = std::nullopt,
            R"doc(
Samples ``n`` values from this dynamic Bayesian network. This method returns a :class:`pyarrow.RecordBatch` with ``n``
instances.

:param n: Number of instances to sample.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
)doc")
        .def("save", &CppClass::save, py::arg("filename"), py::arg("include_cpd") = false, R"doc(
Saves the dynamic Bayesian network in a pickle file with the given name. If ``include_cpd`` is True, it also saves the
conditional probability distributions (CPDs) in the dynamic Bayesian network.

:param filename: File name of the saved dynamic Bayesian network.
:param include_cpd: Include the CPDs.
)doc")
        .def("__str__", &CppClass::ToString);
}

template <typename DerivedBN>
py::class_<DerivedBN, BayesianNetwork, std::shared_ptr<DerivedBN>> register_DerivedBayesianNetwork(
    py::module& m, const char* derivedbn_name, const char* docs) {
    auto derived_str = std::string(derivedbn_name);

    auto n_ctr = R"doc(
Initializes the :class:`)doc" +
                 derived_str + R"doc(` with the given ``nodes``.

:param nodes: List of node names.
)doc";

    auto a_ctr = R"doc(
Initializes the :class:`)doc" +
                 derived_str + R"doc(` with the given ``arcs`` (the nodes are extracted from the arcs).

:param arcs: Arcs of the :class:`)doc" +
                 derived_str + "`.";

    auto na_ctr = R"doc(
Initializes the :class:`)doc" +
                  derived_str + R"doc(` with the given ``nodes`` and ``arcs``.

:param nodes: List of node names.
:param arcs: Arcs of the :class:`)doc" +
                  derived_str + "`.";

    auto g_ctr = R"doc(
Initializes the :class:`)doc" +
                 derived_str + R"doc(` with the given ``graph``.

:param graph: :class:`Dag <pybnesian.Dag>` of the Bayesian network.)doc";

    return py::class_<DerivedBN, BayesianNetwork, std::shared_ptr<DerivedBN>>(m, derivedbn_name, docs)
        .def(py::init<const std::vector<std::string>&>(), py::arg("nodes"), n_ctr.c_str())
        .def(py::init<const ArcStringVector&>(), py::arg("arcs"), a_ctr.c_str())
        .def(py::init<const std::vector<std::string>&, const ArcStringVector&>(),
             py::arg("nodes"),
             py::arg("arcs"),
             na_ctr.c_str())
        .def(py::init<const Dag&>(), py::arg("graph"), g_ctr.c_str())
        .def(py::pickle([](const DerivedBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__derived_bn_setstate__<DerivedBN>(t); }));
}

template <typename DerivedBN>
py::class_<DerivedBN, ConditionalBayesianNetwork, std::shared_ptr<DerivedBN>>
register_DerivedConditionalBayesianNetwork(py::module& m, const char* derivedbn_name, const char* docs) {
    auto derived_str = std::string(derivedbn_name);

    auto n_ctr = R"doc(
Initializes the :class:`)doc" +
                 derived_str + R"doc(` with the given ``nodes`` and ``interface_nodes``.

:param nodes: List of node names.
:param interface_nodes: List of interface node names.
)doc";

    auto na_ctr = R"doc(
Initializes the :class:`)doc" +
                  derived_str + R"doc(` with the given ``nodes``, ``interface_nodes`` and ``arcs``.

:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param arcs: Arcs of the :class:`)doc" +
                  derived_str + "`.";

    auto g_ctr = R"doc(
Initializes the :class:`)doc" +
                 derived_str + R"doc(` with the given ``graph``.

:param graph: :class:`ConditionalDag <pybnesian.ConditionalDag>` of the conditional Bayesian network.)doc";

    return py::class_<DerivedBN, ConditionalBayesianNetwork, std::shared_ptr<DerivedBN>>(m, derivedbn_name, docs)
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             n_ctr.c_str())
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, const ArcStringVector&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             na_ctr.c_str())
        .def(py::init<const ConditionalDag&>(), py::arg("graph"), g_ctr.c_str())
        .def(py::pickle([](const DerivedBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__derived_bn_setstate__<DerivedBN>(t); }));
}

template <typename DerivedBN>
py::class_<DerivedBN, DynamicBayesianNetwork, std::shared_ptr<DerivedBN>> register_DerivedDynamicBayesianNetwork(
    py::module& m, const char* derivedbn_name, const char* docs) {
    auto derived_str = std::string(derivedbn_name);
    auto empty_ctr = R"doc(
Initializes the :class:`)doc" +
                     derived_str + R"doc(` with the given ``variables`` and ``markovian_order``. It creates
empty static and transition Bayesian networks.

:param variables: List of variable names.
:param markovian_order: Markovian order of the dynamic Bayesian network.
)doc";

    auto networks_ctr = R"doc(
Initializes the :class:`)doc" +
                        derived_str + R"doc(` with the given ``variables`` and ``markovian_order``. The static
and transition Bayesian networks are initialized with ``static_bn`` and ``transition_bn`` respectively.

Both ``static_bn`` and ``transition_bn`` must contain the expected nodes:

- For the static network, it must contain the nodes from ``[variable_name]_t_1`` to
  ``[variable_name]_t_[markovian_order]``.
- For the transition network, it must contain the nodes ``[variable_name]_t_0``, and the interface nodes from
  ``[variable_name]_t_1`` to ``[variable_name]_t_[markovian_order]``.

:param variables: List of variable names.
:param markovian_order: Markovian order of the dynamic Bayesian network.
:param static_bn: Static Bayesian network.
:param transition_bn: Transition Bayesian network.
)doc";

    return py::class_<DerivedBN, DynamicBayesianNetwork, std::shared_ptr<DerivedBN>>(m, derivedbn_name, docs)
        .def(py::init<const std::vector<std::string>&, int>(),
             py::arg("variables"),
             py::arg("markovian_order"),
             empty_ctr.c_str())
        .def(py::init<const std::vector<std::string>&,
                      int,
                      std::shared_ptr<BayesianNetworkBase>,
                      std::shared_ptr<ConditionalBayesianNetworkBase>>(),
             py::arg("variables"),
             py::arg("markovian_order"),
             py::arg("static_bn"),
             py::arg("transition_bn"),
             networks_ctr.c_str())
        .def(py::pickle([](const DerivedBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__derived_dbn_setstate__<DerivedBN>(t); }));
}

void pybindings_models(py::module& root) {
    py::class_<BayesianNetworkType, PyBayesianNetworkType, std::shared_ptr<BayesianNetworkType>> bn_type(
        root, "BayesianNetworkType", R"doc(
A representation of a :class:`BayesianNetwork` that defines its behaviour.
)doc");

    py::class_<BayesianNetworkBase, PyBayesianNetworkBase<>, std::shared_ptr<BayesianNetworkBase>> bn_base(
        root, "BayesianNetworkBase", R"doc(
This class defines an interface of base operations for all the Bayesian networks.

It reproduces many of the methods in the underlying graph to perform additional initializations and simplify the access.
See :ref:`graph-ref`.
)doc");

    py::class_<ConditionalBayesianNetworkBase,
               BayesianNetworkBase,
               PyConditionalBayesianNetworkBase<>,
               std::shared_ptr<ConditionalBayesianNetworkBase>>
        cbn_base(root, "ConditionalBayesianNetworkBase", R"doc(
This class defines an interface of base operations for the conditional Bayesian networks.

It includes some methods of the :class:`ConditionalDag <pybnesian.ConditionalDag>` to simplify the access to the
graph.
)doc");

    bn_type.def(py::init<>(), R"doc(Initializes a new :class:`BayesianNetworkType`)doc")
        .def("is_homogeneous", &BayesianNetworkType::is_homogeneous, R"doc(
Checks whether the Bayesian network is homogeneous.

A Bayesian network is homogeneous if the :class:`FactorType <pybnesian.FactorType>` of all the nodes are forced
to be the same: for example, a Gaussian network is homogeneous because the
:class:`FactorType <pybnesian.FactorType>` type of each node is always
:class:`LinearGaussianCPDType <pybnesian.LinearGaussianCPDType>`.

:returns: True if the Bayesian network is homogeneous, False otherwise.
)doc")
        .def("default_node_type", &BayesianNetworkType::default_node_type, R"doc(
Returns the default :class:`FactorType <pybnesian.FactorType>` of each node in this Bayesian network type. 
This method is only needed for homogeneous Bayesian networks and returns the unique possible
:class:`FactorType <pybnesian.FactorType>`.

:returns: default :class:`FactorType <pybnesian.FactorType>` for the nodes.
)doc")
        .def("data_default_node_type",
             &BayesianNetworkType::data_default_node_type,
             R"doc(
Returns a list of default :class:`FactorType <pybnesian.FactorType>` for the nodes of this Bayesian network type with
data type ``datatype``. This method is only needed for non-homogeneous Bayesian networks and defines the priority of use of the different :class:`FactorType <pybnesian.FactorType>` for the given ``datatype``. If a :class:`FactorType <pybnesian.FactorType>` is blacklisted for a given node, the next element in the list is used as the default :class:`FactorType <pybnesian.FactorType>`. See also :func:`BayesianNetworkBase.set_unknown_node_types() <pybnesian.BayesianNetworkBase.set_unknown_node_types>`.

:param datatype: :class:`pyarrow.DataType` defining the type of data for a node.
:returns: List of default :class:`FactorType <pybnesian.FactorType>` for a node given the ``datatype``.
)doc",
             py::arg("datatype"))
        .def("new_bn", &BayesianNetworkType::new_bn, py::arg("nodes"), R"doc(
Returns an empty unconditional Bayesian network of this type with the given ``nodes``.

:param nodes: Nodes of the new Bayesian network.
:returns: A new empty unconditional Bayesian network.
)doc")
        .def("new_cbn", &BayesianNetworkType::new_cbn, py::arg("nodes"), py::arg("interface_nodes"), R"doc(
Returns an empty conditional Bayesian network of this type with the given ``nodes`` and
``interface_nodes``.

:param nodes: Nodes of the new Bayesian network.
:param nodes: Interface nodes of the new Bayesian network.
:returns: A new empty conditional Bayesian network.
)doc")
        // The equality operator do not compile in GCC, so it is implemented with lambdas:
        // https://github.com/pybind/pybind11/issues/1487
        .def(
            "__eq__",
            [](const BayesianNetworkType& self, const BayesianNetworkType& other) { return self == other; },
            py::arg("other"),
            py::is_operator())
        .def(
            "__ne__",
            [](const BayesianNetworkType& self, const BayesianNetworkType& other) { return self != other; },
            py::arg("other"),
            py::is_operator())
        // .def(py::self == py::self)
        // .def(py::self != py::self)
        .def("__str__", &BayesianNetworkType::ToString)
        .def("__repr__", &BayesianNetworkType::ToString)
        .def("__getstate__", [](const BayesianNetworkType& self) { return self.__getstate__(); })
        .def("__setstate__", [](py::object& self, py::tuple& t) { PyBayesianNetworkType::__setstate__(self, t); })
        .def("__hash__", &BayesianNetworkType::hash);

    {
        py::options options;
        options.disable_function_signatures();

        bn_type
            .def("compatible_node_type",
                 py::overload_cast<const ConditionalBayesianNetworkBase&,
                                   const std::string&,
                                   const std::shared_ptr<FactorType>&>(&BayesianNetworkType::compatible_node_type,
                                                                       py::const_),
                 py::arg("model"),
                 py::arg("node"),
                 py::arg("node_type"))
            .def("compatible_node_type",
                 py::overload_cast<const BayesianNetworkBase&, const std::string&, const std::shared_ptr<FactorType>&>(
                     &BayesianNetworkType::compatible_node_type, py::const_),
                 py::arg("model"),
                 py::arg("node"),
                 py::arg("node_type"),
                 R"doc(
compatible_node_type(model: BayesianNetworkBase or ConditionalBayesianNetworkBase, node: str, node_type: pybnesian.FactorType) -> bool

Checks whether the :class:`FactorType <pybnesian.FactorType>` ``node_type`` is allowed for ``node`` by
this :class:`BayesianNetworkType`.

:param model: BayesianNetwork model.
:param node: Name of the node to check.
:param node_type: :class:`FactorType <pybnesian.FactorType>` for ``node``.
:returns: True if the current :class:`FactorType <pybnesian.FactorType>` is allowed, False otherwise.
)doc")
            .def("can_have_arc",
                 py::overload_cast<const ConditionalBayesianNetworkBase&, const std::string&, const std::string&>(
                     &BayesianNetworkType::can_have_arc, py::const_),
                 py::arg("model"),
                 py::arg("source"),
                 py::arg("target"))
            .def("can_have_arc",
                 py::overload_cast<const BayesianNetworkBase&, const std::string&, const std::string&>(
                     &BayesianNetworkType::can_have_arc, py::const_),
                 py::arg("model"),
                 py::arg("source"),
                 py::arg("target"),
                 R"doc(
can_have_arc(model: BayesianNetworkBase or ConditionalBayesianNetworkBase, source: str, target: str) -> bool

Checks whether the :class:`BayesianNetworkType` allows an arc ``source`` -> ``target`` in the Bayesian network ``model``.

:param model: BayesianNetwork model.
:param source: Name of the source node.
:param target: Name of the target node.
:returns: True if the arc ``source`` -> ``target`` is allowed in ``model``, False otherwise.
)doc")
            .def("alternative_node_type",
                 py::overload_cast<const ConditionalBayesianNetworkBase&, const std::string&>(
                     &BayesianNetworkType::alternative_node_type, py::const_),
                 py::arg("model"),
                 py::arg("node"))
            .def("alternative_node_type",
                 py::overload_cast<const BayesianNetworkBase&, const std::string&>(
                     &BayesianNetworkType::alternative_node_type, py::const_),
                 py::arg("model"),
                 py::arg("node"),
                 R"doc(
alternative_node_type(model: BayesianNetworkBase or ConditionalBayesianNetworkBase, source: str) -> List[pybnesian.FactorType]

Returns all feasible alternative :class:`FactorType <pybnesian.FactorType>` for ``node``.

:param model: BayesianNetwork model.
:param node: Name of the node.
:returns: A list of alternative :class:`FactorType <pybnesian.FactorType>`. If you implement this method in a
    Python-derived class, you can return an empty list or None to specify that no changes are possible.
)doc");
    }

    py::class_<GaussianNetworkType, BayesianNetworkType, std::shared_ptr<GaussianNetworkType>>(
        root, "GaussianNetworkType", R"doc(
This :class:`BayesianNetworkType` represents a Gaussian network: homogeneous with
:class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` factors.
)doc")
        .def(py::init(&GaussianNetworkType::get))
        .def(py::pickle([](const GaussianNetworkType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return GaussianNetworkType::get(); }));

    py::class_<SemiparametricBNType, BayesianNetworkType, std::shared_ptr<SemiparametricBNType>>(
        root, "SemiparametricBNType", R"doc(
This :class:`BayesianNetworkType` represents a semiparametric Bayesian network: non-homogeneous with
:class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` and
:class:`CKDE <pybnesian.CKDE>` factors for continuous data. The default is
:class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>`. It also supports discrete data using
:class:`DiscreteFactor <pybnesian.DiscreteFactor>`.

In a SemiparametricBN network, the discrete nodes can only have discrete parents.
)doc")
        .def(py::init(&SemiparametricBNType::get))
        .def(py::pickle([](const SemiparametricBNType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return SemiparametricBNType::get(); }));

    py::class_<KDENetworkType, BayesianNetworkType, std::shared_ptr<KDENetworkType>>(root, "KDENetworkType", R"doc(
This :class:`BayesianNetworkType` represents a KDE Bayesian network: homogeneous with
:class:`CKDE <pybnesian.CKDE>` factors.
)doc")
        .def(py::init(&KDENetworkType::get))
        .def(py::pickle([](const KDENetworkType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return KDENetworkType::get(); }));

    py::class_<DiscreteBNType, BayesianNetworkType, std::shared_ptr<DiscreteBNType>>(root, "DiscreteBNType", R"doc(
This :class:`BayesianNetworkType` represents a discrete Bayesian network: homogeneous with
:class:`DiscreteFactor <pybnesian.DiscreteFactor>` factors.
)doc")
        .def(py::init(&DiscreteBNType::get))
        .def(py::pickle([](const DiscreteBNType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return DiscreteBNType::get(); }));

    py::class_<HomogeneousBNType, BayesianNetworkType, std::shared_ptr<HomogeneousBNType>>(root, "HomogeneousBNType")
        .def(py::init<>([](std::shared_ptr<FactorType> default_node_type) {
                 return HomogeneousBNType(FactorType::keep_python_alive(default_node_type));
             }),
             py::arg("default_factor_type"),
             R"doc(
Initializes an :class:`HomogeneousBNType` with a default node type.

:param default_factor_type: Default factor type for all the nodes in the Bayesian network.
)doc")
        .def(py::pickle([](const HomogeneousBNType& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return HomogeneousBNType::__setstate__(t); }));

    py::class_<HeterogeneousBNType, BayesianNetworkType, std::shared_ptr<HeterogeneousBNType>>(root,
                                                                                               "HeterogeneousBNType")
        .def(py::init<>([](std::vector<std::shared_ptr<FactorType>> default_factor_type) {
                 return HeterogeneousBNType(FactorType::keep_vector_python_alive(default_factor_type));
             }),
             py::arg("default_factor_type"),
             R"doc(
Initializes an :class:`HeterogeneousBNType` with a list of default node types for all the data types.

:param default_factor_type: Default factor type for all the nodes in the Bayesian network.
)doc")
        .def(py::init<>([](const models::MapDataToFactor& default_factor_types) {
                 return HeterogeneousBNType(models::keep_MapDataToFactor_alive(default_factor_types));
             }),
             py::arg("default_factor_types"),
             R"doc(
Initializes an :class:`HeterogeneousBNType` with a default node type for a set of data types.

:param default_factor_type: Default factor type depending on the factor type.
)doc")
        .def("single_default", &HeterogeneousBNType::single_default, R"doc(
Checks whether the :class:`HeterogeneousBNType` defines only a default
:class:`FactorType <pybnesian.FactorType>` for all the data types.

:returns: True if it defines a single :class:`FactorType <pybnesian.FactorType>` for all the data types.
    False if different default :class:`FactorType <pybnesian.FactorType>` is defined for different data types.
)doc")
        .def("default_node_types", &HeterogeneousBNType::default_node_types, R"doc(
Returns the dict of default :class:`FactorType <pybnesian.FactorType>` for each data type.

:returns: dict of default :class:`FactorType <pybnesian.FactorType>` for each data type.
)doc")
        .def(py::pickle([](const HeterogeneousBNType& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return HeterogeneousBNType::__setstate__(t); }));

    py::class_<CLGNetworkType, BayesianNetworkType, std::shared_ptr<CLGNetworkType>>(root, "CLGNetworkType", R"doc(
This :class:`BayesianNetworkType` represents a conditional linear Gaussian (CLG) network: heterogeneous with
:class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` factors for the continuous data and
:class:`DiscreteFactor <pybnesian.DiscreteFactor>` for the categorical data.

In a CLG network, the discrete nodes can only have discrete parents, while the continuous nodes can have discrete and
continuous parents.
)doc")
        .def(py::init(&CLGNetworkType::get))
        .def(py::pickle([](const CLGNetworkType& self) { return self.__getstate__(); },
                        [](py::tuple&) { return CLGNetworkType::get(); }));

    register_BayesianNetwork_methods<BayesianNetworkBase>(bn_base);
    register_ConditionalBayesianNetwork_methods<ConditionalBayesianNetworkBase>(cbn_base);

    py::class_<DynamicBayesianNetworkBase, PyDynamicBayesianNetworkBase<>, std::shared_ptr<DynamicBayesianNetworkBase>>
        dbn_base(root, "DynamicBayesianNetworkBase", R"doc(
This class defines an interface of a dynamic Bayesian network.

A dynamic Bayesian network is defined over a set of variables. Each variable is replicated in different nodes (one for
each temporal slice). Thus, we differentiate in this documentation between the terms "variable" and "node". To create
the nodes, we suffix the variable names using the structure ``[variable_name]_t_[temporal_index]``. The
``variable_name`` is the name of each variable, and ``temporal_index`` is an index with a range [0-``markovian_order``].
The index “0” is considered the “present”, the index “1” delays the temporal one step into the “past”, and so on… This
is related with the way :class:`DynamicDataFrame <pybnesian.DynamicDataFrame>` generates the columns.

The dynamic Bayesian is composed of two Bayesian networks:

- a static Bayesian network that defines the probability distribution of the first ``markovian_order`` instances. It
  estimates the probability f(``t_1``,..., ``t_[markovian_order]``). This Bayesian network is represented with a normal
  Bayesian network.

- a transition Bayesian network that defines the probability distribution of the i-th instance given the previous
  ``markovian_order`` instances. It estimates the probability f(``t_0`` | ``t_1``, ..., ``t_[markovian_order]``), where
  ``t_0`` (the present) is the i-th instance. Once the probability of the i-th instance is estimated, the transition
  network moves a step forward, to estimate the (i+1)-th instance, and so on. This transition Bayesian network is
  represented with a conditional Bayesian network.

Both Bayesian networks must be of the same :class:`BayesianNetworkType`.
)doc");
    register_DynamicBayesianNetwork_methods<DynamicBayesianNetworkBase>(dbn_base);

    /** Replicate here the constructors because the Python derived object is destroyed if
     * goes out of scope. It will be probably fixed in:
     * https://github.com/pybind/pybind11/pull/2839
     * See also:
     * https://github.com/pybind/pybind11/issues/1333.
     */
    py::class_<BayesianNetwork, BayesianNetworkBase, PyBayesianNetwork<>, std::shared_ptr<BayesianNetwork>> bn(
        root, "BayesianNetwork");
    bn.def(py::init(
               [](std::shared_ptr<BayesianNetworkType> t, const std::vector<std::string>& nodes) {
                   return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), nodes);
               },
               [](std::shared_ptr<BayesianNetworkType> t, const std::vector<std::string>& nodes) {
                   return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t), nodes);
               }),
           py::arg("type"),
           py::arg("nodes"),
           R"doc(
Initializes the :class:`BayesianNetwork` with a given ``type`` and ``nodes``.

:param type: :class:`BayesianNetworkType` of this Bayesian network.
:param nodes: List of node names.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const std::vector<std::string>& nodes,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t),
                                                              nodes,
                                                              util::keep_FactorTypeVector_python_alive(node_types));
                 },
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const std::vector<std::string>& nodes,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t),
                                                                  nodes,
                                                                  util::keep_FactorTypeVector_python_alive(node_types));
                 }),
             py::arg("type"),
             py::arg("nodes"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`BayesianNetwork` with a given ``type`` and ``nodes``. It specifies the ``node_types`` for the
nodes.

:param type: :class:`BayesianNetworkType` of this Bayesian network.
:param nodes: List of node names.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t, const ArcStringVector& arcs) {
                     return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), arcs);
                 },
                 [](std::shared_ptr<BayesianNetworkType> t, const ArcStringVector& arcs) {
                     return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t), arcs);
                 }),
             py::arg("type"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`BayesianNetwork` with a given ``type`` and ``arcs`` (the nodes are extracted from the arcs).

:param type: :class:`BayesianNetworkType` of this Bayesian network.
:param arcs: Arcs of the Bayesian network.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const ArcStringVector& arcs,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t),
                                                              arcs,
                                                              util::keep_FactorTypeVector_python_alive(node_types));
                 },
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const ArcStringVector& arcs,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t),
                                                                  arcs,
                                                                  util::keep_FactorTypeVector_python_alive(node_types));
                 }),
             py::arg("type"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`BayesianNetwork` with a given ``type`` and ``arcs`` (the nodes are extracted from the arcs).
It specifies the ``node_types`` for the nodes.

:param type: :class:`BayesianNetworkType` of this Bayesian network.
:param arcs: Arcs of the Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const std::vector<std::string>& nodes,
                    const ArcStringVector& arcs) {
                     return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), nodes, arcs);
                 },
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const std::vector<std::string>& nodes,
                    const ArcStringVector& arcs) {
                     return std::make_shared<PyBayesianNetwork<>>(
                         BayesianNetworkType::keep_python_alive(t), nodes, arcs);
                 }),
             py::arg("type"),
             py::arg("nodes"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`BayesianNetwork` with a given ``type``, ``nodes`` and ``arcs``.

:param type: :class:`BayesianNetworkType` of this Bayesian network.
:param nodes: List of node names.
:param arcs: Arcs of the Bayesian network.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const std::vector<std::string>& nodes,
                    const ArcStringVector& arcs,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t),
                                                              nodes,
                                                              arcs,
                                                              util::keep_FactorTypeVector_python_alive(node_types));
                 },
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const std::vector<std::string>& nodes,
                    const ArcStringVector& arcs,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t),
                                                                  nodes,
                                                                  arcs,
                                                                  util::keep_FactorTypeVector_python_alive(node_types));
                 }),
             py::arg("type"),
             py::arg("nodes"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`BayesianNetwork` with a given ``type``, ``nodes`` and ``arcs``. It specifies the ``node_types``
for the nodes.

:param type: :class:`BayesianNetworkType` of this Bayesian network.
:param nodes: List of node names.
:param arcs: Arcs of the Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t, const Dag& graph) {
                     return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t), graph);
                 },
                 [](std::shared_ptr<BayesianNetworkType> t, const Dag& graph) {
                     return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t), graph);
                 }),
             py::arg("type"),
             py::arg("graph"),
             R"doc(
Initializes the :class:`BayesianNetwork` with a given ``type``, and ``graph``

:param type: :class:`BayesianNetworkType` of this Bayesian network.
:param graph: :class:`Dag <pybnesian.Dag>` of the Bayesian network.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t, const Dag& graph, const FactorTypeVector& node_types) {
                     return std::make_shared<BayesianNetwork>(BayesianNetworkType::keep_python_alive(t),
                                                              graph,
                                                              util::keep_FactorTypeVector_python_alive(node_types));
                 },
                 [](std::shared_ptr<BayesianNetworkType> t, const Dag& graph, const FactorTypeVector& node_types) {
                     return std::make_shared<PyBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t),
                                                                  graph,
                                                                  util::keep_FactorTypeVector_python_alive(node_types));
                 }),
             py::arg("type"),
             py::arg("graph"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`BayesianNetwork` with a given ``type``, and ``graph``. It specifies the ``node_types``
for the nodes.

:param type: :class:`BayesianNetworkType` of this Bayesian network.
:param graph: :class:`Dag <pybnesian.Dag>` of the Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def("__getstate__", [](const BayesianNetwork& self) { return self.__getstate__(); })
        .def("__setstate__", [](py::object& self, py::tuple& t) {
            // If is a C++ BayesianNetwork
            if (self.get_type().ptr() == py::type::of<BayesianNetwork>().ptr()) {
                models::__nonderived_bn_setstate__<BayesianNetwork>(self, t);
            } else {
                PyBayesianNetwork<BayesianNetwork>::__setstate__(self, t);
            }
        });

    register_BNGeneric_methods<BayesianNetwork>(bn);

    py::class_<ConditionalBayesianNetwork,
               ConditionalBayesianNetworkBase,
               PyConditionalBayesianNetwork<>,
               std::shared_ptr<ConditionalBayesianNetwork>>
        cbn(root, "ConditionalBayesianNetwork");
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
                }),
            py::arg("type"),
            py::arg("nodes"),
            py::arg("interface_nodes"),
            R"doc(
Initializes the :class:`ConditionalBayesianNetwork` with a given ``type``, ``nodes`` and ``interface_nodes``.

:param type: :class:`BayesianNetworkType` of this conditional Bayesian network.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const std::vector<std::string>& nodes,
                    const std::vector<std::string>& interface_nodes,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<ConditionalBayesianNetwork>(
                         BayesianNetworkType::keep_python_alive(t),
                         nodes,
                         interface_nodes,
                         util::keep_FactorTypeVector_python_alive(node_types));
                 },
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const std::vector<std::string>& nodes,
                    const std::vector<std::string>& interface_nodes,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<PyConditionalBayesianNetwork<>>(
                         BayesianNetworkType::keep_python_alive(t),
                         nodes,
                         interface_nodes,
                         util::keep_FactorTypeVector_python_alive(node_types));
                 }),
             py::arg("type"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalBayesianNetwork` with a given ``type``, ``nodes`` and ``interface_nodes``. It
specifies the ``node_types`` for the nodes.

:param type: :class:`BayesianNetworkType` of this conditional Bayesian network.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
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
                 }),
             py::arg("type"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`ConditionalBayesianNetwork` with a given ``type``, ``nodes``, ``interface_nodes`` and ``arcs``.

:param type: :class:`BayesianNetworkType` of this conditional Bayesian network.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param arcs: Arcs of the conditional Bayesian network.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const std::vector<std::string>& nodes,
                    const std::vector<std::string>& interface_nodes,
                    const ArcStringVector& arcs,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<ConditionalBayesianNetwork>(
                         BayesianNetworkType::keep_python_alive(t),
                         nodes,
                         interface_nodes,
                         arcs,
                         util::keep_FactorTypeVector_python_alive(node_types));
                 },
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const std::vector<std::string>& nodes,
                    const std::vector<std::string>& interface_nodes,
                    const ArcStringVector& arcs,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<PyConditionalBayesianNetwork<>>(
                         BayesianNetworkType::keep_python_alive(t),
                         nodes,
                         interface_nodes,
                         arcs,
                         util::keep_FactorTypeVector_python_alive(node_types));
                 }),
             py::arg("type"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalBayesianNetwork` with a given ``type``, ``nodes``, ``interface_nodes`` and ``arcs``.
It specifies the ``node_types`` for the nodes.

:param type: :class:`BayesianNetworkType` of this conditional Bayesian network.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param arcs: Arcs of the conditional Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t, const ConditionalDag& graph) {
                     return std::make_shared<ConditionalBayesianNetwork>(BayesianNetworkType::keep_python_alive(t),
                                                                         graph);
                 },
                 [](std::shared_ptr<BayesianNetworkType> t, const ConditionalDag& graph) {
                     return std::make_shared<PyConditionalBayesianNetwork<>>(BayesianNetworkType::keep_python_alive(t),
                                                                             graph);
                 }),
             py::arg("type"),
             py::arg("graph"),
             R"doc(
Initializes the :class:`ConditionalBayesianNetwork` with a given ``type``, and ``graph``

:param type: :class:`BayesianNetworkType` of this conditional Bayesian network.
:param graph: :class:`ConditionalDag <pybnesian.ConditionalDag>` of the conditional Bayesian network.
)doc")
        .def(py::init(
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const ConditionalDag& graph,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<ConditionalBayesianNetwork>(
                         BayesianNetworkType::keep_python_alive(t),
                         graph,
                         util::keep_FactorTypeVector_python_alive(node_types));
                 },
                 [](std::shared_ptr<BayesianNetworkType> t,
                    const ConditionalDag& graph,
                    const FactorTypeVector& node_types) {
                     return std::make_shared<PyConditionalBayesianNetwork<>>(
                         BayesianNetworkType::keep_python_alive(t),
                         graph,
                         util::keep_FactorTypeVector_python_alive(node_types));
                 }),
             py::arg("type"),
             py::arg("graph"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalBayesianNetwork` with a given ``type``, and ``graph``. It specifies the
``node_types`` for the nodes.

:param type: :class:`BayesianNetworkType` of this conditional Bayesian network.
:param graph: :class:`ConditionalDag <pybnesian.ConditionalDag>` of the conditional Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def("__getstate__", [](const ConditionalBayesianNetwork& self) { return self.__getstate__(); })
        .def("__setstate__", [](py::object& self, py::tuple& t) {
            // If is a C++ type
            if (self.get_type().ptr() == py::type::of<ConditionalBayesianNetwork>().ptr()) {
                models::__nonderived_bn_setstate__<ConditionalBayesianNetwork>(self, t);
            } else {
                PyBayesianNetwork<ConditionalBayesianNetwork>::__setstate__(self, t);
            }
        });
    register_BNGeneric_methods<ConditionalBayesianNetwork>(cbn);

    py::class_<DynamicBayesianNetwork,
               DynamicBayesianNetworkBase,
               PyDynamicBayesianNetwork<>,
               std::shared_ptr<DynamicBayesianNetwork>>
        dbn(root, "DynamicBayesianNetwork");
    dbn.def(py::init(
                [](std::shared_ptr<BayesianNetworkType> t,
                   const std::vector<std::string>& variables,
                   int markovian_order) {
                    return std::make_shared<DynamicBayesianNetwork>(
                        BayesianNetworkType::keep_python_alive(t), variables, markovian_order);
                },
                [](std::shared_ptr<BayesianNetworkType> t,
                   const std::vector<std::string>& variables,
                   int markovian_order) {
                    return std::make_shared<PyDynamicBayesianNetwork<>>(
                        BayesianNetworkType::keep_python_alive(t), variables, markovian_order);
                }),
            py::arg("type"),
            py::arg("variables"),
            py::arg("markovian_order"),
            R"doc(
Initializes the :class:`DynamicBayesianNetwork` with the given ``variables`` and ``markovian_order``. It creates
empty the static and transition Bayesian networks with the given ``type``.

:param type: :class:`BayesianNetworkType` of the static and transition Bayesian networks.
:param variables: List of node names.
:param markovian_order: Markovian order of the dynamic Bayesian network.
)doc")
        .def(py::init<>(
                 [](const std::vector<std::string>& variables,
                    int markovian_order,
                    std::shared_ptr<BayesianNetworkBase> static_bn,
                    std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn) {
                     return std::make_shared<DynamicBayesianNetwork>(
                         variables,
                         markovian_order,
                         BayesianNetworkBase::keep_python_alive(static_bn),
                         ConditionalBayesianNetworkBase::keep_python_alive(transition_bn));
                 },
                 [](const std::vector<std::string>& variables,
                    int markovian_order,
                    std::shared_ptr<BayesianNetworkBase> static_bn,
                    std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn) {
                     return std::make_shared<PyDynamicBayesianNetwork<>>(
                         variables,
                         markovian_order,
                         BayesianNetworkBase::keep_python_alive(static_bn),
                         ConditionalBayesianNetworkBase::keep_python_alive(transition_bn));
                 }),
             py::arg("variables"),
             py::arg("markovian_order"),
             py::arg("static_bn"),
             py::arg("transition_bn"),
             R"doc(
Initializes the :class:`DynamicBayesianNetwork` with the given ``variables`` and ``markovian_order``. The static and
transition Bayesian networks are initialized with ``static_bn`` and ``transition_bn`` respectively.

Both ``static_bn`` and ``transition`` must contain the expected nodes:

- For the static network, it must contain the nodes from ``[variable_name]_t_1`` to
  ``[variable_name]_t_[markovian_order]``.
- For the transition network, it must contain the nodes ``[variable_name]_t_0``, and the interface nodes from
  ``[variable_name]_t_1`` to ``[variable_name]_t_[markovian_order]``.

The static and transition networks must have the same :class:`BayesianNetworkType`.

:param variables: List of node names.
:param markovian_order: Markovian order of the dynamic Bayesian network.
:param static_bn: Static Bayesian network.
:param transition_bn: Transition Bayesian network.
)doc")
        .def_property("include_cpd", &DynamicBayesianNetwork::include_cpd, &DynamicBayesianNetwork::set_include_cpd)
        .def("__getstate__", [](const DynamicBayesianNetwork& self) { return self.__getstate__(); })
        .def("__setstate__", [](py::object& self, py::tuple& t) {
            if (self.get_type().ptr() == py::type::of<DynamicBayesianNetwork>().ptr()) {
                models::__nonderived_dbn_setstate__(self, t);
            } else {
                PyDynamicBayesianNetwork<DynamicBayesianNetwork>::__setstate__(self, t);
            }
        });

    register_DerivedBayesianNetwork<GaussianNetwork>(root, "GaussianNetwork", R"doc(
This class implements a :class:`BayesianNetwork` with the type :class:`GaussianNetworkType`.
)doc");

    auto spbn = register_DerivedBayesianNetwork<SemiparametricBN>(root, "SemiparametricBN", R"doc(
This class implements a :class:`BayesianNetwork` with the type :class:`SemiparametricBNType`.
)doc");
    spbn.def(py::init<const std::vector<std::string>&, FactorTypeVector&>(),
             py::arg("nodes"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`SemiparametricBN` with the given ``nodes``. It specifies the ``node_types``
for the nodes.

:param nodes: List of node names.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init<const ArcStringVector&, FactorTypeVector&>(), py::arg("arcs"), py::arg("node_types"), R"doc(
Initializes the :class:`SemiparametricBN` with the given ``arcs`` (the nodes are extracted from the arcs). It specifies
the ``node_types`` for the nodes.

:param arcs: Arcs of the SemiparametricBN.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init<const std::vector<std::string>&, const ArcStringVector&, FactorTypeVector&>(),
             py::arg("nodes"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`SemiparametricBN` with the given ``nodes`` and ``arcs``. It specifies the ``node_types`` for the
nodes.

:param nodes: List of node names.
:param arcs: Arcs of the :class:`SemiparametricBN`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init<const Dag&, FactorTypeVector&>(), py::arg("graph"), py::arg("node_types"), R"doc(
Initializes the :class:`SemiparametricBN` with the given ``graph``. It specifies the ``node_types`` for the nodes.

:param graph: :class:`Dag <pybnesian.Dag>` of the Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc");

    register_DerivedBayesianNetwork<KDENetwork>(root, "KDENetwork", R"doc(
This class implements a :class:`BayesianNetwork` with the type :class:`KDENetworkType`.
)doc");
    register_DerivedBayesianNetwork<DiscreteBN>(root, "DiscreteBN", R"doc(
This class implements a :class:`BayesianNetwork` with the type :class:`DiscreteBNType`.
)doc");

    py::class_<HomogeneousBN, BayesianNetwork, std::shared_ptr<HomogeneousBN>>(root, "HomogeneousBN", R"doc(
This class implements an homogeneous Bayesian network. This Bayesian network can be used with any
:class:`FactorType <pybnesian.FactorType>`. You can set the :class:`FactorType` in the constructor.
)doc")
        .def(py::init([](std::shared_ptr<FactorType> ft, const std::vector<std::string>& nodes) {
                 return HomogeneousBN(FactorType::keep_python_alive(ft), nodes);
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             R"doc(
Initializes the :class:`HomogeneousBN` of ``factor_type`` with the given ``nodes``.

:param factor_type: :class:`FactorType <pybnesian.FactorType>` for all the nodes.
:param nodes: List of node names.
)doc")
        .def(py::init([](std::shared_ptr<FactorType> ft, const ArcStringVector& arcs) {
                 return HomogeneousBN(FactorType::keep_python_alive(ft), arcs);
             }),
             py::arg("factor_type"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`HomogeneousBN` of ``factor_type`` with the given ``arcs`` (the nodes are extracted from the
arcs).

:param factor_type: :class:`FactorType <pybnesian.FactorType>` for all the nodes.
:param arcs: Arcs of the :class:`HomogeneousBN`.
)doc")
        .def(
            py::init(
                [](std::shared_ptr<FactorType> ft, const std::vector<std::string>& nodes, const ArcStringVector& arcs) {
                    return HomogeneousBN(FactorType::keep_python_alive(ft), nodes, arcs);
                }),
            py::arg("factor_type"),
            py::arg("nodes"),
            py::arg("arcs"),
            R"doc(
Initializes the :class:`HomogeneousBN` of ``factor_type`` with the given ``nodes`` and ``arcs``.

:param factor_type: :class:`FactorType <pybnesian.FactorType>` for all the nodes.
:param nodes: List of node names.
:param arcs: Arcs of the :class:`HomogeneousBN`.
)doc")
        .def(py::init([](std::shared_ptr<FactorType> ft, const Dag& graph) {
                 return HomogeneousBN(FactorType::keep_python_alive(ft), graph);
             }),
             py::arg("factor_type"),
             py::arg("graph"),
             R"doc(
Initializes the :class:`HomogeneousBN` of ``factor_type`` with the given ``graph``.

:param factor_type: :class:`FactorType <pybnesian.FactorType>` for all the nodes.
:param graph: :class:`Dag <pybnesian.Dag>` of the Bayesian network.
)doc")
        .def(py::pickle([](const HomogeneousBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__homogeneous_setstate__<HomogeneousBN>(t); }));

    py::class_<HeterogeneousBN, BayesianNetwork, std::shared_ptr<HeterogeneousBN>>(root, "HeterogeneousBN", R"doc(
This class implements an heterogeneous Bayesian network. This Bayesian network accepts a different
:class:`FactorType <pybnesian.FactorType>` for each node. You can set the default :class:`FactorType` in the
constructor.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft, const std::vector<std::string>& nodes) {
                 return HeterogeneousBN(FactorType::keep_vector_python_alive(ft), nodes);
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of default ``factor_type`` with the given ``nodes``.

:param factor_type: List of default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network.
:param nodes: List of node names.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft,
                         const std::vector<std::string>& nodes,
                         const FactorTypeVector& node_types) {
                 return HeterogeneousBN(FactorType::keep_vector_python_alive(ft),
                                        nodes,
                                        util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of default ``factor_type`` with the given ``nodes`` and ``node_types``.

:param factor_type: List of default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network.
:param nodes: List of node names.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft, const ArcStringVector& arcs) {
                 return HeterogeneousBN(FactorType::keep_vector_python_alive(ft), arcs);
             }),
             py::arg("factor_type"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of default ``factor_type`` with the given ``arcs`` (the nodes are extracted
from the arcs).

:param factor_type: List of default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network.
:param arcs: Arcs of the :class:`HeterogeneousBN`.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft,
                         const ArcStringVector& arcs,
                         const FactorTypeVector& node_types) {
                 return HeterogeneousBN(FactorType::keep_vector_python_alive(ft),
                                        arcs,
                                        util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_type"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of default ``factor_type`` with the given ``arcs`` (the nodes are extracted from the arcs) and ``node_types``.

:param factor_type: List of default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network.
:param arcs: Arcs of the :class:`HeterogeneousBN`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft,
                         const std::vector<std::string>& nodes,
                         const ArcStringVector& arcs) {
                 return HeterogeneousBN(FactorType::keep_vector_python_alive(ft), nodes, arcs);
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of default ``factor_type`` with the given ``nodes`` and ``arcs``.

:param factor_type: List of default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network.
:param nodes: List of node names.
:param arcs: Arcs of the :class:`HeterogeneousBN`.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft,
                         const std::vector<std::string>& nodes,
                         const ArcStringVector& arcs,
                         const FactorTypeVector& node_types) {
                 return HeterogeneousBN(FactorType::keep_vector_python_alive(ft),
                                        nodes,
                                        arcs,
                                        util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of default ``factor_type`` with the given ``nodes``, ``arcs`` and ``node_types``.

:param factor_type: List of default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network.
:param nodes: List of node names.
:param arcs: Arcs of the :class:`HeterogeneousBN`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft, const Dag& graph) {
                 return HeterogeneousBN(FactorType::keep_vector_python_alive(ft), graph);
             }),
             py::arg("factor_type"),
             py::arg("graph"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of default ``factor_type`` with the given ``graph``.

:param factor_type: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network.
:param graph: :class:`Dag <pybnesian.Dag>` of the Bayesian network.
)doc")
        .def(py::init(
                 [](std::vector<std::shared_ptr<FactorType>> ft, const Dag& graph, const FactorTypeVector& node_types) {
                     return HeterogeneousBN(FactorType::keep_vector_python_alive(ft),
                                            graph,
                                            util::keep_FactorTypeVector_python_alive(node_types));
                 }),
             py::arg("factor_type"),
             py::arg("graph"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of default ``factor_type`` with the given ``graph`` and ``node_types``.

:param factor_type: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network.
:param graph: :class:`Dag <pybnesian.Dag>` of the Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](MapDataToFactor fts, const std::vector<std::string>& nodes) {
                 return HeterogeneousBN(models::keep_MapDataToFactor_alive(fts), nodes);
             }),
             py::arg("factor_types"),
             py::arg("nodes"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of different default ``factor_types``, with the given ``nodes``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param nodes: List of node names.
)doc")
        .def(py::init(
                 [](MapDataToFactor fts, const std::vector<std::string>& nodes, const FactorTypeVector& node_types) {
                     return HeterogeneousBN(models::keep_MapDataToFactor_alive(fts),
                                            nodes,
                                            util::keep_FactorTypeVector_python_alive(node_types));
                 }),
             py::arg("factor_types"),
             py::arg("nodes"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of different default ``factor_types``, with the given ``nodes`` and ``node_types``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param nodes: List of node names.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](MapDataToFactor fts, const ArcStringVector& arcs) {
                 return HeterogeneousBN(models::keep_MapDataToFactor_alive(fts), arcs);
             }),
             py::arg("factor_types"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of different default ``factor_types`` with the given ``arcs`` (the nodes are
extracted from the arcs).

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param arcs: Arcs of the :class:`HeterogeneousBN`.
)doc")
        .def(py::init([](MapDataToFactor fts, const ArcStringVector& arcs, const FactorTypeVector& node_types) {
                 return HeterogeneousBN(models::keep_MapDataToFactor_alive(fts),
                                        arcs,
                                        util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_types"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of different default ``factor_types`` with the given ``arcs`` (the nodes are
extracted from the arcs) and ``node_types``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param arcs: Arcs of the :class:`HeterogeneousBN`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](MapDataToFactor fts, const std::vector<std::string>& nodes, const ArcStringVector& arcs) {
                 return HeterogeneousBN(models::keep_MapDataToFactor_alive(fts), nodes, arcs);
             }),
             py::arg("factor_types"),
             py::arg("nodes"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of different default ``factor_types`` with the given ``nodes`` and ``arcs``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param nodes: List of node names.
:param arcs: Arcs of the :class:`HeterogeneousBN`.
)doc")
        .def(py::init([](MapDataToFactor fts,
                         const std::vector<std::string>& nodes,
                         const ArcStringVector& arcs,
                         const FactorTypeVector& node_types) {
                 return HeterogeneousBN(models::keep_MapDataToFactor_alive(fts),
                                        nodes,
                                        arcs,
                                        util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_types"),
             py::arg("nodes"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of different default ``factor_types`` with the given ``nodes``, ``arcs`` and ``node_types``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param nodes: List of node names.
:param arcs: Arcs of the :class:`HeterogeneousBN`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](MapDataToFactor fts, const Dag& graph) {
                 return HeterogeneousBN(models::keep_MapDataToFactor_alive(fts), graph);
             }),
             py::arg("factor_types"),
             py::arg("graph"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of different default ``factor_types`` with the given ``graph``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param graph: :class:`Dag <pybnesian.Dag>` of the Bayesian network.
)doc")
        .def(py::init([](MapDataToFactor fts, const Dag& graph, const FactorTypeVector& node_types) {
                 return HeterogeneousBN(models::keep_MapDataToFactor_alive(fts),
                                        graph,
                                        util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_types"),
             py::arg("graph"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`HeterogeneousBN` of different default ``factor_types`` with the given ``graph`` and ``node_types``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param graph: :class:`Dag <pybnesian.Dag>` of the Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::pickle([](const HeterogeneousBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__heterogeneous_setstate__<HeterogeneousBN>(t); }));

    auto clg = register_DerivedBayesianNetwork<CLGNetwork>(root, "CLGNetwork", R"doc(
This class implements a :class:`BayesianNetwork` with the type :class:`CLGNetworkType`.
)doc");

    clg.def(py::init<const std::vector<std::string>&, FactorTypeVector&>(),
            py::arg("nodes"),
            py::arg("node_types"),
            R"doc(
Initializes the :class:`CLGNetwork` with the given ``nodes``. It specifies the ``node_types``
for the nodes.

:param nodes: List of node names.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init<const ArcStringVector&, FactorTypeVector&>(), py::arg("arcs"), py::arg("node_types"), R"doc(
Initializes the :class:`CLGNetwork` with the given ``arcs`` (the nodes are extracted from the arcs). It specifies
the ``node_types`` for the nodes.

:param arcs: Arcs of the :class:`CLGNetwork`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init<const std::vector<std::string>&, const ArcStringVector&, FactorTypeVector&>(),
             py::arg("nodes"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`CLGNetwork` with the given ``nodes`` and ``arcs``. It specifies the ``node_types`` for the
nodes.

:param nodes: List of node names.
:param arcs: Arcs of the :class:`CLGNetwork`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init<const Dag&, FactorTypeVector&>(), py::arg("graph"), py::arg("node_types"), R"doc(
Initializes the :class:`CLGNetwork` with the given ``graph``. It specifies the ``node_types`` for the nodes.

:param graph: :class:`Dag <pybnesian.Dag>` of the Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc");

    register_DerivedConditionalBayesianNetwork<ConditionalGaussianNetwork>(root, "ConditionalGaussianNetwork", R"doc(
This class implements a :class:`ConditionalBayesianNetwork` with the type :class:`GaussianNetworkType`.
)doc");
    auto conditional_spbn = register_DerivedConditionalBayesianNetwork<ConditionalSemiparametricBN>(
        root, "ConditionalSemiparametricBN", R"doc(
This class implements a :class:`ConditionalBayesianNetwork` with the type :class:`SemiparametricBNType`.
)doc");
    conditional_spbn
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, FactorTypeVector&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalSemiparametricBN` with the given ``nodes`` and ``interface_nodes``. It specifies the
``node_types`` for the nodes.

:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init<const std::vector<std::string>&,
                      const std::vector<std::string>&,
                      const ArcStringVector&,
                      FactorTypeVector&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalSemiparametricBN` with the given ``nodes``, ``interface_nodes`` and ``arcs``. It
specifies the ``node_types`` for the nodes.

:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param arcs: Arcs of the :class:`ConditionalSemiparametricBN`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init<const ConditionalDag&, FactorTypeVector&>(), py::arg("graph"), py::arg("node_types"), R"doc(
Initializes the :class:`ConditionalSemiparametricBN` with the given ``graph``. It specifies the ``node_types`` for the
nodes.

:param graph: :class:`ConditionalDag <pybnesian.ConditionalDag>` of the conditional Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc");
    register_DerivedConditionalBayesianNetwork<ConditionalKDENetwork>(root, "ConditionalKDENetwork", R"doc(
This class implements a :class:`ConditionalBayesianNetwork` with the type :class:`KDENetworkType`.
)doc");
    register_DerivedConditionalBayesianNetwork<ConditionalDiscreteBN>(root, "ConditionalDiscreteBN", R"doc(
This class implements a :class:`ConditionalBayesianNetwork` with the type :class:`DiscreteBNType`.
)doc");

    py::class_<ConditionalHomogeneousBN, ConditionalBayesianNetwork, std::shared_ptr<ConditionalHomogeneousBN>>(
        root, "ConditionalHomogeneousBN", R"doc(
This class implements an homogeneous conditional Bayesian network. This conditional Bayesian network can be used with
any :class:`FactorType <pybnesian.FactorType>`. You can set the :class:`FactorType` in the constructor.
)doc")
        .def(py::init([](std::shared_ptr<FactorType> ft,
                         const std::vector<std::string>& nodes,
                         const std::vector<std::string>& interface_nodes) {
                 return ConditionalHomogeneousBN(FactorType::keep_python_alive(ft), nodes, interface_nodes);
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             R"doc(
Initializes the :class:`ConditionalHomogeneousBN` of ``factor_type`` with the given ``nodes`` and ``interface_nodes``.

:param factor_type: :class:`FactorType` for all the nodes.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
)doc")
        .def(py::init([](std::shared_ptr<FactorType> ft,
                         const std::vector<std::string>& nodes,
                         const std::vector<std::string>& interface_nodes,
                         const ArcStringVector& arcs) {
                 return ConditionalHomogeneousBN(FactorType::keep_python_alive(ft), nodes, interface_nodes, arcs);
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`ConditionalHomogeneousBN` of ``factor_type`` with the given ``nodes``, ``interface_nodes`` and
``arcs``.

:param factor_type: :class:`FactorType` for all the nodes.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param arcs: Arcs of the :class:`ConditionalHomogeneousBN`.
)doc")
        .def(py::init([](std::shared_ptr<FactorType> ft, const ConditionalDag& graph) {
                 return ConditionalHomogeneousBN(FactorType::keep_python_alive(ft), graph);
             }),
             py::arg("factor_type"),
             py::arg("graph"),
             R"doc(
Initializes the :class:`ConditionalHomogeneousBN` of ``factor_type`` with the given ``graph``.

:param factor_type: :class:`FactorType` for all the nodes.
:param graph: :class:`ConditionalDag <pybnesian.ConditionalDag>` of the conditional Bayesian network.)doc")
        .def(py::pickle([](const ConditionalHomogeneousBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__homogeneous_setstate__<ConditionalHomogeneousBN>(t); }));

    py::class_<ConditionalHeterogeneousBN, ConditionalBayesianNetwork, std::shared_ptr<ConditionalHeterogeneousBN>>(
        root, "ConditionalHeterogeneousBN", R"doc(
This class implements an heterogeneous conditional Bayesian network. This conditional Bayesian network accepts a
different :class:`FactorType <pybnesian.FactorType>` for each node. You can set the default :class:`FactorType`
in the constructor.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft,
                         const std::vector<std::string>& nodes,
                         const std::vector<std::string>& interface_nodes) {
                 return ConditionalHeterogeneousBN(FactorType::keep_vector_python_alive(ft), nodes, interface_nodes);
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of default ``factor_type`` with the given ``nodes`` and ``interface_nodes``.

:param factor_type: List of default :class:`FactorType` for the conditional Bayesian network.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft,
                         const std::vector<std::string>& nodes,
                         const std::vector<std::string>& interface_nodes,
                         const FactorTypeVector& node_types) {
                 return ConditionalHeterogeneousBN(FactorType::keep_vector_python_alive(ft),
                                                   nodes,
                                                   interface_nodes,
                                                   util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of default ``factor_type`` with the given ``nodes``, ``interface_nodes`` and ``node_types``.

:param factor_type: List of default :class:`FactorType` for the conditional Bayesian network.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft,
                         const std::vector<std::string>& nodes,
                         const std::vector<std::string>& interface_nodes,
                         const ArcStringVector& arcs) {
                 return ConditionalHeterogeneousBN(
                     FactorType::keep_vector_python_alive(ft), nodes, interface_nodes, arcs);
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of default ``factor_type`` with the given ``nodes``,
``interface_nodes`` and ``arcs``.

:param factor_type: List of default :class:`FactorType` for the conditional Bayesian network.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param arcs: Arcs of the :class:`ConditionalHeterogeneousBN`.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft,
                         const std::vector<std::string>& nodes,
                         const std::vector<std::string>& interface_nodes,
                         const ArcStringVector& arcs,
                         const FactorTypeVector& node_types) {
                 return ConditionalHeterogeneousBN(FactorType::keep_vector_python_alive(ft),
                                                   nodes,
                                                   interface_nodes,
                                                   arcs,
                                                   util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_type"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of default ``factor_type`` with the given ``nodes``,
``interface_nodes``, ``arcs`` and ``node_types``.

:param factor_type: List of default :class:`FactorType` for the conditional Bayesian network.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param arcs: Arcs of the :class:`ConditionalHeterogeneousBN`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft, const ConditionalDag& graph) {
                 return ConditionalHeterogeneousBN(FactorType::keep_vector_python_alive(ft), graph);
             }),
             py::arg("factor_type"),
             py::arg("graph"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of default ``factor_type`` with the given ``graph``.

:param factor_type: List of default :class:`FactorType` for the conditional Bayesian network.
:param graph: :class:`ConditionalDag <pybnesian.ConditionalDag>` of the conditional Bayesian network.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft,
                         const ConditionalDag& graph,
                         const FactorTypeVector& node_types) {
                 return ConditionalHeterogeneousBN(FactorType::keep_vector_python_alive(ft),
                                                   graph,
                                                   util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_type"),
             py::arg("graph"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of default ``factor_type`` with the given ``graph`` and ``node_types``.

:param factor_type: List of default :class:`FactorType` for the conditional Bayesian network.
:param graph: :class:`ConditionalDag <pybnesian.ConditionalDag>` of the conditional Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](MapDataToFactor fts,
                         const std::vector<std::string>& nodes,
                         const std::vector<std::string>& interface_nodes) {
                 return ConditionalHeterogeneousBN(models::keep_MapDataToFactor_alive(fts), nodes, interface_nodes);
             }),
             py::arg("factor_types"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of different default ``factor_types`` with the given ``nodes`` and
``interface_nodes``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
)doc")
        .def(py::init([](MapDataToFactor fts,
                         const std::vector<std::string>& nodes,
                         const std::vector<std::string>& interface_nodes,
                         const FactorTypeVector& node_types) {
                 return ConditionalHeterogeneousBN(models::keep_MapDataToFactor_alive(fts),
                                                   nodes,
                                                   interface_nodes,
                                                   util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_types"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of different default ``factor_types`` with the given ``nodes``, ``interface_nodes`` and ``node_types``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](MapDataToFactor fts,
                         const std::vector<std::string>& nodes,
                         const std::vector<std::string>& interface_nodes,
                         const ArcStringVector& arcs) {
                 return ConditionalHeterogeneousBN(
                     models::keep_MapDataToFactor_alive(fts), nodes, interface_nodes, arcs);
             }),
             py::arg("factor_types"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of different default ``factor_types`` with the given ``nodes``,
``interface_nodes`` and ``arcs``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param arcs: Arcs of the :class:`ConditionalHeterogeneousBN`.
)doc")
        .def(py::init([](MapDataToFactor fts,
                         const std::vector<std::string>& nodes,
                         const std::vector<std::string>& interface_nodes,
                         const ArcStringVector& arcs,
                         const FactorTypeVector& node_types) {
                 return ConditionalHeterogeneousBN(models::keep_MapDataToFactor_alive(fts),
                                                   nodes,
                                                   interface_nodes,
                                                   arcs,
                                                   util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_types"),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of different default ``factor_types`` with the given ``nodes``,
``interface_nodes``, ``arcs`` and ``node_types``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param arcs: Arcs of the :class:`ConditionalHeterogeneousBN`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init([](MapDataToFactor fts, const ConditionalDag& graph) {
                 return ConditionalHeterogeneousBN(models::keep_MapDataToFactor_alive(fts), graph);
             }),
             py::arg("factor_types"),
             py::arg("graph"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of different default ``factor_types`` with the given ``graph``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param graph: :class:`ConditionalDag <pybnesian.ConditionalDag>` of the conditional Bayesian network.
)doc")
        .def(py::init([](MapDataToFactor fts, const ConditionalDag& graph, const FactorTypeVector& node_types) {
                 return ConditionalHeterogeneousBN(models::keep_MapDataToFactor_alive(fts),
                                                   graph,
                                                   util::keep_FactorTypeVector_python_alive(node_types));
             }),
             py::arg("factor_types"),
             py::arg("graph"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalHeterogeneousBN` of different default ``factor_types`` with the given ``graph`` and
``node_types``.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param graph: :class:`ConditionalDag <pybnesian.ConditionalDag>` of the conditional Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(
            py::pickle([](const ConditionalHeterogeneousBN& self) { return self.__getstate__(); },
                       [](py::tuple& t) { return models::__heterogeneous_setstate__<ConditionalHeterogeneousBN>(t); }));

    auto conditional_clg =
        register_DerivedConditionalBayesianNetwork<ConditionalCLGNetwork>(root, "ConditionalCLGNetwork", R"doc(
This class implements a :class:`ConditionalBayesianNetwork` with the type :class:`CLGNetworkType`.
)doc");

    conditional_clg
        .def(py::init<const std::vector<std::string>&, const std::vector<std::string>&, FactorTypeVector&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalCLGNetwork` with the given ``nodes`` and ``interface_nodes``. It specifies the
``node_types`` for the nodes.

:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init<const std::vector<std::string>&,
                      const std::vector<std::string>&,
                      const ArcStringVector&,
                      FactorTypeVector&>(),
             py::arg("nodes"),
             py::arg("interface_nodes"),
             py::arg("arcs"),
             py::arg("node_types"),
             R"doc(
Initializes the :class:`ConditionalCLGNetwork` with the given ``nodes``, ``interface_nodes`` and ``arcs``. It
specifies the ``node_types`` for the nodes.

:param nodes: List of node names.
:param interface_nodes: List of interface node names.
:param arcs: Arcs of the :class:`ConditionalCLGNetwork`.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc")
        .def(py::init<const ConditionalDag&, FactorTypeVector&>(), py::arg("graph"), py::arg("node_types"), R"doc(
Initializes the :class:`ConditionalCLGNetwork` with the given ``graph``. It specifies the ``node_types`` for the
nodes.

:param graph: :class:`ConditionalDag <pybnesian.ConditionalDag>` of the conditional Bayesian network.
:param node_types: List of node type tuples (``node``, :class:`FactorType <pybnesian.FactorType>`) that
                   specifies the type for each node.
)doc");

    register_DerivedDynamicBayesianNetwork<DynamicGaussianNetwork>(root, "DynamicGaussianNetwork", R"doc(
This class implements a :class:`DynamicBayesianNetwork` with the type :class:`GaussianNetworkType`.
)doc");
    register_DerivedDynamicBayesianNetwork<DynamicSemiparametricBN>(root, "DynamicSemiparametricBN", R"doc(
This class implements a :class:`DynamicBayesianNetwork` with the type :class:`SemiparametricBNType`.
)doc");
    register_DerivedDynamicBayesianNetwork<DynamicKDENetwork>(root, "DynamicKDENetwork", R"doc(
This class implements a :class:`DynamicBayesianNetwork` with the type :class:`KDENetworkType`.
)doc");
    register_DerivedDynamicBayesianNetwork<DynamicDiscreteBN>(root, "DynamicDiscreteBN", R"doc(
This class implements a :class:`DynamicBayesianNetwork` with the type :class:`DiscreteBN`.
)doc");

    py::class_<DynamicHomogeneousBN, DynamicBayesianNetwork, std::shared_ptr<DynamicHomogeneousBN>>(
        root, "DynamicHomogeneousBN", R"doc(
This class implements an homogeneous dynamic Bayesian network. This dynamic Bayesian network can be used with
any :class:`FactorType <pybnesian.FactorType>`. You can set the :class:`FactorType` in the constructor.
)doc")
        .def(py::init(
                 [](std::shared_ptr<FactorType> ft, const std::vector<std::string>& variables, int markovian_order) {
                     return DynamicHomogeneousBN(FactorType::keep_python_alive(ft), variables, markovian_order);
                 }),
             py::arg("factor_type"),
             py::arg("variables"),
             py::arg("markovian_order"),
             R"doc(
Initializes the :class:`DynamicHomogeneousBN` of ``factor_type`` with the given ``variables`` and ``markovian_order``.
It creates empty static and transition Bayesian networks.

:param factor_type: :class:`FactorType` for all the nodes.
:param variables: List of variable names.
:param markovian_order: Markovian order of the dynamic Bayesian network.
)doc")
        .def(py::init([](const std::vector<std::string>& variables,
                         int markovian_order,
                         std::shared_ptr<BayesianNetworkBase> static_bn,
                         std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn) {
                 return DynamicHomogeneousBN(variables,
                                             markovian_order,
                                             BayesianNetworkBase::keep_python_alive(static_bn),
                                             ConditionalBayesianNetworkBase::keep_python_alive(transition_bn));
             }),
             py::arg("variables"),
             py::arg("markovian_order"),
             py::arg("static_bn"),
             py::arg("transition_bn"),
             R"doc(
Initializes the :class:`DynamicHomogeneousBN` with the given ``variables`` and ``markovian_order``. The static
and transition Bayesian networks are initialized with ``static_bn`` and ``transition_bn`` respectively.

Both ``static_bn`` and ``transition_bn`` must contain the expected nodes:

- For the static network, it must contain the nodes from ``[variable_name]_t_1`` to
  ``[variable_name]_t_[markovian_order]``.
- For the transition network, it must contain the nodes ``[variable_name]_t_0``, and the interface nodes from
  ``[variable_name]_t_1`` to ``[variable_name]_t_[markovian_order]``.

The type of ``static_bn`` and ``transition_bn`` must be :class:`HomogeneousBNType`.

:param variables: List of variable names.
:param markovian_order: Markovian order of the dynamic Bayesian network.
:param static_bn: Static Bayesian network.
:param transition_bn: Transition Bayesian network.
)doc")
        .def(py::pickle([](const DynamicHomogeneousBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__derived_dbn_setstate__<DynamicHomogeneousBN>(t); }));

    py::class_<DynamicHeterogeneousBN, DynamicBayesianNetwork, std::shared_ptr<DynamicHeterogeneousBN>>(
        root, "DynamicHeterogeneousBN", R"doc(
This class implements an heterogeneous dynamic Bayesian network. This dynamic Bayesian network accepts a
different :class:`FactorType <pybnesian.FactorType>` for each node. You can set the default :class:`FactorType`
in the constructor.
)doc")
        .def(py::init([](std::vector<std::shared_ptr<FactorType>> ft,
                         const std::vector<std::string>& variables,
                         int markovian_order) {
                 return DynamicHeterogeneousBN(FactorType::keep_vector_python_alive(ft), variables, markovian_order);
             }),
             py::arg("factor_type"),
             py::arg("variables"),
             py::arg("markovian_order"),
             R"doc(
Initializes the :class:`DynamicHeterogeneousBN` of default ``factor_type`` with the given ``variables`` and
``markovian_order``. It creates empty static and transition Bayesian networks.

:param factor_type: Default :class:`FactorType` for the dynamic Bayesian network.
:param variables: List of variable names.
:param markovian_order: Markovian order of the dynamic Bayesian network.
)doc")
        .def(py::init([](MapDataToFactor fts, const std::vector<std::string>& variables, int markovian_order) {
                 return DynamicHeterogeneousBN(models::keep_MapDataToFactor_alive(fts), variables, markovian_order);
             }),
             py::arg("factor_types"),
             py::arg("variables"),
             py::arg("markovian_order"),
             R"doc(
Initializes the :class:`DynamicHeterogeneousBN` of different default ``factor_types`` with the given ``variables`` and
``markovian_order``. It creates empty static and transition Bayesian networks.

:param factor_types: Default :class:`FactorType <pybnesian.FactorType>` for the Bayesian network for each
    different data type.
:param variables: List of variable names.
:param markovian_order: Markovian order of the dynamic Bayesian network.
)doc")
        .def(py::init([](const std::vector<std::string>& variables,
                         int markovian_order,
                         std::shared_ptr<BayesianNetworkBase> static_bn,
                         std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn) {
                 return DynamicHeterogeneousBN(variables,
                                               markovian_order,
                                               BayesianNetworkBase::keep_python_alive(static_bn),
                                               ConditionalBayesianNetworkBase::keep_python_alive(transition_bn));
             }),
             py::arg("variables"),
             py::arg("markovian_order"),
             py::arg("static_bn"),
             py::arg("transition_bn"),
             R"doc(
Initializes the :class:`DynamicHeterogeneousBN` with the given ``variables`` and ``markovian_order``. The static
and transition Bayesian networks are initialized with ``static_bn`` and ``transition_bn`` respectively.

Both ``static_bn`` and ``transition_bn`` must contain the expected nodes:

- For the static network, it must contain the nodes from ``[variable_name]_t_1`` to
  ``[variable_name]_t_[markovian_order]``.
- For the transition network, it must contain the nodes ``[variable_name]_t_0``, and the interface nodes from
  ``[variable_name]_t_1`` to ``[variable_name]_t_[markovian_order]``.

The type of ``static_bn`` and ``transition_bn`` must be :class:`HeterogeneousBNType`.

:param variables: List of variable names.
:param markovian_order: Markovian order of the dynamic Bayesian network.
:param static_bn: Static Bayesian network.
:param transition_bn: Transition Bayesian network.
)doc")
        .def(py::pickle([](const DynamicHeterogeneousBN& self) { return self.__getstate__(); },
                        [](py::tuple& t) { return models::__derived_dbn_setstate__<DynamicHeterogeneousBN>(t); }));

    register_DerivedDynamicBayesianNetwork<DynamicCLGNetwork>(root, "DynamicCLGNetwork", R"doc(
This class implements a :class:`DynamicBayesianNetwork` with the type :class:`CLGNetworkType`.
)doc");
}
