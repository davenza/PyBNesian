#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <learning/operators/operators.hpp>

namespace py = pybind11;

using learning::operators::Operator, learning::operators::ArcOperator, learning::operators::AddArc,
    learning::operators::RemoveArc, learning::operators::FlipArc, learning::operators::ChangeNodeType,
    learning::operators::OperatorTabuSet, learning::operators::OperatorSet, learning::operators::ArcOperatorSet,
    learning::operators::ChangeNodeTypeSet, learning::operators::OperatorPool;

void register_ArcOperators(py::module& m) {
    py::class_<AddArc, ArcOperator, std::shared_ptr<AddArc>>(m, "AddArc")
        .def(py::init<std::string, std::string, double>())
        .def("apply", &AddArc::apply)
        .def("opposite", &AddArc::opposite, py::return_value_policy::take_ownership);

    py::class_<RemoveArc, ArcOperator, std::shared_ptr<RemoveArc>>(m, "RemoveArc")
        .def(py::init<std::string, std::string, double>())
        .def("apply", &RemoveArc::apply)
        .def("opposite", &RemoveArc::opposite, py::return_value_policy::take_ownership);

    py::class_<FlipArc, ArcOperator, std::shared_ptr<FlipArc>>(m, "FlipArc")
        .def(py::init<std::string, std::string, double>())
        .def("apply", &FlipArc::apply)
        .def("opposite", &FlipArc::opposite, py::return_value_policy::take_ownership);
}

void register_OperatorTabuSet(py::module& m) {
    // TODO: Implement copy operation.
    py::class_<OperatorTabuSet>(m, "OperatorTabuSet")
        .def(py::init<>())
        .def(py::init<const OperatorTabuSet&>())
        .def("insert", py::overload_cast<const std::shared_ptr<Operator>&>(&OperatorTabuSet::insert))
        .def("contains", py::overload_cast<const std::shared_ptr<Operator>&>(&OperatorTabuSet::contains, py::const_))
        .def("clear", &OperatorTabuSet::clear)
        .def("empty", &OperatorTabuSet::empty);
}

template <typename OperatorBase = Operator>
class PyOperator : public OperatorBase {
public:
    using OperatorBase::OperatorBase;

    void apply(BayesianNetworkBase& m) const override {
        PYBIND11_OVERRIDE_PURE(void,         /* Return type */
                               OperatorBase, /* Parent class */
                               apply,        /* Name of function in C++ (must match Python name) */
                               &m            /* Argument(s) */
        );
    }

    std::vector<std::string> nodes_changed(const BayesianNetworkBase& m) const override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::string>, /* Return type */
                               OperatorBase,             /* Parent class */
                               nodes_changed,            /* Name of function in C++ (must match Python name) */
                               &m                        /* Argument(s) */
        );
    }

    std::vector<std::string> nodes_changed(const ConditionalBayesianNetworkBase& m) const override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::string>, /* Return type */
                               OperatorBase,             /* Parent class */
                               nodes_changed,            /* Name of function in C++ (must match Python name) */
                               &m                        /* Argument(s) */
        );
    }
    std::shared_ptr<Operator> opposite() const override {
        // PYBIND11_OVERRIDE_PURE(
        //     std::shared_ptr<Operator>, /* Return type */
        //     OperatorBase,      /* Parent class */
        //     opposite,          /* Name of function in C++ (must match Python name) */
        //                 /* Argument(s) */
        // );

        // This is needed to keep the Python instance alive.
        // See the working issue: https://github.com/pybind/pybind11/issues/1333
        // The solution provided here is adapted from:
        // https://github.com/pybind/pybind11/issues/1049#issuecomment-326688270
        {
            py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
            // Try to look up the overridden method on the Python side.
            py::function override = pybind11::get_override(this, "opposite");
            if (override) {
                auto o = override();
                auto keep_python_state_alive = std::make_shared<py::object>(o);
                auto ptr = o.cast<Operator*>();

                return std::shared_ptr<Operator>(keep_python_state_alive, ptr);
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"Operator::opposite\"");
    }

    std::shared_ptr<Operator> copy() const override {
        // PYBIND11_OVERRIDE_PURE(
        //     std::shared_ptr<Operator>, /* Return type */
        //     OperatorBase,      /* Parent class */
        //     copy,          /* Name of function in C++ (must match Python name) */
        //                 /* Argument(s) */
        // );

        // This is needed to keep the Python instance alive.
        // See the working issue: https://github.com/pybind/pybind11/issues/1333
        // The solution provided here is adapted from:
        // https://github.com/pybind/pybind11/issues/1049#issuecomment-326688270
        {
            py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
            // Try to look up the overridden method on the Python side.
            py::function override = pybind11::get_override(this, "copy");
            if (override) {
                auto o = override();
                auto keep_python_state_alive = std::make_shared<py::object>(o);
                auto ptr = o.cast<Operator*>();

                return std::shared_ptr<Operator>(keep_python_state_alive, ptr);
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"Operator::copy\"");
    }

    std::string ToString() const override {
        PYBIND11_OVERRIDE_PURE(std::string,  /* Return type */
                               OperatorBase, /* Parent class */
                               ToString,     /* Name of function in C++ (must match Python name) */
                                             /* Argument(s) */
        );
    }

    bool operator==(const Operator& a) const override {
        PYBIND11_OVERRIDE_PURE_NAME(bool,         /* Return type */
                                    OperatorBase, /* Parent class */
                                    "__eq__",
                                    operator==, /* Name of function in C++ (must match Python name) */
                                    &a          /* Argument(s) */
        );
    }

    std::size_t hash() const override {
        PYBIND11_OVERRIDE_PURE_NAME(std::int64_t, /* Return type */
                                    OperatorBase, /* Parent class */
                                    "__hash__",
                                    hash, /* Name of function in C++ (must match Python name) */
                                          /* Argument(s) */
        );
    }
};

template <typename ArcOperatorBase = ArcOperator>
class PyArcOperator : public PyOperator<ArcOperatorBase> {
public:
    using PyOperator<ArcOperatorBase>::PyOperator;

    const std::string& source() const {
        PYBIND11_OVERRIDE(const std::string&, /* Return type */
                          ArcOperatorBase,    /* Parent class */
                          source,             /* Name of function in C++ (must match Python name) */
                                              /* Argument(s) */
        );
    }

    const std::string& target() const {
        PYBIND11_OVERRIDE(const std::string&, /* Return type */
                          ArcOperatorBase,    /* Parent class */
                          target,             /* Name of function in C++ (must match Python name) */
                                              /* Argument(s) */
        );
    }
};

class PyOperatorSet : public OperatorSet {
    using OperatorSet::OperatorSet;

    void cache_scores(const BayesianNetworkBase& model, const Score& score) override {
        PYBIND11_OVERRIDE_PURE(void,         /* Return type */
                               OperatorSet,  /* Parent class */
                               cache_scores, /* Name of function in C++ (must match Python name) */
                               &model,       /* Argument(s) */
                               &score);
    }

    std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model) const override {
        // PYBIND11_OVERRIDE_PURE(
        //     std::shared_ptr<Operator>, /* Return type */
        //     OperatorSet,      /* Parent class */
        //     find_max,          /* Name of function in C++ (must match Python name) */
        //     &model       /* Argument(s) */
        // );

        // This is needed to keep the Python instance alive.
        // See the working issue: https://github.com/pybind/pybind11/issues/1333
        // The solution provided here is adapted from:
        // https://github.com/pybind/pybind11/issues/1049#issuecomment-326688270
        {
            py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
            // Try to look up the overridden method on the Python side.
            py::function override = pybind11::get_override(this, "find_max");
            if (override) {
                auto o = override(&model);
                auto keep_python_state_alive = std::make_shared<py::object>(o);
                auto ptr = o.cast<Operator*>();

                return std::shared_ptr<Operator>(keep_python_state_alive, ptr);
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::find_max\"");
    }

    std::shared_ptr<Operator> find_max(const BayesianNetworkBase& model, const OperatorTabuSet& tabu) const override {
        // PYBIND11_OVERRIDE_PURE_NAME(
        //     std::shared_ptr<Operator>, /* Return type */
        //     OperatorSet,      /* Parent class */
        //     "find_max_tabu",
        //     find_max,          /* Name of function in C++ (must match Python name) */
        //     &model,       /* Argument(s) */
        //     &tabu
        // );

        // This is needed to keep the Python instance alive.
        // See the working issue: https://github.com/pybind/pybind11/issues/1333
        // The solution provided here is adapted from:
        // https://github.com/pybind/pybind11/issues/1049#issuecomment-326688270
        {
            py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
            // Try to look up the overridden method on the Python side.
            py::function override = pybind11::get_override(this, "find_max_tabu");
            if (override) {
                auto o = override(&model, &tabu);
                auto keep_python_state_alive = std::make_shared<py::object>(o);
                auto ptr = o.cast<Operator*>();

                return std::shared_ptr<Operator>(keep_python_state_alive, ptr);
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::find_max_tabu\"");
    }

    void update_scores(const BayesianNetworkBase& model,
                       const Score& score,
                       const std::vector<std::string>& changed_nodes) override {
        PYBIND11_OVERRIDE_PURE(void,          /* Return type */
                               OperatorSet,   /* Parent class */
                               update_scores, /* Name of function in C++ (must match Python name) */
                               &model,        /* Argument(s) */
                               &score,
                               changed_nodes);
    }

    void cache_scores(const ConditionalBayesianNetworkBase& model, const Score& score) override {
        PYBIND11_OVERRIDE_PURE(void,         /* Return type */
                               OperatorSet,  /* Parent class */
                               cache_scores, /* Name of function in C++ (must match Python name) */
                               &model,       /* Argument(s) */
                               &score);
    }

    std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model) const override {
        // PYBIND11_OVERRIDE_PURE(
        //     std::shared_ptr<Operator>, /* Return type */
        //     OperatorSet,      /* Parent class */
        //     find_max,          /* Name of function in C++ (must match Python name) */
        //     &model       /* Argument(s) */
        // );

        // This is needed to keep the Python instance alive.
        // See the working issue: https://github.com/pybind/pybind11/issues/1333
        // The solution provided here is adapted from:
        // https://github.com/pybind/pybind11/issues/1049#issuecomment-326688270
        {
            py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
            // Try to look up the overridden method on the Python side.
            py::function override = pybind11::get_override(this, "find_max");
            if (override) {
                auto o = override(&model);
                auto keep_python_state_alive = std::make_shared<py::object>(o);
                auto ptr = o.cast<Operator*>();

                return std::shared_ptr<Operator>(keep_python_state_alive, ptr);
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::find_max\"");
    }

    std::shared_ptr<Operator> find_max(const ConditionalBayesianNetworkBase& model,
                                       const OperatorTabuSet& tabu) const override {
        // PYBIND11_OVERRIDE_PURE_NAME(
        //     std::shared_ptr<Operator>, /* Return type */
        //     OperatorSet,      /* Parent class */
        //     "find_max_tabu",
        //     find_max,          /* Name of function in C++ (must match Python name) */
        //     &model,       /* Argument(s) */
        //     &tabu
        // );

        // This is needed to keep the Python instance alive.
        // See the working issue: https://github.com/pybind/pybind11/issues/1333
        // The solution provided here is adapted from:
        // https://github.com/pybind/pybind11/issues/1049#issuecomment-326688270
        {
            py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
            // Try to look up the overridden method on the Python side.
            py::function override = pybind11::get_override(this, "find_max_tabu");
            if (override) {
                auto o = override(&model, &tabu);
                auto keep_python_state_alive = std::make_shared<py::object>(o);
                auto ptr = o.cast<Operator*>();

                return std::shared_ptr<Operator>(keep_python_state_alive, ptr);
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::find_max_tabu\"");
    }

    void update_scores(const ConditionalBayesianNetworkBase& model,
                       const Score& score,
                       const std::vector<std::string>& changed_nodes) override {
        PYBIND11_OVERRIDE_PURE(void,          /* Return type */
                               OperatorSet,   /* Parent class */
                               update_scores, /* Name of function in C++ (must match Python name) */
                               &model,        /* Argument(s) */
                               &score,
                               changed_nodes);
    }

    void set_arc_blacklist(const ArcStringVector& arc_blacklist) override {
        PYBIND11_OVERRIDE_PURE(void,              /* Return type */
                               OperatorSet,       /* Parent class */
                               set_arc_blacklist, /* Name of function in C++ (must match Python name) */
                               &arc_blacklist     /* Argument(s) */
        );
    }

    void set_arc_whitelist(const ArcStringVector& arc_whitelist) override {
        PYBIND11_OVERRIDE_PURE(void,              /* Return type */
                               OperatorSet,       /* Parent class */
                               set_arc_whitelist, /* Name of function in C++ (must match Python name) */
                               &arc_whitelist     /* Argument(s) */
        );
    }

    void set_max_indegree(int max_indegree) override {
        PYBIND11_OVERRIDE_PURE(void,             /* Return type */
                               OperatorSet,      /* Parent class */
                               set_max_indegree, /* Name of function in C++ (must match Python name) */
                               max_indegree      /* Argument(s) */
        );
    }

    void set_type_whitelist(const FactorTypeVector& type_whitelist) override {
        PYBIND11_OVERRIDE_PURE(void,               /* Return type */
                               OperatorSet,        /* Parent class */
                               set_type_whitelist, /* Name of function in C++ (must match Python name) */
                               &type_whitelist     /* Argument(s) */
        );
    }
};

void pybindings_operators(py::module& root) {
    auto operators = root.def_submodule("operators", "Learning operators submodule");

    py::class_<Operator, PyOperator<>, std::shared_ptr<Operator>>(operators, "Operator")
        .def(py::init<double>())
        .def_property_readonly("delta", &Operator::delta)
        // .def_property_readonly("type", &Operator::type)
        .def("apply", &Operator::apply)
        .def("nodes_changed", py::overload_cast<const BayesianNetworkBase&>(&Operator::nodes_changed, py::const_))
        .def("nodes_changed",
             py::overload_cast<const ConditionalBayesianNetworkBase&>(&Operator::nodes_changed, py::const_))
        .def("opposite", &Operator::opposite, py::return_value_policy::take_ownership)
        .def(
            "__eq__", [](const Operator& self, const Operator& other) { return self == other; }, py::is_operator())
        .def(
            "__ne__", [](const Operator& self, const Operator& other) { return self != other; }, py::is_operator());

    py::class_<ArcOperator, Operator, PyArcOperator<>, std::shared_ptr<ArcOperator>>(operators, "ArcOperator")
        .def(py::init<std::string, std::string, double>())
        .def_property_readonly("source", &ArcOperator::source)
        .def_property_readonly("target", &ArcOperator::target);

    register_ArcOperators(operators);

    py::class_<ChangeNodeType, Operator, std::shared_ptr<ChangeNodeType>>(operators, "ChangeNodeType")
        .def(py::init<std::string, std::shared_ptr<FactorType>, double>())
        .def_property_readonly("node", &ChangeNodeType::node)
        .def_property_readonly("node_type", &ChangeNodeType::node_type)
        .def("apply", &ChangeNodeType::apply)
        .def("opposite", &ChangeNodeType::opposite);

    register_OperatorTabuSet(operators);

    py::class_<OperatorSet, PyOperatorSet, std::shared_ptr<OperatorSet>>(operators, "OperatorSet")
        .def(py::init<>())
        .def("cache_scores",
             [](OperatorSet& self, BayesianNetworkBase& model, Score& score) { self.cache_scores(model, score); })
        .def("find_max", [](OperatorSet& self, BayesianNetworkBase& model) { return self.find_max(model); })
        .def("find_max",
             [](OperatorSet& self, BayesianNetworkBase& model, OperatorTabuSet& tabu) {
                 return self.find_max(model, tabu);
             })
        .def("update_scores",
             [](OperatorSet& self,
                const BayesianNetworkBase& model,
                const Score& score,
                const std::vector<std::string>& variables) { self.update_scores(model, score, variables); })
        .def("set_arc_blacklist", py::overload_cast<const ArcStringVector&>(&OperatorSet::set_arc_blacklist))
        .def("set_arc_whitelist", py::overload_cast<const ArcStringVector&>(&OperatorSet::set_arc_whitelist))
        .def("set_max_indegree", &OperatorSet::set_max_indegree)
        .def("set_type_whitelist", &OperatorSet::set_type_whitelist);

    py::class_<ArcOperatorSet, OperatorSet, std::shared_ptr<ArcOperatorSet>>(operators, "ArcOperatorSet")
        .def(py::init<ArcStringVector, ArcStringVector, int>(),
             py::arg("blacklist") = ArcStringVector(),
             py::arg("whitelist") = ArcStringVector(),
             py::arg("max_indegree") = 0);

    py::class_<ChangeNodeTypeSet, OperatorSet, std::shared_ptr<ChangeNodeTypeSet>>(operators, "ChangeNodeTypeSet")
        .def(py::init<FactorTypeVector>(), py::arg("type_whitelist") = FactorTypeVector());

    py::class_<OperatorPool, OperatorSet, std::shared_ptr<OperatorPool>>(operators, "OperatorPool")
        .def(py::init<std::vector<std::shared_ptr<OperatorSet>>>());
}
