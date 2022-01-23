#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <learning/operators/operators.hpp>

namespace py = pybind11;

using learning::operators::Operator, learning::operators::ArcOperator, learning::operators::AddArc,
    learning::operators::RemoveArc, learning::operators::FlipArc, learning::operators::ChangeNodeType,
    learning::operators::OperatorTabuSet, learning::operators::LocalScoreCache, learning::operators::OperatorSet,
    learning::operators::ArcOperatorSet, learning::operators::ChangeNodeTypeSet, learning::operators::OperatorPool;

void register_ArcOperators(py::module& m) {
    py::class_<AddArc, ArcOperator, std::shared_ptr<AddArc>>(m, "AddArc", R"doc(
This operator adds the arc ``source`` -> ``target``.
)doc")
        .def(py::init<std::string, std::string, double>(),
             py::arg("source"),
             py::arg("target"),
             py::arg("delta"),
             R"doc(
Initializes the :class:`AddArc` operator of the arc ``source`` -> ``target`` with delta score ``delta``.

:param source: Name of the source node.
:param target: Name of the target node.
:param delta: Delta score of the operator.
)doc");

    py::class_<RemoveArc, ArcOperator, std::shared_ptr<RemoveArc>>(m, "RemoveArc", R"doc(
This operator removes the arc ``source`` -> ``target``.
)doc")
        .def(
            py::init<std::string, std::string, double>(), py::arg("source"), py::arg("target"), py::arg("delta"), R"doc(
Initializes the :class:`RemoveArc` operator of the arc ``source`` -> ``target`` with delta score ``delta``.

:param source: Name of the source node.
:param target: Name of the target node.
:param delta: Delta score of the operator.
)doc");

    py::class_<FlipArc, ArcOperator, std::shared_ptr<FlipArc>>(m, "FlipArc", R"doc(
This operator flips (reverses) the arc ``source`` -> ``target``.
)doc")
        .def(
            py::init<std::string, std::string, double>(), py::arg("source"), py::arg("target"), py::arg("delta"), R"doc(
Initializes the :class:`FlipArc` operator of the arc ``source`` -> ``target`` with delta score ``delta``.

:param source: Name of the source node.
:param target: Name of the target node.
:param delta: Delta score of the operator.
)doc");
}

void register_OperatorTabuSet(py::module& m) {
    // TODO: Implement copy operation.
    py::class_<OperatorTabuSet>(m, "OperatorTabuSet", R"doc(
An :class:`OperatorTabuSet` that contains forbidden operators.
)doc")
        .def(py::init<>(), R"doc(
Creates an empty :class:`OperatorTabuSet`.
)doc")
        .def(
            "insert",
            [](OperatorTabuSet& self, const std::shared_ptr<Operator>& op) {
                self.insert(Operator::keep_python_alive(op));
            },
            py::arg("operator"),
            R"doc(
Inserts an operator into the tabu set.

:param operator: Operator to insert.
)doc")
        .def("contains",
             py::overload_cast<const std::shared_ptr<Operator>&>(&OperatorTabuSet::contains, py::const_),
             py::arg("operator"),
             R"doc(
Checks whether this tabu set contains ``operator``.

:param operator: The operator to be checked.
:returns: True if the tabu set contains the ``operator``, False otherwise.
)doc")
        .def("clear", &OperatorTabuSet::clear, R"doc(
Erases all the operators from the set.
)doc")
        .def("empty", &OperatorTabuSet::empty, R"doc(
Checks if the set has no operators

:returns: True if the set is empty, False otherwise.
)doc");
}

template <typename OperatorBase = Operator>
class PyOperator : public OperatorBase {
public:
    using OperatorBase::OperatorBase;

    bool is_python_derived() const override { return true; }

    void apply(BayesianNetworkBase& m) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const OperatorBase*>(this), "apply");

        if (override) {
            override(m.shared_from_this());
        } else {
            py::pybind11_fail("Tried to call pure virtual function \"Operator::apply\"");
        }
    }

    std::vector<std::string> nodes_changed(const BayesianNetworkBase& m) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const OperatorBase*>(this), "nodes_changed");

        if (override) {
            auto o = override(m.shared_from_this());

            try {
                return o.cast<std::vector<std::string>>();
            } catch (py::cast_error& e) {
                throw std::runtime_error("The returned object of Operator::nodes_changed is not a list of str.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"Operator::nodes_changed\"");
    }

    std::vector<std::string> nodes_changed(const ConditionalBayesianNetworkBase& m) const override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const OperatorBase*>(this), "nodes_changed");

        if (override) {
            auto o = override(m.shared_from_this());

            try {
                return o.cast<std::vector<std::string>>();
            } catch (py::cast_error& e) {
                throw std::runtime_error("The returned object of Operator::nodes_changed is not a list of str.");
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"Operator::nodes_changed\"");
    }
    std::shared_ptr<Operator> opposite(const BayesianNetworkBase& m) const override {
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
            py::function override = pybind11::get_override(static_cast<const OperatorBase*>(this), "opposite");
            if (override) {
                auto o = override(m.shared_from_this());

                if (o.is(py::none())) {
                    throw std::invalid_argument("Operator::opposite cannot return None.");
                }

                try {
                    auto op = o.cast<std::shared_ptr<Operator>>();
                    Operator::keep_python_alive(op);
                    return op;
                } catch (py::cast_error& e) {
                    throw std::runtime_error("The returned object of Operator::opposite is not a Operator.");
                }
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"Operator::opposite\"");
    }

    std::shared_ptr<Operator> opposite(const ConditionalBayesianNetworkBase& m) const override {
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
            py::function override = pybind11::get_override(static_cast<const OperatorBase*>(this), "opposite");
            if (override) {
                auto o = override(m.shared_from_this());

                if (o.is(py::none())) {
                    throw std::invalid_argument("Operator::opposite cannot return None.");
                }

                try {
                    auto op = o.cast<std::shared_ptr<Operator>>();
                    Operator::keep_python_alive(op);
                    return op;
                } catch (py::cast_error& e) {
                    throw std::runtime_error("The returned object of Operator::opposite is not a Operator.");
                }
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"Operator::opposite\"");
    }

    std::string ToString() const override {
        PYBIND11_OVERRIDE_PURE_NAME(std::string,  /* Return type */
                                    OperatorBase, /* Parent class */
                                    "__str__",
                                    ToString, /* Name of function in C++ (must match Python name) */
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
public:
    PyOperatorSet(bool calculate_local_score = true) : m_calculate_local_score(calculate_local_score) {}

    bool is_python_derived() const override { return true; }

    void cache_scores(const BayesianNetworkBase& model, const Score& score) override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const OperatorSet*>(this), "cache_scores");

        if (override) {
            if (m_calculate_local_score) {
                initialize_local_cache(model);

                if (owns_local_cache()) {
                    this->m_local_cache->cache_local_scores(model, score);
                }
            }

            override(model.shared_from_this(), &score);
        } else {
            py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::cache_scores\"");
        }
    }

    void cache_scores(const ConditionalBayesianNetworkBase& model, const Score& score) override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const OperatorSet*>(this), "cache_scores");

        if (override) {
            if (m_calculate_local_score) {
                initialize_local_cache(model);

                if (owns_local_cache()) {
                    this->m_local_cache->cache_local_scores(model, score);
                }
            }

            override(model.shared_from_this(), &score);
        } else {
            py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::cache_scores\"");
        }
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

        if (m_calculate_local_score) raise_uninitialized();

        {
            py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
            // Try to look up the overridden method on the Python side.
            py::function override = pybind11::get_override(static_cast<const OperatorSet*>(this), "find_max");
            if (override) {
                auto o = override(model.shared_from_this());
                try {
                    auto op = o.cast<std::shared_ptr<Operator>>();
                    Operator::keep_python_alive(op);
                    return op;
                } catch (py::cast_error& e) {
                    throw std::runtime_error("The returned object of Operator::find_max is not a Operator.");
                }
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::find_max\"");
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

        if (m_calculate_local_score) raise_uninitialized();

        {
            py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
            // Try to look up the overridden method on the Python side.
            py::function override = pybind11::get_override(static_cast<const OperatorSet*>(this), "find_max");
            if (override) {
                auto o = override(model.shared_from_this());
                try {
                    auto op = o.cast<std::shared_ptr<Operator>>();
                    Operator::keep_python_alive(op);
                    return op;
                } catch (py::cast_error& e) {
                    throw std::runtime_error("The returned object of Operator::find_max is not a Operator.");
                }
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

        if (m_calculate_local_score) raise_uninitialized();

        {
            py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
            // Try to look up the overridden method on the Python side.
            py::function override = pybind11::get_override(static_cast<const OperatorSet*>(this), "find_max_tabu");
            if (override) {
                auto o = override(model.shared_from_this(), &tabu);
                try {
                    auto op = o.cast<std::shared_ptr<Operator>>();
                    Operator::keep_python_alive(op);
                    return op;
                } catch (py::cast_error& e) {
                    throw std::runtime_error("The returned object of Operator::find_max_tabu is not a Operator.");
                }
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::find_max_tabu\"");
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

        if (m_calculate_local_score) raise_uninitialized();

        {
            py::gil_scoped_acquire gil;  // Acquire the GIL while in this scope.
            // Try to look up the overridden method on the Python side.
            py::function override = pybind11::get_override(static_cast<const OperatorSet*>(this), "find_max_tabu");
            if (override) {
                auto o = override(model.shared_from_this(), &tabu);
                try {
                    auto op = o.cast<std::shared_ptr<Operator>>();
                    Operator::keep_python_alive(op);
                    return op;
                } catch (py::cast_error& e) {
                    throw std::runtime_error("The returned object of Operator::find_max_tabu is not a Operator.");
                }
            }
        }

        py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::find_max_tabu\"");
    }

    void update_scores(const BayesianNetworkBase& model,
                       const Score& score,
                       const std::vector<std::string>& changed_nodes) override {
        if (m_calculate_local_score) raise_uninitialized();

        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const OperatorSet*>(this), "update_scores");

        if (override) {
            override(model.shared_from_this(), &score, changed_nodes);
        } else {
            py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::update_scores\"");
        }
    }

    void update_scores(const ConditionalBayesianNetworkBase& model,
                       const Score& score,
                       const std::vector<std::string>& changed_nodes) override {
        if (m_calculate_local_score) raise_uninitialized();

        pybind11::gil_scoped_acquire gil;
        pybind11::function override = pybind11::get_override(static_cast<const OperatorSet*>(this), "update_scores");

        if (override) {
            override(model.shared_from_this(), &score, changed_nodes);
        } else {
            py::pybind11_fail("Tried to call pure virtual function \"OperatorSet::update_scores\"");
        }
    }

    void set_arc_blacklist(const ArcStringVector& arc_blacklist) override {
        PYBIND11_OVERRIDE(void,              /* Return type */
                          OperatorSet,       /* Parent class */
                          set_arc_blacklist, /* Name of function in C++ (must match Python name) */
                          arc_blacklist      /* Argument(s) */
        );
    }

    void set_arc_whitelist(const ArcStringVector& arc_whitelist) override {
        PYBIND11_OVERRIDE(void,              /* Return type */
                          OperatorSet,       /* Parent class */
                          set_arc_whitelist, /* Name of function in C++ (must match Python name) */
                          arc_whitelist      /* Argument(s) */
        );
    }

    void set_max_indegree(int max_indegree) override {
        PYBIND11_OVERRIDE(void,             /* Return type */
                          OperatorSet,      /* Parent class */
                          set_max_indegree, /* Name of function in C++ (must match Python name) */
                          max_indegree      /* Argument(s) */
        );
    }

    void set_type_blacklist(const FactorTypeVector& type_blacklist) override {
        PYBIND11_OVERRIDE(void,               /* Return type */
                          OperatorSet,        /* Parent class */
                          set_type_blacklist, /* Name of function in C++ (must match Python name) */
                          type_blacklist      /* Argument(s) */
        );
    }

    void set_type_whitelist(const FactorTypeVector& type_whitelist) override {
        PYBIND11_OVERRIDE(void,               /* Return type */
                          OperatorSet,        /* Parent class */
                          set_type_whitelist, /* Name of function in C++ (must match Python name) */
                          type_whitelist      /* Argument(s) */
        );
    }

    void finished() override {
        {
            pybind11::gil_scoped_acquire gil;
            pybind11::function override = pybind11::get_override(static_cast<const OperatorSet*>(this), "finished");

            if (override) {
                override();
            }
        }

        OperatorSet::finished();
    }

private:
    bool m_calculate_local_score;
};

void pybindings_operators(py::module& root) {
    py::class_<Operator, PyOperator<>, std::shared_ptr<Operator>> op(root, "Operator", R"doc(
An operator is the representation of a change in a Bayesian network structure. Each operator has a delta score
associated that measures the difference in score when the operator is applied to the Bayesian network.
)doc");

    op.def(py::init<double>(), py::arg("delta"), R"doc(
Initializes an :class:`Operator` with a given ``delta``.

:param delta: Delta score of the operator.
)doc")
        .def("delta", &Operator::delta, R"doc(
Gets the delta score of the operator.

:returns: Delta score of the operator.
)doc")
        .def("apply", &Operator::apply, py::arg("model"), R"doc(
Apply the operator to the ``model``.

:param model: Bayesian network model.
)doc")
        .def("__str__", &Operator::ToString)
        .def(
            "__eq__",
            [](const Operator& self, const Operator& other) { return self == other; },
            py::arg("other"),
            py::is_operator())
        .def(
            "__ne__",
            [](const Operator& self, const Operator& other) { return self != other; },
            py::arg("other"),
            py::is_operator())
        .def("__hash__", &Operator::hash, R"doc(
Returns the hash value of this operator. **Two equal operators (without taking into account the delta value) must return
the same hash value.**

:returns: Hash value of ``self`` operator.
)doc");
    {
        py::options options;
        options.disable_function_signatures();

        op.def("opposite",
               py::overload_cast<const ConditionalBayesianNetworkBase&>(&Operator::opposite, py::const_),
               py::arg("model"),
               py::return_value_policy::take_ownership)
            .def("opposite",
                 py::overload_cast<const BayesianNetworkBase&>(&Operator::opposite, py::const_),
                 py::arg("model"),
                 py::return_value_policy::take_ownership,
                 R"doc(
opposite(self: pybnesian.Operator, model: BayesianNetworkBase or ConditionalBayesianNetworkBase) -> Operator

Returns an operator that reverses this :class:`Operator` given the ``model``. For example:

.. doctest::

    >>> from pybnesian import AddArc, RemoveArc, GaussianNetwork
    >>> gbn = GaussianNetwork(["a", "b"])
    >>> add = AddArc("a", "b", 1)
    >>> assert add.opposite(gbn) == RemoveArc("a", "b", -1)

:param model: The model where the ``self`` operator would be applied.
:returns: The opposite operator of ``self``.
)doc")
            .def("nodes_changed",
                 py::overload_cast<const BayesianNetworkBase&>(&Operator::nodes_changed, py::const_),
                 py::arg("model"))
            .def("nodes_changed",
                 py::overload_cast<const ConditionalBayesianNetworkBase&>(&Operator::nodes_changed, py::const_),
                 py::arg("model"),
                 R"doc(
nodes_changed(self: pybnesian.Operator, model: BayesianNetworkBase or ConditionalBayesianNetworkBase) -> List[str]

Gets the list of nodes whose local score changes when the operator is applied.

:param model: Bayesian network model.
:returns: List of nodes whose local score changes when the operator is applied.
)doc");
    }

    py::class_<ArcOperator, Operator, PyArcOperator<>, std::shared_ptr<ArcOperator>>(root, "ArcOperator", R"doc(
This class implements an operator that perfoms a change in a single arc.
)doc")
        .def(
            py::init<std::string, std::string, double>(), py::arg("source"), py::arg("target"), py::arg("delta"), R"doc(
Initializes an :class:`ArcOperator` of the arc ``source`` -> ``target``  with delta score ``delta``.

:param source: Name of the source node.
:param target: Name of the target node.
:param delta: Delta score of the operator.
)doc")
        .def("source", &ArcOperator::source, R"doc(
Gets the source of the :class:`ArcOperator`.

:returns: Name of the source node.
)doc")
        .def("target", &ArcOperator::target, R"doc(
Gets the target of the :class:`ArcOperator`.

:returns: Name of the target node.
)doc");

    register_ArcOperators(root);

    py::class_<ChangeNodeType, Operator, std::shared_ptr<ChangeNodeType>>(root, "ChangeNodeType", R"doc(
This operator changes the :class:`FactorType` of a node.
)doc")
        .def(py::init<>([](std::string node, std::shared_ptr<FactorType> node_type, double delta) {
                 return ChangeNodeType(node, FactorType::keep_python_alive(node_type), delta);
             }),
             py::arg("node"),
             py::arg("node_type"),
             py::arg("delta"),
             R"doc(
Initializes the :class:`ChangeNodeType` operator to change the type of the ``node`` to a new ``node_type``.

:param node: Name of the source node.
:param node_type: The new :class:`FactorType` of the ``node``.
:param delta: Delta score of the operator.
)doc")
        .def("node", &ChangeNodeType::node, R"doc(
Gets the node of the :class:`ChangeNodeType`.

:returns: Node of the operator.
)doc")
        .def("node_type", &ChangeNodeType::node_type, R"doc(
Gets the new :class:`FactorType` of the :class:`ChangeNodeType`.

:returns: New :class:`FactorType` of the node.
)doc");

    register_OperatorTabuSet(root);

    py::class_<LocalScoreCache, std::shared_ptr<LocalScoreCache>>(root, "LocalScoreCache", R"doc(
This class implements a cache for the local score of each node.
)doc")
        .def(py::init<>(), R"doc(
Initializes an empty :class:`LocalScoreCache`.
)doc")
        .def(py::init<const BayesianNetworkBase&>(), py::arg("model"), R"doc(
Initializes a :class:`LocalScoreCache` for the given ``model``.

:param model: A Bayesian network model.
)doc")
        .def("cache_local_scores", &LocalScoreCache::cache_local_scores, py::arg("model"), py::arg("score"), R"doc(
Caches the local score for all the nodes.

:param model: A Bayesian network model.
:param score: A :class:`Score <pybnesian.Score>` object to calculate the score.
)doc")
        .def("cache_vlocal_scores", &LocalScoreCache::cache_vlocal_scores, py::arg("model"), py::arg("score"), R"doc(
Caches the validation local score for all the nodes.

:param model: A Bayesian network model.
:param score: A :class:`ValidatedScore <pybnesian.ValidatedScore>` object to calculate the score.
)doc")
        .def("update_local_score",
             &LocalScoreCache::update_local_score,
             py::arg("model"),
             py::arg("score"),
             py::arg("node"),
             R"doc(
Updates the local score of the ``node`` in the ``model``.

:param model: A Bayesian network model.
:param score: A :class:`Score <pybnesian.Score>` object to calculate the score.
:param node: A node name.
)doc")
        .def("update_vlocal_score",
             &LocalScoreCache::update_vlocal_score,
             py::arg("model"),
             py::arg("score"),
             py::arg("node"),
             R"doc(
Updates the validation local score of the ``node`` in the ``model``.

:param model: A Bayesian network model.
:param score: A :class:`ValidatedScore <pybnesian.ValidatedScore>` object to calculate the score.
:param node: A node name.
)doc")
        .def("sum", &LocalScoreCache::sum, R"doc(
Sums the local score for all the variables.

:returns: Total score.
)doc")
        .def("local_score", &LocalScoreCache::local_score, py::arg("model"), py::arg("node"), R"doc(
Returns the local score of the ``node`` in the ``model``.

:param model: A Bayesian network model.
:param node: A node name.
:returns: Local score of ``node`` in ``model``.
)doc");

    py::class_<OperatorSet, PyOperatorSet, std::shared_ptr<OperatorSet>>(root, "OperatorSet", R"doc(
The :class:`OperatorSet` coordinates a set of operators. It caches/updates the score of each operator in the set and
finds the operator with the best score.
)doc")
        .def(py::init<bool>(), py::arg("calculate_local_cache") = true, R"doc(
Initializes an :class:`OperatorSet`.

If ``calculate_local_cache`` is True, a :class:`LocalScoreCache` is automatically
initialized when :func:`OperatorSet.cache_scores` is called. Also, the local score cache is automatically updated on
each :func:`OperatorSet.update_scores` call. Therefore, the local score cache is always updated. You can always get the
local score cache using :func:`OperatorSet.local_score_cache`. The local score values can be accessed using
:func:`LocalScoreCache.local_score`.

If ``calculate_local_cache`` is False, there is no local cache.

:param calculate_local_cache: If True automatically initializes and updates a :class:`LocalScoreCache`.
)doc")
        .def("local_score_cache", &OperatorSet::local_score_cache, R"doc(
Returns the current :class:`LocalScoreCache` of this :class:`OperatorSet`.

:returns: :class:`LocalScoreCache` of this operator set.
)doc")
        .def(
            "cache_scores",
            [](OperatorSet& self, BayesianNetworkBase& model, Score& score) { self.cache_scores(model, score); },
            py::arg("model"),
            py::arg("score"),
            R"doc(
Caches the delta score values of each operator in the set.

:param model: Bayesian network model.
:param score: The :class:`Score <pybnesian.Score>` object to cache the scores.
)doc")
        .def(
            "find_max",
            [](OperatorSet& self, BayesianNetworkBase& model) { return self.find_max(model); },
            py::arg("model"),
            R"doc(
Finds the best operator in the set to apply to the ``model``. This function must not return an invalid operator:

- An operator that creates cycles.
- An operator that contradicts blacklists, whitelists or max indegree.

If no valid operator is available in the set, it returns ``None``.

:param model: Bayesian network model.
:returns: The best valid operator, or ``None`` if there is no valid operator.
)doc")
        .def(
            "find_max_tabu",
            [](OperatorSet& self, BayesianNetworkBase& model, OperatorTabuSet& tabu) {
                return self.find_max(model, tabu);
            },
            py::arg("model"),
            py::arg("tabu_set"),
            R"doc(
This method is similar to :func:`OperatorSet.find_max`, but it also receives a ``tabu_set`` of operators.

This method must not return an operator in the ``tabu_set`` in addition to the restrictions of
:func:`OperatorSet.find_max`.

:param model: Bayesian network model.
:param tabu_set: Tabu set of operators.
:returns: The best valid operator, or ``None`` if there is no valid operator.
)doc")
        .def(
            "update_scores",
            [](OperatorSet& self,
               const BayesianNetworkBase& model,
               const Score& score,
               const std::vector<std::string>& variables) { self.update_scores(model, score, variables); },
            py::arg("model"),
            py::arg("score"),
            py::arg("changed_nodes"),
            R"doc(
Updates the delta score values of the operators in the set after applying an operator in the ``model``.
``changed_nodes`` determines the nodes whose local score has changed after applying the operator.

:param model: Bayesian network model.
:param score: The :class:`Score <pybnesian.Score>` object to cache the scores.
:param changed_nodes: The nodes whose local score has changed.
)doc")
        .def("set_arc_blacklist",
             py::overload_cast<const ArcStringVector&>(&OperatorSet::set_arc_blacklist),
             py::arg("arc_blacklist"),
             R"doc(
Sets the arc blacklist (a list of arcs that cannot be added).

:param arc_blacklist: The list of blacklisted arcs.
)doc")
        .def("set_arc_whitelist",
             py::overload_cast<const ArcStringVector&>(&OperatorSet::set_arc_whitelist),
             py::arg("arc_whitelist"),
             R"doc(
Sets the arc whitelist (a list of arcs that are forced).

:param arc_whitelist: The list of whitelisted arcs.
)doc")
        .def("set_max_indegree", &OperatorSet::set_max_indegree, py::arg("max_indegree"), R"doc(
Sets the max indegree allowed. This may change the set of valid operators.

:param max_indegree: Max indegree allowed.
)doc")
        .def(
            "set_type_blacklist",
            [](OperatorSet& self, const FactorTypeVector& type_blacklist) {
                self.set_type_blacklist(util::keep_FactorTypeVector_python_alive(type_blacklist));
            },
            py::arg("type_blacklist"),
            R"doc(
Sets the type blacklist (a list of :class:`FactorType` that are not allowed).

:param type_blacklist: The list of blacklisted :class:`FactorType`.
)doc")
        .def(
            "set_type_whitelist",
            [](OperatorSet& self, const FactorTypeVector& type_whitelist) {
                self.set_type_whitelist(util::keep_FactorTypeVector_python_alive(type_whitelist));
            },
            py::arg("type_whitelist"),
            R"doc(
Sets the type whitelist (a list of :class:`FactorType` that are forced).

:param type_whitelist: The list of whitelisted :class:`FactorType`.
)doc")
        .def("finished", &OperatorSet::finished, R"doc(
Marks the finalization of the algorithm. It clears the state of the object, so
:func:`OperatorSet.cache_scores` can be called again.
)doc");

    py::class_<ArcOperatorSet, OperatorSet, std::shared_ptr<ArcOperatorSet>>(root, "ArcOperatorSet", R"doc(
This set of operators contains all the operators related with arc changes (:class:`AddArc`, :class:`RemoveArc`,
:class:`FlipArc`)
)doc")
        .def(py::init<ArcStringVector, ArcStringVector, int>(),
             py::arg("blacklist") = ArcStringVector(),
             py::arg("whitelist") = ArcStringVector(),
             py::arg("max_indegree") = 0,
             R"doc(
Initializes an :class:`ArcOperatorSet` with optional sets of arc blacklists/whitelists and maximum indegree.

:param blacklist: List of blacklisted arcs.
:param whitelist: List of whitelisted arcs.
:param max_indegree: Max indegree allowed.
)doc");

    py::class_<ChangeNodeTypeSet, OperatorSet, std::shared_ptr<ChangeNodeTypeSet>>(root, "ChangeNodeTypeSet", R"doc(
This set of operators contains all the possible operators of type :class:`ChangeNodeType`.
)doc")
        .def(py::init<>([](FactorTypeVector type_blacklist, FactorTypeVector type_whitelist) {
                 return ChangeNodeTypeSet(util::keep_FactorTypeVector_python_alive(type_blacklist),
                                          util::keep_FactorTypeVector_python_alive(type_whitelist));
             }),
             py::arg("type_blacklist") = FactorTypeVector(),
             py::arg("type_whitelist") = FactorTypeVector(),
             R"doc(
Initializes a :class:`ChangeNodeTypeSet` with blacklisted and whitelisted :class:`FactorType`.

:param type_blacklist: The list of blacklisted :class:`FactorType`.
:param type_whitelist: The list of whitelisted :class:`FactorType`.
)doc");

    py::class_<OperatorPool, OperatorSet, std::shared_ptr<OperatorPool>>(root, "OperatorPool", R"doc(
This set of operators can join a list of :class:`OperatorSet`, so that they can act as a single :class:`OperatorSet`.
)doc")
        .def(py::init<>([](std::vector<std::shared_ptr<OperatorSet>> opsets) {
                 return OperatorPool(OperatorSet::keep_vector_python_alive(opsets));
             }),
             py::arg("opsets"),
             R"doc(
Initializes an :class:`OperatorPool` with a list of :class:`OperatorSet`.

:param opsets: List of :class:`OperatorSet`.
)doc");
}
