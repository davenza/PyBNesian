Serialization
*************

All the relevant objects (graphs, factors, Bayesian networks, etc) can be saved/loaded using the pickle format.

These objects can be saved using directly :func:`pickle.dump <pickle.dump>` and :func:`pickle.load <pickle.load>`. For example:

.. doctest::

    >>> import pickle
    >>> from pybnesian import Dag
    >>> g = Dag(["a", "b", "c", "d"], [("a", "b")])
    >>> with open("saved_graph.pickle", "wb") as f:
    ...     pickle.dump(g, f)
    >>> with open("saved_graph.pickle", "rb") as f:
    ...     lg = pickle.load(f)
    >>> assert lg.nodes() == ["a", "b", "c", "d"]
    >>> assert lg.arcs() == [("a", "b")]

.. testcleanup::

    import os
    os.remove('saved_graph.pickle')

We can reduce some boilerplate code using the ``save`` methods: :func:`Factor.save() <pybnesian.Factor.save>`,
:func:`UndirectedGraph.save() <pybnesian.UndirectedGraph.save>`,
:func:`DirectedGraph.save() <pybnesian.DirectedGraph.save>`,
:func:`BayesianNetworkBase.save() <pybnesian.BayesianNetworkBase.save>`, etc... Also, the :func:`load <pybnesian.load>`
can load any saved object:

.. doctest::

    >>> import pickle
    >>> from pybnesian import load, Dag
    >>> g = Dag(["a", "b", "c", "d"], [("a", "b")])
    >>> g.save("saved_graph")
    >>> lg = load("saved_graph.pickle")
    >>> assert lg.nodes() == ["a", "b", "c", "d"]
    >>> assert lg.arcs() == [("a", "b")]

.. autofunction:: pybnesian.load