.. _graph-ref:

Graph Module
************

.. automodule:: pybnesian.graph

Graphs
======
All the nodes in the graph are represented by a name and are associated with a non-negative unique index.

The name can be obtained from the unique index using the method ``name()``, while the unique index can be obtained 
from the index using the method ``index()``.

Removing a node invalidates the index of the removed node, while leaving the other nodes unaffected.
When adding a node, the graph may reuse previously invalidated indices to avoid wasting too much memory.

If there are not removal of nodes in a graph, the unique indices are in the range [0-``num_nodes()``). The removal of
nodes, can lead to some indices being greater or equal to ``num_nodes()``:

.. doctest::

    >>> from pybnesian.graph import UndirectedGraph
    >>> g = UndirectedGraph(["a", "b", "c", "d"])
    >>> g.index("a")
    0
    >>> g.index("b")
    1
    >>> g.index("c")
    2
    >>> g.index("d")
    3
    >>> g.remove_node("a")
    >>> g.index("b")
    1
    >>> g.index("c")
    2
    >>> g.index("d")
    3
    >>> assert g.index("d") >= g.num_nodes()

Sometimes, this effect may be undesirable because we want to identify our nodes with a index 
in a range [0-``num_nodes()``). For this reason, there is a ``collapsed_index()`` method and other related
methods ``index_from_collapsed()``, ``collapsed_from_index()`` and ``collapsed_name()``.
Note that the collapsed index is not unique, because removing a node can change the collapsed index of at most
one other node.

.. doctest::

    >>> from pybnesian.graph import UndirectedGraph
    >>> g = UndirectedGraph(["a", "b", "c", "d"])
    >>> g.collapsed_index("a")
    0
    >>> g.collapsed_index("b")
    1
    >>> g.collapsed_index("c")
    2
    >>> g.collapsed_index("d")
    3
    >>> g.remove_node("a")
    >>> g.collapsed_index("b")
    1
    >>> g.collapsed_index("c")
    2
    >>> g.collapsed_index("d")
    0
    >>> assert all([g.collapsed_index(n) < g.num_nodes() for n in g.nodes()])

.. autoclass:: pybnesian.graph.UndirectedGraph
    :members:
    :special-members: __init__, __iter__
.. autoclass:: pybnesian.graph.DirectedGraph
    :members:
    :special-members: __init__, __iter__
.. autoclass:: pybnesian.graph.Dag
    :show-inheritance:
    :members:
    :special-members: __init__, __iter__
.. autoclass:: pybnesian.graph.PartiallyDirectedGraph
    :members:
    :special-members: __init__, __iter__
    
Conditional Graphs
==================

A conditional graph is the underlying graph in a conditional Bayesian networks ([PGM]_, Section 5.6). In a conditional
Bayesian network, only the normal nodes can have a conditional probability density, while the interface nodes are always
observed. A conditional graph splits all the nodes in two subsets: normal nodes and interface nodes. In a conditional
graph, the interface nodes can not have parents.

In a conditional graph, normal nodes can be returned with ``nodes()``, the interface nodes with
``interface_nodes()`` and the joint set of nodes with ``joint_nodes()``. Also, there are many other functions
that have the prefix ``interface`` and ``joint`` to denote the interface and joint sets of nodes. Among them, there is
a collapsed index version for only interface nodes, ``interface_collapsed_index()``, and the joint set of nodes,
``joint_collapsed_index()``. Note that the collapsed index for each set of nodes is independent.

    
.. autoclass:: pybnesian.graph.ConditionalUndirectedGraph
    :members:
    :special-members: __init__, __iter__
.. autoclass:: pybnesian.graph.ConditionalDirectedGraph
    :members:
    :special-members: __init__, __iter__
.. autoclass:: pybnesian.graph.ConditionalDag
    :show-inheritance:
    :members:
    :special-members: __init__, __iter__
.. autoclass:: pybnesian.graph.ConditionalPartiallyDirectedGraph
    :members:
    :special-members: __init__, __iter__


Bibliography
============

.. [dag2pdag] Chickering, M. (2002). Learning Equivalence Classes of Bayesian-Network Structures.
              Journal of Machine Learning Research, 2, 445-498.
.. [dag2pdag_extra] Chickering, M. (1995). A Transformational Characterization of Equivalent Bayesian Network 
                    Structures. Proceedings of the Eleventh Conference on Uncertainty in Artificial Intelligence
                    (UAI'95), Montreal.
.. [pdag2dag] Dorit, D. and Tarsi, M. (1992). A simple algorithm to construct a consistent extension of a partially
              oriented graph (Report No: R-185).

.. [PGM] Koller, D. and Friedman, N. (2009). Probabilistic Graphical Models. MIT press.