*********
PyBNesian
*********

.. automodule:: pybnesian

Dependencies
============

- Python 3.x.
- C++17 compatible compiler.
- OpenCL 1.2 headers/library available.

The library has been tested on Ubuntu 16.04/20.04, but should be compatible with other operating systems.

Libraries
---------

The library depends on `NumPy <https://numpy.org/>`_, `Apache Arrow`_, and
`pybind11 <https://github.com/pybind/pybind11>`_.

Installation
============

PyBNesian is not uploaded to PyPI right now, but you can still install it with pip:

.. code-block:: bash

    pip install git+https://github.com/davenza/PyBNesian

If needed you can select a C++ compiler by setting the environment variable `CC`. For example, in Ubuntu, we can use
Clang 11 with the following command before installing `PyBNesian`:

.. code-block:: bash

    export CC=clang-11

Testing
=======

The library contains tests that can be executed using `pytest <https://docs.pytest.org/>`_. They also require
`scipy <https://www.scipy.org/>`_ and `pandas <https://pandas.pydata.org/>`_ installed. Install them using pip:

.. code-block:: bash

    pip install pytest scipy pandas

Run the tests with:

.. code-block:: bash

    pytest

Usage Example
=============

.. doctest:: python

    >>> from pybnesian.models import GaussianNetwork
    >>> from pybnesian.factors.continuous import LinearGaussianCPD
    >>> # Create a GaussianNetwork with 4 nodes and no arcs.
    >>> gbn = GaussianNetwork(['a', 'b', 'c', 'd'])
    >>> # Create a GaussianNetwork with 4 nodes and 3 arcs.
    >>> gbn = GaussianNetwork(['a', 'b', 'c', 'd'], [('a', 'c'), ('b', 'c'), ('c', 'd')])

    >>> # Return the nodes of the network.
    >>> print("Nodes: " + str(gbn.nodes()))
    Nodes: ['a', 'b', 'c', 'd']
    >>> # Return the arcs of the network.
    >>> print("Arcs: " + str(gbn.nodes()))
    Arcs: ['a', 'b', 'c', 'd']
    >>> # Return the parents of c.
    >>> print("Parents of c: " + str(gbn.parents('c')))
    Parents of c: ['b', 'a']
    >>> # Return the children of c.
    >>> print("Children of c: " + str(gbn.children('c')))
    Children of c: ['d']

    >>> # You can access to the graph of the network.
    >>> graph = gbn.graph()
    >>> # Return the roots of the graph.
    >>> print("Roots: " + str(sorted(graph.roots())))
    Roots: ['a', 'b']
    >>> # Return the leaves of the graph.
    >>> print("Leaves: " + str(sorted(graph.leaves())))
    Leaves: ['d']
    >>> # Return the topological sort.
    >>> print("Topological sort: " + str(graph.topological_sort()))
    Topological sort: ['a', 'b', 'c', 'd']

    >>> # Add an arc.
    >>> gbn.add_arc('a', 'b')
    >>> # Flip (reverse) an arc.
    >>> gbn.flip_arc('a', 'b')
    >>> # Remove an arc.
    >>> gbn.remove_arc('b', 'a')

    >>> # We can also add nodes.
    >>> gbn.add_node('e')
    4
    >>> # We can get the number of nodes
    >>> assert gbn.num_nodes() == 5
    >>> # ... and the number of arcs
    >>> assert gbn.num_arcs() == 3
    >>> # Remove a node.
    >>> gbn.remove_node('b')

    >>> # Each node has an unique index to identify it
    >>> print("Indices: " + str(gbn.indices()))
    Indices: {'e': 4, 'c': 2, 'd': 3, 'a': 0}
    >>> idx_a = gbn.index('a')

    >>> # And we can get the node name from the index
    >>> print("Node 2: " + str(gbn.name(2)))
    Node 2: c

    >>> # The model is not fitted right now.
    >>> assert gbn.fitted() == False

    >>> # Create a LinearGaussianCPD (variable, parents, betas, variance)
    >>> d_cpd = LinearGaussianCPD("d", ["c"], [3, 1.2], 0.5)

    >>> # Add the CPD to the GaussianNetwork
    >>> gbn.add_cpds([d_cpd])

    >>> # The CPD is still not fitted because there are 3 nodes without CPD.
    >>> assert gbn.fitted() == False

    >>> # Let's generate some random data to fit the model.
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> import pandas as pd
    >>> DATA_SIZE = 100
    >>> a_array = np.random.normal(3, np.sqrt(0.5), size=DATA_SIZE)
    >>> c_array = -4.2 - 1.2*a_array + np.random.normal(0, np.sqrt(0.75), size=DATA_SIZE)
    >>> d_array = 3 + 1.2 * c_array + np.random.normal(0, np.sqrt(0.5), size=DATA_SIZE)
    >>> e_array = np.random.normal(0, 1, size=DATA_SIZE)
    >>> df = pd.DataFrame({'a': a_array,
    ...                    'c': c_array,
    ...                    'd': d_array,
    ...                    'e': e_array
    ...                })

    >>> # Fit the model. You can pass a pandas.DataFrame or a pyarrow.RecordBatch as argument.
    >>> # This fits the remaining CPDs
    >>> gbn.fit(df)
    >>> assert gbn.fitted() == True

    >>> # Check the learned CPDs.
    >>> print(gbn.cpd('a'))
    [LinearGaussianCPD] P(a) = N(3.043, 0.396)
    >>> print(gbn.cpd('c'))
    [LinearGaussianCPD] P(c | a) = N(-4.423 + -1.083*a, 0.659)
    >>> print(gbn.cpd('d'))
    [LinearGaussianCPD] P(d | c) = N(3.000 + 1.200*c, 0.500)
    >>> print(gbn.cpd('e'))
    [LinearGaussianCPD] P(e) = N(-0.020, 1.144)

    >>> # You can sample some data
    >>> sample = gbn.sample(50)

    >>> # Compute the log-likelihood of each instance
    >>> ll = gbn.logl(sample)
    >>> # or the sum of log-likelihoods.
    >>> sll = gbn.slogl(sample)
    >>> assert np.isclose(ll.sum(), sll)

    >>> # Save the model, include the CPDs in the file.
    >>> gbn.save('test', include_cpd=True)

    >>> # Load the model
    >>> from pybnesian import load
    >>> loaded_gbn = load('test.pickle')

    >>> # Learn the structure using greedy hill-climbing.
    >>> from pybnesian.learning.algorithms import hc
    >>> from pybnesian.models import GaussianNetworkType
    >>> # Learn a Gaussian network.
    >>> learned = hc(df, bn_type=GaussianNetworkType())
    >>> learned.num_arcs()
    2

.. testcleanup::

    import os
    os.remove('test.pickle')