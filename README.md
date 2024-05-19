![build](https://img.shields.io/github/actions/workflow/status/davenza/PyBNesian/release.yml?branch=master)
[![Documentation Status](https://readthedocs.org/projects/pybnesian/badge/?version=latest)](https://pybnesian.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/pybnesian?color=blue)

# PyBNesian

- `PyBNesian` is a Python package that implements Bayesian networks. Currently, it is mainly dedicated to learning Bayesian networks.

- `PyBNesian` is implemented in C++, to achieve significant performance gains. It uses [Apache Arrow](https://arrow.apache.org/) to enable fast interoperability between Python and C++. In addition, some parts are implemented in OpenCL to achieve GPU acceleration.

- `PyBNesian` allows extending its functionality using Python code, so new research can be easily developed.


Implementation
=====================

Currently `PyBNesian` implements the following features:

Models
-----------

- [x] Bayesian networks.

- [x] Conditional Bayesian networks (see section 5.6 of [1]).

- [x] Dynamic Bayesian networks.

which can have different types of CPDs:

- [x] Multinomial.

- [x] Linear Gaussian.

- [x] Conditional kernel density estimation (ratio of two kernel density estimation models). Accelerated with OpenCL.

with this combinations of CPDs, we implement the following types of networks (which can also be Conditional or Dynamic):

- [x] Discrete networks.

- [x] Gaussian networks.

- [x] Semiparametric networks.

- [x] Hybrid networks (conditional linear Gaussian networks and semiparametric networks).

Graphs
-----------------

- [x] DAGs.

- [x] Directed graphs.

- [x] Undirected graphs.

- [x] Partially directed graphs.

Graph classes implement useful functionalities for probabilistic graphical models, such as moving between DAG-PDAG representation or fast access to root and leaves.

Learning
---------------

It implements different structure learning algorithms:

- [x] Greedy hill-climbing (for Bayesian networks and Conditional Bayesian networks).

- [x] PC-stable (for Bayesian networks and Conditional Bayesian networks).

- [x] MMPC (for Bayesian networks and Conditional Bayesian networks).

- [x] MMHC (for Bayesian networks and conditional Bayesian networks).

- [x] DMMHC (for dynamic Bayesian networks).

The score and search algorithms can be used with the following scores:

- [x] BIC.

- [x] BGe.

- [x] BDe.

- [x] Cross-validation likelihood.

- [x] Holdout likelihood.

- [x] Cross-validated likelihood with validation dataset. This score combines the cross-validation likelihood with a validation dataset to control the overfitting.

and the following the following learning operators:

- [x] Arc operations: add arc, remove arc, flip arc.

- [x] Change Node Type (for semiparametric Bayesian networks).

The following independence tests are implemented for the constraint-based algorithms:

- [x] Chi-square test.

- [x] partial correlation test t-test.

- [x] A likelihood-ratio test based on mutual information assuming a Gaussian distribution for the continuous data.

- [x] CMIknn [2].

- [x] RCoT [3].

It also implements the parameter learning:

- [x] Maximum Likelihood Estimator.

Inference
-----------------------

Not implemented right now, as the priority is the learning algorithms. However, all the CPDs and models have a `sample()` method, which can be used to create easily an approximate inference engine based on sampling.

Serialization
-----------------------

All relevant objects (graphs, CPDs, Bayesian networks, etc) can be saved/loaded using the pickle format.

Other implementations
-----------------

`PyBNesian` exposes the implementation of other models or techniques used within the library.

- [x] Apply cross-validation to a dataset.

- [x] Apply holdout to a dataset.

- [x] Kernel Density Estimation. Accelerated with OpenCL.

- [ ] K-d Tree. (implemented but not exposed yet).

Weighted sums of chi-squared random variables:

- [ ] Hall-Buckley-Eagleson approximation. (implemented but not exposed yet).

- [ ] Lindsay-Pilla-Basak approximation. (implemented but not exposed yet).

Usage example
===========================

```python
>>> from pybnesian import GaussianNetwork, LinearGaussianCPD
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
>>> from pybnesian import hc, GaussianNetworkType
>>> # Learn a Gaussian network.
>>> learned = hc(df, bn_type=GaussianNetworkType())
>>> learned.num_arcs()
2
```

Dependencies
============

- Python 3.8, 3.9, 3.10, 3.11 and 3.12.

The library has been tested on Ubuntu 16.04/20.04/22.04 and Windows 10/11, but should be compatible with other operating systems.

Libraries
---------

The library depends on [NumPy](https://numpy.org/), [Apache Arrow](https://arrow.apache.org/), [pybind11](https://github.com/pybind/pybind11), [NLopt](https://nlopt.readthedocs.io/en/latest/), [libfort](https://github.com/seleznevae/libfort) and [Boost](https://www.boost.org/).


Installation
============

PyBNesian can be installed with pip:

```
pip install pybnesian
```
Build from Source
=================

Prerequisites
-------------

- Python 3.8, 3.9, 3.10, 3.11 or 3.12.
- C++17 compatible compiler.
- CMake.
- Git.
- OpenCL drivers installed.


Building
--------

Clone the repository:

```
git clone https://github.com/davenza/PyBNesian.git
cd PyBNesian
git checkout v0.5.0 # You can checkout a specific version if you want
pip install .
```

Testing
=========================

The library contains tests that can be executed using `pytest`. They also require `scipy` and `pandas` installed.

``
pip install pytest scipy pandas
``

Run the tests with:

``
pytest
``

## References
<a id="1">[1]</a> 
D. Koller and N. Friedman, 
Probabilistic Graphical Models: Principles and Techniques,
The MIT Press, 2009.

<a id="2">[2]</a> 
J. Runge, 
Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information. International Conference on Artificial Intelligence and Statistics, AISTATS 2018, 84, 2018, pp. 938–947.

<a id="3">[3]</a> 
E. V. Strobl and K. Zhang and S., Visweswaran. Approximate kernel-based conditional independence tests for fast non-parametric causal discovery. Journal of Causal Inference, 7(1), 2019, pp 1-24.