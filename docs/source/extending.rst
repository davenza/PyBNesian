Extending PyBNesian from Python
*******************************

PyBNesian is completely implemented in C++ for better performance. However, some functionality might not be yet
implemented.

PyBNesian allows extending its functionality easily using Python code. This extension code can interact smoothly with
the C++ implementation, so that we can reuse most of the current implemented models or algorithms. Also, C++ code is
usually much faster than Python, so reusing the implementation also provides performance improvements.

Almost all components of the library can be extended:

- Factors: to include new conditional probability distributions.
- Models: to include new types of Bayesian network models.
- Independence tests: to include new conditional independence tests.
- Learning scores: to include new learning scores.
- Learning operators: to include new operators.
- Learning callbacks: callback function on each iteration of
  :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`.

The extended functionality can be used exactly equal to the base functionality.

.. note::

    You should avoid re-implementing the base functionality using extensions. Extension code is usually worse
    in performance for two reasons:
    
    - Usually, the Python code is slower than C++ (unless you have a really good implementation!).
    - Crossing the Python<->C++ boundary has a performance cost. Reducing the transition between languages is always
      good for performance

For all the extensible components, the strategy is always to implement an abstract class.

.. warning::
    .. _warning-constructor:

    All the classes that need to be inherited are developed in C++. For this reason, in the constructor of the new
    classes it is always necessary to explicitly call the constructor of the parent class. This should be the first line
    of the constructor.
    
    For example, when inheriting from
    :class:`FactorType <pybnesian.FactorType>`, **DO NOT DO this:**

    .. code-block::

        class NewFactorType(FactorType):
            def __init__(self):
                # Some code in the constructor
    
    The following code is correct:

    .. code-block::

        class NewFactorType(FactorType):
        def __init__(self):
            FactorType.__init__(self)
            # Some code in the constructor

    Check the constructor details of the abstract classes in the :ref:`API-reference` to make sure you call the parent
    constructor with the correct parameters.

    If you have forgotten to call the parent constructor, the following error message will be displayed when creating a
    new object (for pybind11>=2.6):

    .. code-block::
        
        >>> t = NewFactorType()
        TypeError: pybnesian.FactorType.__init__() must be called when overriding __init__

Factor Extension
================

Implementing a new factor usually involves creating two new classes that inherit from
:class:`FactorType <pybnesian.FactorType>` and :class:`Factor <pybnesian.Factor>`. A
:class:`FactorType <pybnesian.FactorType>` is the representation of a
:class:`Factor <pybnesian.Factor>` type. A :class:`Factor <pybnesian.Factor>` is an specific instance of
a factor (a conditional probability distribution for a given variable and evidence).

These two classes are
usually related: a :class:`FactorType <pybnesian.FactorType>` can create instances of
:class:`Factor <pybnesian.Factor>` (with :func:`FactorType.new_factor() <pybnesian.FactorType.new_factor>`),
and a :class:`Factor <pybnesian.Factor>` returns its corresponding :class:`FactorType <pybnesian.FactorType>`
(with :func:`Factor.type() <pybnesian.Factor.type>`).

A new :class:`FactorType <pybnesian.FactorType>` need to implement the following methods:

- :func:`FactorType.__str__() <pybnesian.FactorType.__str__>`.
- :func:`FactorType.new_factor() <pybnesian.FactorType.new_factor>`.

A new :class:`Factor <pybnesian.Factor>` need to implement the following methods:

- :func:`Factor.__str__() <pybnesian.Factor.__str__>`.
- :func:`Factor.type() <pybnesian.Factor.type>`.
- :func:`Factor.fitted() <pybnesian.Factor.fitted>`.
- :func:`Factor.fit() <pybnesian.Factor.fit>`. This method is needed for
  :func:`BayesianNetworkBase.fit() <pybnesian.BayesianNetworkBase.fit>` or
  :func:`DynamicBayesianNetworkBase.fit() <pybnesian.DynamicBayesianNetworkBase.fit>`.
- :func:`Factor.logl() <pybnesian.Factor.logl>`. This method is needed for
  :func:`BayesianNetworkBase.logl() <pybnesian.BayesianNetworkBase.logl>` or
  :func:`DynamicBayesianNetworkBase.logl() <pybnesian.DynamicBayesianNetworkBase.logl>`.
- :func:`Factor.slogl() <pybnesian.Factor.slogl>`. This method is needed for
  :func:`BayesianNetworkBase.slogl() <pybnesian.BayesianNetworkBase.slogl>` or
  :func:`DynamicBayesianNetworkBase.slogl() <pybnesian.DynamicBayesianNetworkBase.slogl>`.
- :func:`Factor.sample() <pybnesian.Factor.sample>`. This method is needed for
  :func:`BayesianNetworkBase.sample() <pybnesian.BayesianNetworkBase.sample>` or
  :func:`DynamicBayesianNetworkBase.sample() <pybnesian.DynamicBayesianNetworkBase.sample>`.
- :func:`Factor.data_type() <pybnesian.Factor.data_type>`. This method is needed for
  :func:`DynamicBayesianNetworkBase.sample() <pybnesian.DynamicBayesianNetworkBase.sample>`.

You can avoid implementing some of these methods if you do not need them. If a method is needed for a functionality
but it is not implemented, an error message is shown when trying to execute that functionality:

.. code-block::

    Tried to call pure virtual function Class::method

To illustrate, we will create an alternative implementation of a linear Gaussian CPD.

.. _my-lg:

.. code-block:: python
    
    import numpy as np
    from scipy.stats import norm
    import pyarrow as pa
    from pybnesian import FactorType, Factor, CKDEType

    # Define our Factor type
    class MyLGType(FactorType):
        def __init__(self):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            FactorType.__init__(self)
        
        # The __str__ is also used in __repr__ by default.
        def __str__(self):
            return "MyLGType"
        
        # Create the factor instance defined below.
        def new_factor(self, model, variable, evidence):
            return MyLG(variable, evidence)
        
    class MyLG(Factor):
        def __init__(self, variable, evidence):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            # The variable and evidence are accessible through self.variable() and self.evidence().
            Factor.__init__(self, variable, evidence)
            self._fitted = False
            self.beta = np.empty((1 + len(evidence),))
            self.variance = -1

        def __str__(self):
            if self._fitted:
                return "MyLG(beta: " + str(self.beta) + ", variance: " + str(self.variance) + ")"
            else:
                return "MyLG(unfitted)"

        def data_type(self):
            return pa.float64()

        def fit(self, df):
            pandas_df = df.to_pandas()

            # Run least squares to train the linear regression
            restricted_df = pandas_df.loc[:, [self.variable()] + self.evidence()].dropna()
            numpy_variable = restricted_df.loc[:, self.variable()].to_numpy()
            numpy_evidence =  restricted_df.loc[:, self.evidence()].to_numpy()
            linregress_data = np.column_stack((np.ones(numpy_evidence.shape[0]), numpy_evidence))
            (self.beta, res, _, _) = np.linalg.lstsq(linregress_data, numpy_variable, rcond=None)
            self.variance = res[0] / (linregress_data.shape[0] - 1)
            # Model fitted
            self._fitted = True

        def fitted(self):
            return self._fitted

        def logl(self, df):
            pandas_df = df.to_pandas()

            expected_means = self.beta[0] + np.sum(self.beta[1:] * pandas_df.loc[:,self.evidence()], axis=1)
            return norm.logpdf(pandas_df.loc[:,self.variable()], expected_means, np.sqrt(self.variance))

        def sample(self, n, evidence, seed):
            pandas_df = df.to_pandas()

            expected_means = self.beta[0] + np.sum(self.beta[1:] * pandas_df.loc[:,self.evidence()], axis=1)
            return np.random.normal(expected_means, np.sqrt(self.variance))

        def slogl(self, df):
            return self.logl(df).sum()

        def type(self):
            return MyLGType()

.. _factor-extension-serialization:

Serialization
-------------

All the factors can be saved using pickle with the method :func:`Factor.save() <pybnesian.Factor.save>`. The class
:class:`Factor <pybnesian.Factor>` already provides a ``__getstate__`` and ``__setstate__``  implementation that
saves the base information (variable name and evidence variable names). If you need to save more data in your class,
there are two alternatives:

- Implement the methods :func:`Factor.__getstate_extra__()` and :func:`Factor.__setstate_extra__()`. These methods have the
  the same restrictions as the ``__getstate__`` and ``__setstate__`` methods (the returned objects must be pickleable).

- Re-implement the :func:`Factor.__getstate__()` and :func:`Factor.__setstate__()` methods. Note, however, that it is
  needed to call the parent class constructor explicitly in :func:`Factor.__setstate__()` (as in
  :ref:`warning constructor <warning-constructor>`). This is needed to initialize the C++ part of the object. Also, you
  will need to add yourself the base information.

For example, if we want to implement serialization support for our re-implementation of linear Gaussian CPD, we can add
the following code:

.. code-block::

    class MyLG(Factor):
        #
        # Previous code
        #

        def __getstate_extra__(self):
            return {'fitted': self._fitted,
                    'beta': self.beta,
                    'variance': self.variance}

        def __setstate_extra__(self, extra):
            self._fitted = extra['fitted']
            self.beta = extra['beta']
            self.variance = extra['variance']

Alternatively, the following code will also work correctly:

.. code-block::

    class MyLG(Factor):
        #
        # Previous code
        #

        def __getstate__(self):
            # Make sure to include the variable and evidence.
            return {'variable': self.variable(),
                    'evidence': self.evidence(),
                    'fitted': self._fitted,
                    'beta': self.beta,
                    'variance': self.variance}

        def __setstate__(self, extra):
            # Call the parent constructor always in __setstate__ !
            Factor.__init__(self, extra['variable'], extra['evidence'])
            self._fitted = extra['fitted']
            self.beta = extra['beta']
            self.variance = extra['variance']


Using Extended Factors
----------------------

The extended factors can not be used in some specific networks: A
:class:`GaussianNetwork <pybnesian.GaussianNetwork>` only admits
:class:`LinearGaussianCPDType <pybnesian.LinearGaussianCPDType>`, a
:class:`SemiparametricBN <pybnesian.SemiparametricBN>` admits
:class:`LinearGaussianCPDType <pybnesian.LinearGaussianCPDType>` or
:class:`CKDEType <pybnesian.CKDEType>`, and so on...

If you try to use :class:`MyLG` in a Gaussian network, a ``ValueError`` is raised.

.. testsetup::

    import numpy as np
    from scipy.stats import norm
    import pyarrow as pa
    from pybnesian import FactorType, Factor, CKDEType

    # Define our Factor type
    class MyLGType(FactorType):
        def __init__(self):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            FactorType.__init__(self)
        
        # The __str__ is also used in __repr__ by default.
        def __str__(self):
            return "MyLGType"
        
        # Create the factor instance defined below.
        def new_factor(self, model, variable, evidence):
            return MyLG(variable, evidence)
        
    class MyLG(Factor):
        def __init__(self, variable, evidence):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            # The variable and evidence are accessible through self.variable() and self.evidence().
            Factor.__init__(self, variable, evidence)
            self._fitted = False
            self.beta = np.empty((1 + len(evidence),))
            self.variance = -1

        def __str__(self):
            if self._fitted:
                return "MyLG(beta: " + str(self.beta) + ", variance: " + str(self.variance) + ")"
            else:
                return "MyLG(unfitted)"

        def data_type(self):
            return pa.float64()

        def fit(self, df):
            pandas_df = df.to_pandas()

            restricted_df = pandas_df.loc[:, [self.variable()] + self.evidence()].dropna()
            numpy_variable = restricted_df.loc[:, self.variable()].to_numpy()
            numpy_evidence =  restricted_df.loc[:, self.evidence()].to_numpy()

            linregress_data = np.column_stack((np.ones(numpy_evidence.shape[0]), numpy_evidence))

            (self.beta, res, _, _) = np.linalg.lstsq(linregress_data, numpy_variable, rcond=None)
            self.variance = res[0] / (linregress_data.shape[0] - 1 - len(self.evidence()))
            self._fitted = True

        def fitted(self):
            return self._fitted

        def logl(self, df):
            pandas_df = df.to_pandas()

            expected_means = self.beta[0] + np.sum(self.beta[1:] * pandas_df.loc[:,self.evidence()], axis=1)
            return norm.logpdf(pandas_df.loc[:,self.variable()], expected_means, np.sqrt(self.variance))

        def sample(self, n, evidence, seed):
            pandas_df = df.to_pandas()

            expected_means = self.beta[0] + np.sum(self.beta[1:] * pandas_df.loc[:,self.evidence()], axis=1)
            return np.random.normal(expected_means, np.sqrt(self.variance))

        def slogl(self, df):
            return self.logl(df).sum()

        def type(self):
            return MyLGType()

        def __getstate_extra__(self):
            return {'fitted': self._fitted,
                    'beta': self.beta,
                    'variance': self.variance}

        def __setstate_extra__(self, extra):
            self._fitted = extra['fitted']
            self.beta = extra['beta']
            self.variance = extra['variance']

.. doctest::

    >>> from pybnesian import GaussianNetwork
    >>> g = GaussianNetwork(["a", "b", "c", "d"])
    >>> g.set_node_type("a", MyLGType())
    Traceback (most recent call last):
    ...
    ValueError: Wrong factor type "MyLGType" for node "a" in Bayesian network type "GaussianNetworkType"

There are two alternatives to use an extended :class:`Factor <pybnesian.Factor>`:

- Create an extended model (see :ref:`model-extension`) that admits the new extended
  :class:`Factor <pybnesian.Factor>`.
- Use a generic Bayesian network like :class:`HomogeneousBN <pybnesian.HomogeneousBN>` and
  :class:`HeterogeneousBN <pybnesian.HeterogeneousBN>`.

The :class:`HomogeneousBN <pybnesian.HomogeneousBN>` and
:class:`HeterogeneousBN <pybnesian.HeterogeneousBN>` Bayesian networks admit any
:class:`FactorType <pybnesian.FactorType>`. The difference between them is that
:class:`HomogeneousBN <pybnesian.HomogeneousBN>` is homogeneous
(all the nodes have the same :class:`FactorType <pybnesian.FactorType>`) and
:class:`HeterogeneousBN <pybnesian.HeterogeneousBN>` is heterogeneous (each node can have a different
:class:`FactorType <pybnesian.FactorType>`).

Our extended factor :class:`MyLG` can be used with an :class:`HomogeneousBN <pybnesian.HomogeneousBN>` to create
and alternative implementation of a :class:`GaussianNetwork <pybnesian.GaussianNetwork>`:

.. doctest::

    >>> import pandas as pd
    >>> from pybnesian import HomogeneousBN, GaussianNetwork
    >>> # Create some multivariate normal sample data
    >>> def generate_sample_data(size, seed=0):
    ...     np.random.seed(seed)
    ...     a_array = np.random.normal(3, 0.5, size=size)
    ...     b_array = np.random.normal(2.5, 2, size=size)
    ...     c_array = -4.2 + 1.2*a_array + 3.2*b_array + np.random.normal(0, 0.75, size=size)
    ...     d_array = 1.5 - 0.3 * c_array + np.random.normal(0, 0.5, size=size)
    ...     return pd.DataFrame({'a': a_array, 'b': b_array, 'c': c_array, 'd': d_array})
    >>> df = generate_sample_data(300)
    >>> df_test = generate_sample_data(20, seed=1)
    >>> # Create an HomogeneousBN and fit it
    >>> homo = HomogeneousBN(MyLGType(), ["a", "b", "c", "d"], [("a", "c")])
    >>> homo.fit(df)
    >>> # Create a GaussianNetwork and fit it
    >>> gbn = GaussianNetwork(["a", "b", "c", "d"], [("a", "c")])
    >>> gbn.fit(df)
    >>> # Check parameters
    >>> def check_parameters(cpd1, cpd2):
    ...     assert np.all(np.isclose(cpd1.beta, cpd2.beta))
    ...     assert np.isclose(cpd1.variance, cpd2.variance)
    >>> # Check the parameters for all CPDs.
    >>> check_parameters(homo.cpd("a"), gbn.cpd("a"))
    >>> check_parameters(homo.cpd("b"), gbn.cpd("b"))
    >>> check_parameters(homo.cpd("c"), gbn.cpd("c"))
    >>> check_parameters(homo.cpd("d"), gbn.cpd("d"))
    >>> # Check the log-likelihood.
    >>> assert np.all(np.isclose(homo.logl(df_test), gbn.logl(df_test)))
    >>> assert np.isclose(homo.slogl(df_test), gbn.slogl(df_test))

The extended factor can also be used in an heterogeneous Bayesian network. For example, we can imitate the behaviour
of a :class:`SemiparametricBN <pybnesian.SemiparametricBN>` using an
:class:`HeterogeneousBN <pybnesian.HeterogeneousBN>`:

.. testsetup::

    import numpy as np
    import pandas as pd
    def generate_sample_data(size, seed=0):
        np.random.seed(seed)
        a_array = np.random.normal(3, 0.5, size=size)
        b_array = np.random.normal(2.5, 2, size=size)
        c_array = -4.2 + 1.2*a_array + 3.2*b_array + np.random.normal(0, 0.75, size=size)
        d_array = 1.5 - 0.3 * c_array + np.random.normal(0, 0.5, size=size)
        return pd.DataFrame({'a': a_array, 'b': b_array, 'c': c_array, 'd': d_array})
        
    def check_parameters(cpd1, cpd2):
        assert np.all(np.isclose(cpd1.beta, cpd2.beta))
        assert np.isclose(cpd1.variance, cpd2.variance)

.. doctest::

    >>> from pybnesian import HeterogeneousBN, CKDEType, SemiparametricBN
    >>> df = generate_sample_data(300)
    >>> df_test = generate_sample_data(20, seed=1)
    >>> # Create an heterogeneous with "MyLG" factors as default.
    >>> het = HeterogeneousBN(MyLGType(),  ["a", "b", "c", "d"], [("a", "c")])
    >>> het.set_node_type("a", CKDEType())
    >>> het.fit(df)
    >>> # Create a SemiparametricBN
    >>> spbn = SemiparametricBN(["a", "b", "c", "d"], [("a", "c")], [("a", CKDEType())])
    >>> spbn.fit(df)
    >>> # Check the parameters of the CPDs
    >>> check_parameters(het.cpd("b"), spbn.cpd("b"))
    >>> check_parameters(het.cpd("c"), spbn.cpd("c"))
    >>> check_parameters(het.cpd("d"), spbn.cpd("d"))
    >>> # Check the log-likelihood.
    >>> assert np.all(np.isclose(het.logl(df_test), spbn.logl(df_test)))
    >>> assert np.isclose(het.slogl(df_test), spbn.slogl(df_test))

The :class:`HeterogeneousBN <pybnesian.HeterogeneousBN>` can also be instantiated using a dict to specify
different default factor types for different data types. For example, we can mix the :class:`MyLG` factor with
:class:`DiscreteFactor <pybnesian.DiscreteFactor>` for discrete data:

.. doctest::

    >>> import pyarrow as pa
    >>> import pandas as pd
    >>> from pybnesian import HeterogeneousBN, CKDEType, DiscreteFactorType, SemiparametricBN

    >>> def generate_hybrid_sample_data(size, seed=0):
    ...     np.random.seed(seed)
    ...     a_array = np.random.normal(3, 0.5, size=size)
    ...     b_categories = np.asarray(['b1', 'b2'])
    ...     b_array = b_categories[np.random.choice(b_categories.size, size, p=[0.5, 0.5])]
    ...     c_array = -4.2 + 1.2 * a_array + np.random.normal(0, 0.75, size=size)
    ...     d_array = 1.5 - 0.3 * c_array + np.random.normal(0, 0.5, size=size)
    ...     return pd.DataFrame({'a': a_array,
    ...                          'b': pd.Series(b_array, dtype='category'),
    ...                          'c': c_array,
    ...                          'd': d_array})

    >>> df = generate_hybrid_sample_data(20)
    >>> # Create an heterogeneous with "MyLG" factors as default for continuous data and
    >>> # "DiscreteFactorType" for categorical data.
    >>> het = HeterogeneousBN({pa.float64(): MyLGType(),
    ...                        pa.float32(): MyLGType(),
    ...                        pa.dictionary(pa.int8(), pa.utf8()): DiscreteFactorType()},
    ...                        ["a", "b", "c", "d"],
    ...                        [("a", "c")])
    >>> het.set_node_type("a", CKDEType())
    >>> het.fit(df)
    >>> assert het.node_type('a') == CKDEType()
    >>> assert het.node_type('b') == DiscreteFactorType()
    >>> assert het.node_type('c') == MyLGType()
    >>> assert het.node_type('d') == MyLGType()

.. _model-extension:

Model Extension
===============

Implementing a new model Bayesian network model involves creating a class that inherits from
:class:`BayesianNetworkType <pybnesian.BayesianNetworkType>`.  Optionally, you also might want to inherit from
:class:`BayesianNetwork <pybnesian.BayesianNetwork>`,
:class:`ConditionalBayesianNetwork <pybnesian.ConditionalBayesianNetwork>`
and :class:`DynamicBayesianNetwork <pybnesian.DynamicBayesianNetwork>`.

A :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>` is the representation of a Bayesian network model.
This is similar to the relation between :class:`FactorType <pybnesian.FactorType>` and a factor. The 
:class:`BayesianNetworkType <pybnesian.BayesianNetworkType>` defines the restrictions and properties that
characterise a Bayesian network model. A :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>` is used by
all the variants of Bayesian network models: :class:`BayesianNetwork <pybnesian.BayesianNetwork>`,
:class:`ConditionalBayesianNetwork <pybnesian.ConditionalBayesianNetwork>`
and :class:`DynamicBayesianNetwork <pybnesian.DynamicBayesianNetwork>`. For this reason, the constructors
:func:`BayesianNetwork.__init__() <pybnesian.BayesianNetwork.__init__>`,
:func:`ConditionalBayesianNetwork.__init__() <pybnesian.ConditionalBayesianNetwork.__init__>`
:func:`DynamicBayesianNetwork.__init__() <pybnesian.DynamicBayesianNetwork.__init__>` take the underlying
:class:`BayesianNetworkType <pybnesian.BayesianNetworkType>` as parameter. Thus, once a new 
:class:`BayesianNetworkType <pybnesian.BayesianNetworkType>` is implemented, you can use your new Bayesian model
with the three variants automatically.

Implementing a :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>` requires to implement the following
methods:

- :func:`BayesianNetworkType.__str__() <pybnesian.BayesianNetworkType.__str__>`.
- :func:`BayesianNetworkType.is_homogeneous() <pybnesian.BayesianNetworkType.is_homogeneous>`.
- :func:`BayesianNetworkType.default_node_type() <pybnesian.BayesianNetworkType.default_node_type>`. This method
  is optional. It is only needed for homogeneous Bayesian networks.
- :func:`BayesianNetworkType.data_default_node_type() <pybnesian.BayesianNetworkType.data_default_node_type>`. 
  This method is optional. It is only needed for non-homogeneous Bayesian networks.
- :func:`BayesianNetworkType.compatible_node_type() <pybnesian.BayesianNetworkType.compatible_node_type>`. This
  method is optional. It is only needed for non-homogeneous Bayesian networks. If not implemented, it accepts any
  :class:`FactorType <pybnesian.FactorType>` for each node.
- :func:`BayesianNetworkType.can_have_arc() <pybnesian.BayesianNetworkType.can_have_arc>`. This
  method is optional. If not implemented, it accepts any arc.
- :func:`BayesianNetworkType.new_bn() <pybnesian.BayesianNetworkType.new_bn>`.
- :func:`BayesianNetworkType.new_cbn() <pybnesian.BayesianNetworkType.new_cbn>`.
- :func:`BayesianNetworkType.alternative_node_type() <pybnesian.BayesianNetworkType.alternative_node_type>`.
  This method is optional. This method is needed to learn a Bayesian network structure with
  :class:`ChangeNodeTypeSet <pybnesian.ChangeNodeTypeSet>`. This method is only needed for
  non-homogeneous Bayesian networks.

To illustrate, we will create a Gaussian network that only admits arcs ``source`` -> ``target`` where
``source`` contains the letter "a". To make the example more interesting we will also use our custom implementation 
:class:`MyLG <my-lg>` (:ref:`in the previous section <my-lg>`).

.. code-block::

    from pybnesian import BayesianNetworkType

    class MyRestrictedGaussianType(BayesianNetworkType):
        def __init__(self):
            # Remember to call the parent constructor.
            BayesianNetworkType.__init__(self)

        # The __str__ is also used in __repr__ by default.
        def __str__(self):
            return "MyRestrictedGaussianType"

        def is_homogeneous(self):
            return True
        
        def default_node_type(self):
            return MyLGType()

        # NOT NEEDED because it is homogeneous. If heterogeneous we would return
        # the default node type for the data_type.
        # def data_default_node_type(self, data_type):
        #     if data_type.equals(pa.float64()) or data_type.equals(pa.float32()):
        #         return MyLGType()
        #     else:
        #         raise ValueError("Wrong data type for MyRestrictedGaussianType")
        #    
        # NOT NEEDED because it is homogeneous. If heterogeneous we would check
        # that the node type is correct.
        # def compatible_node_type(self, model, node):
        #    return self.node_type(node) == MyLGType or self.node_type(node) == ...

        def can_have_arc(self, model, source, target):
            # Our restriction for arcs.
            return "a" in source.lower()

        def new_bn(self, nodes):
            return BayesianNetwork(MyRestrictedGaussianType(), nodes)

        def new_cbn(self, nodes, interface_nodes):
            return ConditionalBayesianNetwork(MyRestrictedGaussianType(), nodes, interface_nodes)

        # NOT NEEDED because it is homogeneous. Also, it is not needed if you do not want to change the node type.
        # def alternative_node_type(self, node):
        #    pass
        
The arc restrictions defined by
:func:`BayesianNetworkType.can_have_arc() <pybnesian.BayesianNetworkType.can_have_arc>` can be an alternative to
the blacklist lists in some learning algorithms. However, this arc restrictions are applied always:

.. testsetup::

    from pybnesian import BayesianNetworkType

    class MyRestrictedGaussianType(BayesianNetworkType):
        def __init__(self):
            # Remember to call the parent constructor.
            BayesianNetworkType.__init__(self)

        # The __str__ is also used in __repr__ by default.
        def __str__(self):
            return "MyRestrictedGaussianType"

        def is_homogeneous(self):
            return True
        
        def default_node_type(self):
            return MyLGType()

        # NOT NEEDED because it is homogeneous. If heterogeneous we would check
        # that the node type is correct.
        # def compatible_node_type(self, model, node):
        #    return self.node_type(node) == MyLGType or self.node_type(node) == ...

        def can_have_arc(self, model, source, target):
            # Our restriction for arcs.
            return "a" in source.lower()

        def new_bn(self, nodes):
            return BayesianNetwork(MyRestrictedGaussianType(), nodes)

        def new_cbn(self, nodes, interface_nodes):
            return ConditionalBayesianNetwork(MyRestrictedGaussianType(), nodes, interface_nodes)

        # NOT NEEDED because it is homogeneous. Also, it is not needed if you do not want to change the node type.
        # def alternative_node_type(self, node):
        #    pass

.. doctest::

    >>> from pybnesian import BayesianNetwork
    >>> g = BayesianNetwork(MyRestrictedGaussianType(), ["a", "b", "c", "d"])
    >>> g.add_arc("a", "b") # This is OK
    >>> g.add_arc("b", "c") # Not allowed
    Traceback (most recent call last):
    ...
    ValueError: Cannot add arc b -> c.
    >>> g.add_arc("c", "a") # Also, not allowed
    Traceback (most recent call last):
    ...
    ValueError: Cannot add arc c -> a.
    >>> g.flip_arc("a", "b") # Not allowed, because it would generate a b -> a arc.
    Traceback (most recent call last):
    ...
    ValueError: Cannot flip arc a -> b.

Creating Bayesian Network Types
-------------------------------

:class:`BayesianNetworkType <pybnesian.BayesianNetworkType>` can adapt the behavior of a Bayesian network
with a few lines of code. However, you may want to create your own Bayesian network class instead of directly using a
:class:`BayesianNetwork <pybnesian.BayesianNetwork>`, 
a :class:`ConditionalBayesianNetwork <pybnesian.ConditionalBayesianNetwork>`
or a :class:`DynamicBayesianNetwork <pybnesian.DynamicBayesianNetwork>`. This has some advantages:

- The source code can be better organized using a different class for each Bayesian network model.
- Using ``type(model)`` over different types of models would return a different type:

.. doctest::
    
    >>> from pybnesian import GaussianNetworkType, BayesianNetwork
    >>> g1 = BayesianNetwork(GaussianNetworkType(), ["a", "b", "c", "d"])
    >>> g2 = BayesianNetwork(MyRestrictedGaussianType(), ["a", "b", "c", "d"])
    >>> assert type(g1) == type(g2) # The class type is the same, but the code would be
    >>>                             # more obvious if it weren't.
    >>> assert g1.type() != g2.type() # You have to use this.

- It allows more customization of the Bayesian network behavior.

To create your own Bayesian network, you have to inherit from
:class:`BayesianNetwork <pybnesian.BayesianNetwork>`, 
:class:`ConditionalBayesianNetwork <pybnesian.ConditionalBayesianNetwork>`
or :class:`DynamicBayesianNetwork <pybnesian.DynamicBayesianNetwork>`:

.. code-block::

    from pybnesian import BayesianNetwork, ConditionalBayesianNetwork,\
                                 DynamicBayesianNetwork

    class MyRestrictedBN(BayesianNetwork):
        def __init__(self, nodes, arcs=None):
            # You can initialize with any BayesianNetwork.__init__ constructor.
            if arcs is None:
                BayesianNetwork.__init__(self, MyRestrictedGaussianType(), nodes)
            else:
                BayesianNetwork.__init__(self, MyRestrictedGaussianType(), nodes, arcs)
            
    class MyConditionalRestrictedBN(ConditionalBayesianNetwork):
        def __init__(self, nodes, interface_nodes, arcs=None):
            # You can initialize with any ConditionalBayesianNetwork.__init__ constructor.
            if arcs is None:
                ConditionalBayesianNetwork.__init__(self, MyRestrictedGaussianType(), nodes,
                                                    interface_nodes)
            else:
                ConditionalBayesianNetwork.__init__(self, MyRestrictedGaussianType(), nodes,
                                                    interface_nodes, arcs)
            
    class MyDynamicRestrictedBN(DynamicBayesianNetwork):
        def __init__(self, variables, markovian_order):
            # You can initialize with any DynamicBayesianNetwork.__init__ constructor.
            DynamicBayesianNetwork.__init__(self, MyRestrictedGaussianType(), variables,
                                            markovian_order)

Also, it is recommended to change the 
:func:`BayesianNetworkType.new_bn() <pybnesian.BayesianNetworkType.new_bn>`
and :func:`BayesianNetworkType.new_cbn() <pybnesian.BayesianNetworkType.new_cbn>` definitions:

.. code-block::

    class MyRestrictedGaussianType(BayesianNetworkType):
        #
        # Previous code
        #

        def new_bn(self, nodes):
            return MyRestrictedBN(nodes)

        def new_cbn(self, nodes, interface_nodes):
            return MyConditionalRestrictedBN(nodes, interface_nodes)


.. testsetup::

    from pybnesian import BayesianNetwork, ConditionalBayesianNetwork,\
                                 DynamicBayesianNetwork

    class MyRestrictedBN(BayesianNetwork):
        def __init__(self, nodes, arcs=None):
            # You can initialize with any BayesianNetwork.__init__ constructor.
            if arcs is None:
                BayesianNetwork.__init__(self, MyRestrictedGaussianType(), nodes)
            else:
                BayesianNetwork.__init__(self, MyRestrictedGaussianType(), nodes, arcs)

        def add_arc(self, source, target):
            print("Adding arc " + source + " -> " + target)
            # Call the base functionality
            BayesianNetwork.add_arc(self, source, target)
        
    class MyConditionalRestrictedBN(ConditionalBayesianNetwork):
        def __init__(self, nodes, interface_nodes, arcs=None):
            # You can initialize with any ConditionalBayesianNetwork.__init__ constructor.
            if arcs is None:
                ConditionalBayesianNetwork.__init__(self, MyRestrictedGaussianType(), nodes,
                                                    interface_nodes)
            else:
                ConditionalBayesianNetwork.__init__(self, MyRestrictedGaussianType(), nodes,
                                                    interface_nodes, arcs)
            
    class MyDynamicRestrictedBN(DynamicBayesianNetwork):
        def __init__(self, variables, markovian_order):
            # You can initialize with any DynamicBayesianNetwork.__init__ constructor.
            DynamicBayesianNetwork.__init__(self, MyRestrictedGaussianType(), variables,
                                            markovian_order)

    from pybnesian import BayesianNetworkType

    class MyRestrictedGaussianType(BayesianNetworkType):
        def __init__(self):
            # Remember to call the parent constructor.
            BayesianNetworkType.__init__(self)

        # The __str__ is also used in __repr__ by default.
        def __str__(self):
            return "MyRestrictedGaussianType"

        def is_homogeneous(self):
            return True
        
        def default_node_type(self):
            return MyLGType()

        # NOT NEEDED because it is homogeneous. If heterogeneous we would check
        # that the node type is correct.
        # def compatible_node_type(self, model, node):
        #    return self.node_type(node) == MyLGType or self.node_type(node) == ...

        def can_have_arc(self, model, source, target):
            # Our restriction for arcs.
            return "a" in source.lower()

        def new_bn(self, nodes):
            return MyRestrictedBN(nodes)

        def new_cbn(self, nodes, interface_nodes):
            return MyConditionalRestrictedBN(nodes, interface_nodes)

Creating your own Bayesian network classes allows you to overload the base functionality. Thus, you can customize
completely the behavior of your Bayesian network. For example, we can print a message each time an arc is added:

.. code-block::

    class MyRestrictedBN(BayesianNetwork):
        #
        # Previous code
        #

        def add_arc(self, source, target):
            print("Adding arc " + source + " -> " + target)
            # Call the base functionality
            BayesianNetwork.add_arc(self, source, target)


.. doctest::

    >>> bn = MyRestrictedBN(["a", "b", "c", "d"])
    >>> bn.add_arc("a", "c")
    Adding arc a -> c
    >>> assert bn.has_arc("a", "c")

.. note::

    :class:`BayesianNetwork <pybnesian.BayesianNetwork>`, 
    :class:`ConditionalBayesianNetwork <pybnesian.ConditionalBayesianNetwork>`
    and :class:`DynamicBayesianNetwork <pybnesian.DynamicBayesianNetwork>` are not abstract classes. These
    classes provide an implementation for the abstract classes
    :class:`BayesianNetworkBase <pybnesian.BayesianNetworkBase>`, 
    :class:`ConditionalBayesianNetworkBase <pybnesian.ConditionalBayesianNetworkBase>`
    or :class:`DynamicBayesianNetworkBase <pybnesian.DynamicBayesianNetworkBase>`.

Serialization
-------------

The Bayesian network models can be saved using pickle with the
:func:`BayesianNetworkBase.save() <pybnesian.BayesianNetworkBase.save>` method. This method saves the structure
of the Bayesian network and, optionally, the factors within the Bayesian network. When the
:func:`BayesianNetworkBase.save() <pybnesian.BayesianNetworkBase.save>` is called,
:attr:`.BayesianNetworkBase.include_cpd` property is first set and then ``__getstate__()`` is called. ``__getstate__()``
saves the factors within the Bayesian network model only if :attr:`.BayesianNetworkBase.include_cpd` is ``True``. The
factors can be saved only if the :class:`Factor <pybnesian.Factor>` is also plickeable (see
:ref:`Factor serialization <factor-extension-serialization>`).

As with factor serialization, an implementation of ``__getstate__()`` and ``__setstate__()`` is provided when
inheriting from :class:`BayesianNetwork <pybnesian.BayesianNetwork>`,
:class:`ConditionalBayesianNetwork <pybnesian.ConditionalBayesianNetwork>`
or :class:`DynamicBayesianNetwork <pybnesian.DynamicBayesianNetwork>`. This implementation saves:

- The underlying graph of the Bayesian network.
- The underlying :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>`.
- The list of :class:`FactorType <pybnesian.FactorType>` for each node.
- The list of :class:`Factor <pybnesian.Factor>` within the Bayesian network (if
  :attr:`.BayesianNetworkBase.include_cpd` is ``True``).

In the case of :class:`DynamicBayesianNetwork <pybnesian.DynamicBayesianNetwork>`, it saves the above list for
both the static and transition networks.

If your extended Bayesian network class need to save more data, there are two alternatives:

- Implement the methods ``__getstate_extra__()`` and ``__setstate_extra__()``. These methods have the
  the same restrictions as the ``__getstate__()`` and ``__setstate__()`` methods (the returned objects must be
  pickleable).

.. code-block::

    class MyRestrictedBN(BayesianNetwork):
        #
        # Previous code
        #

        def __getstate_extra__(self):
            # Save some extra data.
            return {'extra_data': self.extra_data}

        def __setstate_extra__(self, d):
            # Here, you can access the extra data. Initialize the attributes that you need
            self.extra_data = d['extra_data']



- Re-implement the ``__getstate__()`` and ``__setstate__()`` methods. Note, however, that it is needed to call the
  parent class constructor explicitly in the ``__setstate__()`` method (as in
  :ref:`warning constructor <warning-constructor>`). This is needed to initialize the C++ part of the object. Also, you
  will need to add yourself the base information.


  .. code-block::

    class MyRestrictedBN(BayesianNetwork):
        #
        # Previous code
        #

        def __getstate__(self):
        d = {'graph': self.graph(),
             'type': self.type(),
             # You can omit this line if type is homogeneous
             'factor_types': list(self.node_types().items()),
             'extra_data': self.extra_data}

        if self.include_cpd:
            factors = []

            for n in self.nodes():
                if self.cpd(n) is not None:
                    factors.append(self.cpd(n))
            d['factors'] = factors

        return d

    def __setstate__(self, d):
        # Call the parent constructor always in __setstate__ !
        BayesianNetwork.__init__(self, d['type'], d['graph'], d['factor_types'])

        if "factors" in d:
            self.add_cpds(d['factors'])

        # Here, you can access the extra data.
        self.extra_data = d['extra_data']

The same strategy is used to implement serialization in
:class:`ConditionalBayesianNetwork <pybnesian.ConditionalBayesianNetwork>`
and :class:`DynamicBayesianNetwork <pybnesian.DynamicBayesianNetwork>`.

.. warning::

    Some functionalities require to make copies of Bayesian network models. Copying Bayesian network models
    is currently implemented using this serialization suppport. Therefore, it is highly recommended to implement
    ``__getstate_extra__()``/``__setstate_extra__()`` or ``__getstate__()``/``__setstate__()``. Otherwise, the
    extra information defined in the extended classes would be lost.

Independence Test Extension
===========================

Implementing a new conditional independence test involves creating a class that inherits from
:class:`IndependenceTest <pybnesian.IndependenceTest>`.

A new :class:`IndependenceTest <pybnesian.IndependenceTest>` needs to implement the following
methods:

- :func:`IndependenceTest.num_variables() <pybnesian.IndependenceTest.num_variables>`.
- :func:`IndependenceTest.variable_names() <pybnesian.IndependenceTest.variable_names>`.
- :func:`IndependenceTest.has_variables() <pybnesian.IndependenceTest.has_variables>`.
- :func:`IndependenceTest.name() <pybnesian.IndependenceTest.name>`.
- :func:`IndependenceTest.pvalue() <pybnesian.IndependenceTest.pvalue>`.

To illustrate, we will implement a conditional independence test that has perfect information about the
conditional indepencences (an oracle independence test):

.. code-block::

    from pybnesian import IndependenceTest

    class OracleTest(IndependenceTest):

        # An Oracle class that represents the independences of this Bayesian network:
        #
        #  "a"     "b"
        #    \     /
        #     \   /
        #      \ /
        #       V
        #      "c"
        #       |
        #       |
        #       V
        #      "d"
              
        def __init__(self):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            IndependenceTest.__init__(self)
            self.variables = ["a", "b", "c", "d"]

        def num_variables(self):
            return len(self.variables)
        
        def variable_names(self):
            return self.variables

        def has_variables(self, vars): 
            return set(vars).issubset(set(self.variables))

        def name(self, index):
            return self.variables[index]

        def pvalue(self, x, y, z):
            if z is None:
                # a _|_ b
                if set([x, y]) == set(["a", "b"]):
                    return 1
                else:
                    return 0
            else:
                z = list(z)
                if "c" in z:
                    # a _|_ d | "c" in Z
                    if set([x, y]) == set(["a", "d"]):
                        return 1
                    # b _|_ d | "c" in Z
                    if set([x, y]) == set(["b", "d"]):
                        return 1
                return 0

The oracle version of the PC algorithm guarantees the return of the correct network structure. We can use our new oracle
independence test with the :class:`PC <pybnesian.PC>` algorithm.

.. testsetup::

    from pybnesian import IndependenceTest

    class OracleTest(IndependenceTest):

        # An Oracle class that represents the independences of this Bayesian network:
        #
        #  "a"     "b"
        #    \     /
        #     \   /
        #      \ /
        #       V
        #      "c"
        #       |
        #       |
        #       V
        #      "d"
              
        def __init__(self):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            IndependenceTest.__init__(self)
            self.variables = ["a", "b", "c", "d"]

        def num_variables(self):
            return len(self.variables)
        
        def variable_names(self):
            return self.variables

        def has_variables(self, vars): 
            return set(vars).issubset(set(self.variables))

        def name(self, index):
            return self.variables[index]

        def pvalue(self, x, y, z):
            if z is None:
                # a _|_ b
                if set([x, y]) == set(["a", "b"]):
                    return 1
                else:
                    return 0
            else:
                z = list(z)
                if "c" in z:
                    # a _|_ d | "c" in Z
                    if set([x, y]) == set(["a", "d"]):
                        return 1
                    # b _|_ d | "c" in Z
                    if set([x, y]) == set(["b", "d"]):
                        return 1
                return 0

.. doctest::

    >>> from pybnesian import PC
    >>> pc = PC()
    >>> oracle = OracleTest()
    >>> graph = pc.estimate(oracle)
    >>> assert set(graph.arcs()) == {('a', 'c'), ('b', 'c'), ('c', 'd')}
    >>> assert graph.num_edges() == 0

To learn dynamic Bayesian networks your class has to override
:class:`DynamicIndependenceTest <pybnesian.DynamicIndependenceTest>`. A new
:class:`DynamicIndependenceTest <pybnesian.DynamicIndependenceTest>` needs to implement the
following methods:

- :func:`DynamicIndependenceTest.num_variables() <pybnesian.DynamicIndependenceTest.num_variables>`.
- :func:`DynamicIndependenceTest.variable_names() <pybnesian.DynamicIndependenceTest.variable_names>`.
- :func:`DynamicIndependenceTest.has_variables() <pybnesian.DynamicIndependenceTest.has_variables>`.
- :func:`DynamicIndependenceTest.name() <pybnesian.DynamicIndependenceTest.name>`.
- :func:`DynamicIndependenceTest.markovian_order() <pybnesian.DynamicIndependenceTest.markovian_order>`.
- :func:`DynamicIndependenceTest.static_tests() <pybnesian.DynamicIndependenceTest.static_tests>`.
- :func:`DynamicIndependenceTest.transition_tests() <pybnesian.DynamicIndependenceTest.transition_tests>`.

Usually, your extended :class:`IndependenceTest <pybnesian.IndependenceTest>` will use data.
It is easy to implement a related :class:`DynamicIndependenceTest <pybnesian.DynamicIndependenceTest>` by
taking a :class:`DynamicDataFrame <pybnesian.DynamicDataFrame>` as parameter and using the methods
:func:`DynamicDataFrame.static_df() <pybnesian.DynamicDataFrame.static_df>` and
:func:`DynamicDataFrame.transition_df() <pybnesian.DynamicDataFrame.transition_df>` to implement
:func:`DynamicIndependenceTest.static_tests() <pybnesian.DynamicIndependenceTest.static_tests>`
and :func:`DynamicIndependenceTest.transition_tests() <pybnesian.DynamicIndependenceTest.transition_tests>`
respectively.

Learning Scores Extension
=========================

Implementing a new learning score involves creating a class that inherits from
:class:`Score <pybnesian.Score>` or :class:`ValidatedScore <pybnesian.ValidatedScore>`.
The score must be decomposable.

The :class:`ValidatedScore <pybnesian.ValidatedScore>` is an
:class:`Score <pybnesian.Score>` that is evaluated in two different data sets: a training dataset and a
validation dataset.

An extended :class:`Score <pybnesian.Score>` class needs to implement the following methods:

- :func:`Score.has_variables() <pybnesian.Score.has_variables>`.
- :func:`Score.compatible_bn() <pybnesian.Score.compatible_bn>`.
- :func:`Score.score() <pybnesian.Score.score>`. This method is optional. The default
  implementation sums the local score for all the nodes.
- :func:`Score.local_score() <pybnesian.Score.local_score>`. Only the version with 3 arguments
  ``score.local_score(model, variable, evidence)`` needs to be implemented. The version with 2 arguments can not be
  overriden.
- :func:`Score.local_score_node_type() <pybnesian.Score.local_score_node_type>`. This method is
  optional. This method is only needed if the score is used together with
  :class:`ChangeNodeTypeSet <pybnesian.ChangeNodeTypeSet>`.
- :func:`Score.data() <pybnesian.Score.data>`. This method is optional. It is needed to infer the
  default node types in the :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>` algorithm.

In addition, an extended :class:`ValidatedScore <pybnesian.ValidatedScore>` class needs to implement the
following methods to get the score in the validation dataset:

- :func:`ValidatedScore.vscore() <pybnesian.ValidatedScore.vscore>`. This method is optional. The
  default implementation sums the validation local score for all the nodes.
- :func:`ValidatedScore.vlocal_score() <pybnesian.ValidatedScore.vlocal_score>`. Only the version with 3
  arguments ``score.vlocal_score(model, variable, evidence)`` needs to be implemented. The version with 2 arguments can
  not be overriden.
- :func:`ValidatedScore.vlocal_score_node_type() <pybnesian.ValidatedScore.vlocal_score_node_type>`.
  This method is optional. This method is only needed if the score is used together with
  :class:`ChangeNodeTypeSet <pybnesian.ChangeNodeTypeSet>`.

To illustrate, we will implement an oracle score that only returns positive score to the arcs ``a`` -> ``c``,
``b`` -> ``c`` and ``c`` -> ``d``.

.. code-block::

    from pybnesian import Score

    class OracleScore(Score):

        # An oracle class that returns positive scores for the arcs in the 
        # following Bayesian network:
        #
        #  "a"     "b"
        #    \     /
        #     \   /
        #      \ /
        #       V
        #      "c"
        #       |
        #       |
        #       V
        #      "d"

        def __init__(self):
            Score.__init__(self)
            self.variables = ["a", "b", "c", "d"]

        def has_variables(self, vars):
            return set(vars).issubset(set(self.variables))

        def compatible_bn(self, model):
            return self.has_variables(model.nodes())

        def local_score(self, model, variable, evidence):
            if variable == "c":
                v = -1
                if "a" in evidence:
                    v += 1
                if "b" in evidence:
                    v += 1.5
                return v
            elif variable == "d" and evidence == ["c"]:
                return 1
            else:
                return -1

        # NOT NEEDED because this score does not use data.
        # In that case, this method can return None or you can avoid implementing this method.
        def data(self):
            return None

We can use this new score, for example, with a
:class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`.

.. testsetup::

    from pybnesian import Score

    class OracleScore(Score):

        # An oracle class that returns positive scores for the arcs in the following Bayesian network:
        #
        #  "a"     "b"
        #    \     /
        #     \   /
        #      \ /
        #       V
        #      "c"
        #       |
        #       |
        #       V
        #      "d"

        def __init__(self):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            Score.__init__(self)
            self.variables = ["a", "b", "c", "d"]

        def has_variables(self, vars):
            return set(vars).issubset(set(self.variables))

        def compatible_bn(self, model):
            return self.has_variables(model.nodes())

        def local_score(self, model, variable, evidence):
            if variable == "c":
                v = -1
                if "a" in evidence:
                    v += 1
                if "b" in evidence:
                    v += 1.5
                return v
            elif variable == "d" and evidence == ["c"]:
                return 1
            else:
                return -1

.. doctest::

    >>> from pybnesian import GaussianNetwork, GreedyHillClimbing, ArcOperatorSet
    >>>
    >>> hc = GreedyHillClimbing()
    >>> start_model = GaussianNetwork(["a", "b", "c", "d"])
    >>> learned_model = hc.estimate(ArcOperatorSet(), OracleScore(), start_model)
    >>> assert set(learned_model.arcs()) == {('a', 'c'), ('b', 'c'), ('c', 'd')}

To learn dynamic Bayesian networks your class has to override
:class:`DynamicScore <pybnesian.DynamicScore>`. A new
:class:`DynamicScore <pybnesian.DynamicScore>` needs to implement the
following methods:

- :func:`DynamicScore.has_variables() <pybnesian.DynamicScore.has_variables>`.
- :func:`DynamicScore.static_score() <pybnesian.DynamicScore.static_score>`.
- :func:`DynamicScore.transition_score() <pybnesian.DynamicScore.transition_score>`.

Usually, your extended :class:`Score <pybnesian.Score>` will use data.
It is easy to implement a related :class:`DynamicScore <pybnesian.DynamicScore>` by
taking a :class:`DynamicDataFrame <pybnesian.DynamicDataFrame>` as parameter and using the methods
:func:`DynamicDataFrame.static_df() <pybnesian.DynamicDataFrame.static_df>` and
:func:`DynamicDataFrame.transition_df() <pybnesian.DynamicDataFrame.transition_df>` to implement
:func:`DynamicScore.static_score() <pybnesian.DynamicScore.static_score>`
and :func:`DynamicScore.transition_score() <pybnesian.DynamicScore.transition_score>`
respectively.


Learning Operators Extension
============================

Implementing a new learning score involves creating a class that inherits from
:class:`Operator <pybnesian.Operator>` (or
:class:`ArcOperator <pybnesian.ArcOperator>` for operators related with a single arc). Next, a new
:class:`OperatorSet <pybnesian.OperatorSet>` must be defined to use the new learning operator
within a learning algorithm.

An extended :class:`Operator <pybnesian.Operator>` class needs to implement the following methods:

- :func:`Operator.__eq__() <pybnesian.Operator.__eq__>`.  This method is optional. This method
  is needed if the :class:`OperatorTabuSet <pybnesian.OperatorTabuSet>` is used (in the
  :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>` it is used when the score is
  :class:`ValidatedScore <pybnesian.ValidatedScore>`).
- :func:`Operator.__hash__() <pybnesian.Operator.__hash__>`. This method is optional. This method
  is needed if the :class:`OperatorTabuSet <pybnesian.OperatorTabuSet>` is used (in the
  :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>` it is used when the score is
  :class:`ValidatedScore <pybnesian.ValidatedScore>`).
- :func:`Operator.__str__() <pybnesian.Operator.__str__>`.
- :func:`Operator.apply() <pybnesian.Operator.apply>`.
- :func:`Operator.nodes_changed() <pybnesian.Operator.nodes_changed>`.
- :func:`Operator.opposite() <pybnesian.Operator.opposite>`. This method is optional. This method
  is needed if the :class:`OperatorTabuSet <pybnesian.OperatorTabuSet>` is used (in the
  :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>` it is used when the score is
  :class:`ValidatedScore <pybnesian.ValidatedScore>`).

To illustrate, we will create a new :class:`AddArc <pybnesian.AddArc>` operator.

.. code-block::

    from pybnesian import Operator, RemoveArc

    class MyAddArc(Operator):

        def __init__(self, source, target, delta):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            Operator.__init__(self, delta)
            self.source = source
            self.target = target

        def __eq__(self, other):
            return self.source == other.source and self.target == other.target

        def __hash__(self):
            return hash((self.source, self.target))

        def __str__(self):
            return "MyAddArc(" + self.source + " -> " + self.target + ")"

        def apply(self, model):
            model.add_arc(self.source, self.target)

        def nodes_changed(self, model):
            return [self.target]

        def opposite():
            return RemoveArc(self.source, self.target, -self.delta())

To use this new operator, we need to define a :class:`OperatorSet <pybnesian.OperatorSet>` that
returns this type of operators. An extended :class:`OperatorSet <pybnesian.OperatorSet>` class needs
to implement the following methods:

- :func:`OperatorSet.cache_scores() <pybnesian.OperatorSet.cache_scores>`.
- :func:`OperatorSet.find_max() <pybnesian.OperatorSet.find_max>`.
- :func:`OperatorSet.find_max_tabu() <pybnesian.OperatorSet.find_max_tabu>`. This method is optional.
  This method is needed if the :class:`OperatorTabuSet <pybnesian.OperatorTabuSet>` is used (in the
  :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>` it is used when the score is
  :class:`ValidatedScore <pybnesian.ValidatedScore>`).
- :func:`OperatorSet.set_arc_blacklist() <pybnesian.OperatorSet.set_arc_blacklist>`. This method is
  optional. Implement it only if you need to check that an arc is blacklisted.
- :func:`OperatorSet.set_arc_whitelist() <pybnesian.OperatorSet.set_arc_whitelist>`. This method is
  optional. Implement it only if you need to check that an arc is whitelisted.
- :func:`OperatorSet.set_max_indegree() <pybnesian.OperatorSet.set_max_indegree>`. This method is
  optional. Implement it only if you need to check the maximum indegree of the graph.
- :func:`OperatorSet.set_type_blacklist() <pybnesian.OperatorSet.set_type_blacklist>`. This method is
  optional. Implement it only if you need to check that a node type is blacklisted.
- :func:`OperatorSet.set_type_whitelist() <pybnesian.OperatorSet.set_type_whitelist>`. This method is
  optional. Implement it only if you need to check that a node type is whitelisted.
- :func:`OperatorSet.update_scores() <pybnesian.OperatorSet.update_scores>`.
- :func:`OperatorSet.finished() <pybnesian.OperatorSet.finished>`. This method is optional. Implement
  it only if your class needs to clear the state.

To illustrate, we will create an operator set that only contains the :class:`MyAddArc <pybnesian.MyAddArc>` operators. Therefore, this
:class:`OperatorSet <pybnesian.OperatorSet>` can only add arcs.

.. code-block::

    from pybnesian import OperatorSet

    class MyAddArcSet(OperatorSet):

        def __init__(self):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            OperatorSet.__init__(self)
            self.blacklist = set()
            self.max_indegree = 0
            # Contains a dict {(source, target) : delta} of operators.
            self.set = {}

        # Auxiliary method
        def update_node(self, model, score, n):
            lc = self.local_score_cache()

            parents = model.parents(n)

            # Remove the parent operators, they will be added next.
            self.set = {p[0]: p[1] for p in self.set.items() if p[0][1] != n}

            blacklisted_parents = map(lambda op: op[0],
                                        filter(lambda bl : bl[1] == n, self.blacklist))
            # If max indegree == 0, there is no limit.
            if self.max_indegree == 0 or len(parents)  < self.max_indegree:
                possible_parents = set(model.nodes())\
                                    - set(n)\
                                    - set(parents)\
                                    - set(blacklisted_parents)

                for p in possible_parents:
                    if model.can_add_arc(p, n):
                        self.set[(p, n)] = score.local_score(model, n, parents + [p])\
                                           - lc.local_score(model, n)

        def cache_scores(self, model, score):
            for n in model.nodes():
                self.update_node(model, score, n)

        def find_max(self, model):
            sort_ops = sorted(self.set.items(), key=lambda op: op[1], reverse=True)

            for s in sort_ops:
                arc = s[0]
                delta = s[1]
                if model.can_add_arc(arc[0], arc[1]):
                    return MyAddArc(arc[0], arc[1], delta)
            return None

        def find_max_tabu(self, model, tabu):
            sort_ops = sorted(self.set.items(), key=lambda op: op[1], reverse=True)

            for s in sort_ops:
                arc = s[0]
                delta = s[1]
                op = MyAddArc(arc[0], arc[1], delta)
                # The operator can not be in the tabu set.
                if model.can_add_arc(arc[0], arc[1]) and not tabu.contains(op):
                    return op
            return None

        def update_scores(self, model, score, changed_nodes):
            for n in changed_nodes:
                self.update_node(model, score, n)

        def set_arc_blacklist(self, blacklist):
            self.blacklist = set(blacklist)

        def set_max_indegree(self, max_indegree):
            self.max_indegree = max_indegree

        def finished(self):
            self.blacklist.clear()
            self.max_indegree = 0
            self.set.clear()

This :class:`OperatorSet <pybnesian.OperatorSet>` can be used in a
:class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`:

.. testsetup::

    from pybnesian import Operator, RemoveArc, OperatorSet

    class MyAddArc(Operator):

        def __init__(self, source, target, delta):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            Operator.__init__(self, delta)
            self.source = source
            self.target = target

        def __eq__(self, other):
            return self.source == other.source and self.target == other.target

        def __hash__(self):
            return hash((self.source, self.target))

        def __str__(self):
            return "MyAddArc(" + self.source + " -> " + self.target + ")"

        def apply(self, model):
            model.add_arc(self.source, self.target)

        def nodes_changed(self, model):
            return [self.target]

        def opposite():
            return RemoveArc(self.source, self.target, -self.delta())

    class MyAddArcSet(OperatorSet):

        def __init__(self):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            OperatorSet.__init__(self)
            self.blacklist = set()
            self.max_indegree = 0
            self.set = {}

        # Auxiliary method
        def update_node(self, model, score, n):
            lc = self.local_score_cache()

            parents = model.parents(n)

            # Remove the parent operators, they will be added next.
            self.set = {p[0]: p[1] for p in self.set.items() if p[0][1] != n}

            blacklisted_parents = map(lambda op: op[0],
                                        filter(lambda bl : bl[1] == n, self.blacklist))
            # If max indegree == 0, there is no limit.
            if self.max_indegree == 0 or len(parents)  < self.max_indegree:
                possible_parents = set(model.nodes())\
                                    - set(n)\
                                    - set(parents)\
                                    - set(blacklisted_parents)

                for p in possible_parents:
                    if model.can_add_arc(p, n):
                        self.set[(p, n)] = score.local_score(model, n, parents + [p])\
                                           - lc.local_score(model, n)

        def cache_scores(self, model, score):
            for n in model.nodes():
                self.update_node(model, score, n)

        def find_max(self, model):
            sort_ops = sorted(self.set.items(), key=lambda op: op[1], reverse=True)

            for s in sort_ops:
                arc = s[0]
                delta = s[1]
                if model.can_add_arc(arc[0], arc[1]):
                    return MyAddArc(arc[0], arc[1], delta)
            return None

        def find_max_tabu(self, model, tabu):
            sort_ops = sorted(self.set.items(), key=lambda op: op[1], reverse=True)

            for s in sort_ops:
                arc = s[0]
                delta = s[1]
                op = MyAddArc(arc[0], arc[1], delta)
                # The operator can not be in the tabu set.
                if model.can_add_arc(arc[0], arc[1]) and not tabu.contains(op):
                    return op
            return None

        def update_scores(self, model, score, changed_nodes):
            for n in changed_nodes:
                self.update_node(model, score, n)

        def set_arc_blacklist(self, blacklist):
            self.blacklist = set(blacklist)

        def set_max_indegree(self, max_indegree):
            self.max_indegree = max_indegree

        def finished(self):
            self.blacklist.clear()
            self.max_indegree = 0
            self.set.clear()

.. doctest::

    >>> from pybnesian import GreedyHillClimbing
    >>> hc = GreedyHillClimbing()
    >>> add_set = MyAddArcSet()
    >>> # We will use the OracleScore: a -> c <- b, c -> d
    >>> score = OracleScore()
    >>> bn = GaussianNetwork(["a", "b", "c", "d"])
    >>> learned = hc.estimate(add_set, score, bn)
    >>> assert set(learned_model.arcs()) == {("a", "c"), ("b", "c"), ("c", "d")}
    >>> learned = hc.estimate(add_set, score, bn, arc_blacklist=[("b", "c")])
    >>> assert set(learned.arcs()) == {("a", "c"), ("c", "d")}
    >>> learned = hc.estimate(add_set, score, bn, max_indegree=1)
    >>> assert learned.num_arcs() == 2

Callbacks Extension
===================

The greedy hill-climbing algorithm admits a ``callback`` parameter that allows some custom functionality to be run on
each iteration. To create a callback, a new class must be created that inherits from
:class:`Callback <pybnesian.Callback>`. A new
:class:`Callback <pybnesian.Callback>` needs to implement the following method:

- :func:`Callback.call <pybnesian.Callback.call>`.

To illustrate, we will create a callback that prints the last operator applied on each iteration:

.. code-block::

    from pybnesian import Callback

    class PrintOperator(Callback):

        def __init__(self):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            Callback.__init__(self)

        def call(self, model, operator, score, iteration):
            if operator is None:
                if iteration == 0:
                    print("The algorithm starts!")
                else:
                    print("The algorithm ends!")
            else:
                print("Iteration " + str(iteration) + ". Last operator: " + str(operator))

Now, we can use this callback in the :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`:

.. testsetup::

    from pybnesian import Callback

    class PrintOperator(Callback):

        def __init__(self):
            # IMPORTANT: Always call the parent class to initialize the C++ object.
            Callback.__init__(self)

        def call(self, model, operator, score, iteration):
            if operator is None:
                if iteration == 0:
                    print("The algorithm starts!")
                else:
                    print("The algorithm ends!")
            else:
                print("Iteration " + str(iteration) + ". Last operator: " + str(operator))

.. doctest::

    >>> from pybnesian import GreedyHillClimbing
    >>> hc = GreedyHillClimbing()
    >>> add_set = MyAddArcSet()
    >>> # We will use the OracleScore: a -> c <- b, c -> d
    >>> score = OracleScore()
    >>> bn = GaussianNetwork(["a", "b", "c", "d"])
    >>> callback = PrintOperator()
    >>> learned = hc.estimate(add_set, score, bn, callback=callback)
    The algorithm starts!
    Iteration 1. Last operator: MyAddArc(c -> d)
    Iteration 2. Last operator: MyAddArc(b -> c)
    Iteration 3. Last operator: MyAddArc(a -> c)
    The algorithm ends!
