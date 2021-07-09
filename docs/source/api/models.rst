Bayesian Networks
=================

PyBNesian includes many different types of Bayesian networks.

.. _models-abstract-classes:

Abstract Classes
^^^^^^^^^^^^^^^^
This classes are abstract and define the interface for Bayesian network objects. The :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>`
specifies the type of Bayesian networks. 

Each :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>` can be used in many multiple variants
of Bayesian networks: :class:`BayesianNetworkBase <pybnesian.BayesianNetworkBase>` (a normal Bayesian network), :class:`ConditionalBayesianNetworkBase <pybnesian.ConditionalBayesianNetworkBase>`
(a conditional Bayesian network) and :class:`DynamicBayesianNetworkBase <pybnesian.DynamicBayesianNetworkBase>` (a dynamic Bayesian network).

.. autoclass:: pybnesian.BayesianNetworkType
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.BayesianNetworkBase
    :members:
    :special-members: __str__
    
.. autoclass:: pybnesian.ConditionalBayesianNetworkBase
    :show-inheritance:
    :members:
    :special-members: __str__

.. autoclass:: pybnesian.DynamicBayesianNetworkBase
    :members:
    :special-members: __str__

Bayesian Network Types
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pybnesian.GaussianNetworkType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.SemiparametricBNType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.KDENetworkType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DiscreteBNType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.HomogeneousBNType
    :show-inheritance:
    :members:
    :special-members: __init__, __str_

.. autoclass:: pybnesian.HeterogeneousBNType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.CLGNetworkType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__


Bayesian Networks
^^^^^^^^^^^^^^^^^
.. autoclass:: pybnesian.BayesianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Concrete Bayesian Networks
**************************

These classes implements :class:`BayesianNetwork <pybnesian.BayesianNetwork>` with an specific :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>`. Thus, the constructors
do not have the ``type`` parameter.

.. autoclass:: pybnesian.GaussianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.SemiparametricBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.KDENetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DiscreteBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.HomogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.HeterogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.CLGNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Conditional Bayesian Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pybnesian.ConditionalBayesianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Concrete Conditional Bayesian Networks
**************************************

These classes implements :class:`ConditionalBayesianNetwork <pybnesian.ConditionalBayesianNetwork>` with an specific :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>`.
Thus, the constructors do not have the ``type`` parameter.

.. autoclass:: pybnesian.ConditionalGaussianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.ConditionalSemiparametricBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.ConditionalKDENetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.ConditionalDiscreteBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
.. autoclass:: pybnesian.ConditionalHomogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.ConditionalHeterogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.ConditionalCLGNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Dynamic Bayesian Networks
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pybnesian.DynamicBayesianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Concrete Dynamic Bayesian Networks
**************************************

These classes implements :class:`DynamicBayesianNetwork <pybnesian.DynamicBayesianNetwork>` with an specific :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>`. Thus, the
constructors do not have the ``type`` parameter.

.. autoclass:: pybnesian.DynamicGaussianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicSemiparametricBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicKDENetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicDiscreteBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicHomogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicHeterogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicCLGNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__