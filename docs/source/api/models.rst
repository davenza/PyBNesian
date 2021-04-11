Bayesian Networks
=================

.. automodule:: pybnesian.models

.. _models-abstract-classes:

Abstract Classes
^^^^^^^^^^^^^^^^
This classes are abstract and define the interface for Bayesian network objects. The :class:`BayesianNetworkType`
specifies the type of Bayesian networks. 

Each :class:`BayesianNetworkType` can be used in many multiple variants
of Bayesian networks: :class:`BayesianNetworkBase` (a normal Bayesian network), :class:`ConditionalBayesianNetworkBase`
(a conditional Bayesian network) and :class:`DynamicBayesianNetworkBase` (a dynamic Bayesian network).

.. autoclass:: pybnesian.models.BayesianNetworkType
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.BayesianNetworkBase
    :members:
    :special-members: __str__
    
.. autoclass:: pybnesian.models.ConditionalBayesianNetworkBase
    :show-inheritance:
    :members:
    :special-members: __str__

.. autoclass:: pybnesian.models.DynamicBayesianNetworkBase
    :members:
    :special-members: __str__

Bayesian Network Types
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pybnesian.models.GaussianNetworkType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.SemiparametricBNType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.KDENetworkType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.DiscreteBNType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.HomogeneousBNType
    :show-inheritance:
    :members:
    :special-members: __init__, __str_

.. autoclass:: pybnesian.models.HeterogeneousBNType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__


Bayesian Networks
^^^^^^^^^^^^^^^^^
.. autoclass:: pybnesian.models.BayesianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Concrete Bayesian Networks
**************************

These classes implements :class:`BayesianNetwork` with an specific :class:`BayesianNetworkType`. Thus, the constructors
do not have the ``type`` parameter.

.. autoclass:: pybnesian.models.GaussianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.SemiparametricBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.KDENetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.DiscreteBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.HomogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.HeterogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Conditional Bayesian Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pybnesian.models.ConditionalBayesianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Concrete Conditional Bayesian Networks
**************************************

These classes implements :class:`ConditionalBayesianNetwork` with an specific :class:`BayesianNetworkType`.
Thus, the constructors do not have the ``type`` parameter.

.. autoclass:: pybnesian.models.ConditionalGaussianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.ConditionalSemiparametricBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.ConditionalKDENetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.ConditionalDiscreteBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
.. autoclass:: pybnesian.models.ConditionalHomogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.ConditionalHeterogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Dynamic Bayesian Networks
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pybnesian.models.DynamicBayesianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Concrete Dynamic Bayesian Networks
**************************************

These classes implements :class:`DynamicBayesianNetwork` with an specific :class:`BayesianNetworkType`. Thus, the
constructors do not have the ``type`` parameter.

.. autoclass:: pybnesian.models.DynamicGaussianNetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.DynamicSemiparametricBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.DynamicKDENetwork
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.DynamicDiscreteBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.DynamicHomogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.models.DynamicHeterogeneousBN
    :show-inheritance:
    :members:
    :special-members: __init__, __str__