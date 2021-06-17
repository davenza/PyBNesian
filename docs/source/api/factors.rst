Factors module
**************

.. automodule:: pybnesian.factors

Abstract Types
==============

The :class:`FactorType` and :class:`Factor` classes are abstract and both of them need to be implemented to create a new
factor type. Each :class:`Factor` is always associated with a specific :class:`FactorType`.

.. autoclass:: pybnesian.factors.FactorType
    :members:
    :special-members: __init__, __str__
.. autoclass:: pybnesian.factors.Factor
    :members:
    :special-members: __init__, __str__

Continuous Factors
==================
The continuous factors are implemented in the submodule pybnesian.factors.continuous.

Linear Gaussian CPD
-------------------
.. autoclass:: pybnesian.factors.continuous.LinearGaussianCPDType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
.. autoclass:: pybnesian.factors.continuous.LinearGaussianCPD
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Conditional Kernel Density Estimation (CKDE)
--------------------------------------------

.. autoclass:: pybnesian.factors.continuous.CKDEType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
.. autoclass:: pybnesian.factors.continuous.CKDE
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
Discrete Factors
================
The discrete factors are implemented in the submodule pybnesian.factors.discrete.

.. autoclass:: pybnesian.factors.discrete.DiscreteFactorType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
.. autoclass:: pybnesian.factors.discrete.DiscreteFactor
    :show-inheritance:
    :members:
    :special-members: __init__, __str__


Other Types
===========
This types are not factors, but are auxiliary types for other factors.

.. autoclass:: pybnesian.factors.UnknownFactorType
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.factors.continuous.KDE
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.factors.Assignment
    :members:
    :special-members: __init__

Bibliography
============
.. [Scott] Scott, D. W. (2015). Multivariate Density Estimation: Theory, Practice and Visualization. 2nd Edition.
           Wiley