Factors module
**************

The factors are usually represented as conditional probability functions and are a component of a Bayesian network.

Abstract Types
==============

The :class:`FactorType <pybnesian.FactorType>` and :class:`Factor <pybnesian.Factor>` classes are abstract and both of them need to be implemented to create a new
factor type. Each :class:`Factor <pybnesian.Factor>` is always associated with a specific :class:`FactorType <pybnesian.FactorType>`.

.. autoclass:: pybnesian.FactorType
    :members:
    :special-members: __init__, __str__
.. autoclass:: pybnesian.Factor
    :members:
    :special-members: __init__, __str__

Continuous Factors
==================

Linear Gaussian CPD
-------------------
.. autoclass:: pybnesian.LinearGaussianCPDType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
.. autoclass:: pybnesian.LinearGaussianCPD
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Conditional Kernel Density Estimation (CKDE)
--------------------------------------------

.. autoclass:: pybnesian.CKDEType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
.. autoclass:: pybnesian.CKDE
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
Discrete Factors
================

.. autoclass:: pybnesian.DiscreteFactorType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
.. autoclass:: pybnesian.DiscreteFactor
    :show-inheritance:
    :members:
    :special-members: __init__, __str__


Hybrid Factors
==============

.. autoclass:: pybnesian.CLinearGaussianCPD
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
.. autoclass:: pybnesian.HCKDE
    :show-inheritance:
    :members:
    :special-members: __init__, __str__



Other Types
===========
This types are not factors, but are auxiliary types for other factors.

Kernel Density Estimation
-------------------------

.. autoclass:: pybnesian.BandwidthSelector
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.ScottsBandwidth
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.NormalReferenceRule
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.UCV
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.KDE
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.ProductKDE
    :members:
    :special-members: __init__

.. autoexception:: pybnesian.SingularCovarianceData
    :show-inheritance:

    This exception signals that the data has a singular covariance matrix.

Other
-----

.. autoclass:: pybnesian.UnknownFactorType
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.Assignment
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.Args
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.Kwargs
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.Arguments
    :members:
    :special-members: __init__

Bibliography
============
.. [Scott] Scott, D. W. (2015). Multivariate Density Estimation: Theory, Practice and Visualization. 2nd Edition.
           Wiley
.. [MVKSA] José E. Chacón and Tarn Duong. (2018). Multivariate Kernel Smoothing and Its Applications. CRC Press.
.. [Semiparametric] David Atienza and Concha Bielza and Pedro Larrañaga. Semiparametric Bayesian networks. Information
                    Sciences, vol. 584, pp. 564-582, 2022.
.. [HybridSemiparametric] David Atienza and Pedro Larrañaga and Concha Bielza. Hybrid semiparametric Bayesian networks.
                          TEST, vol. 31, pp. 299-327, 2022.