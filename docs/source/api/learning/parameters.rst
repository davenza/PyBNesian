Parameter Learning
==================

.. automodule:: pybnesian.learning.parameters

Currently, it only implements Maximum Likelihood Estimation (MLE) for
:class:`LinearGaussianCPD <pybnesian.factors.continuous.LinearGaussianCPD>` and
:class:`DiscreteFactor <pybnesian.factors.discrete.DiscreteFactor>`.

.. autofunction:: pybnesian.learning.parameters.MLE

.. autoclass:: pybnesian.learning.parameters.LinearGaussianParams
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.learning.parameters.MLELinearGaussianCPD
    :members:

.. autoclass:: pybnesian.learning.parameters.DiscreteFactorParams
    :members:
    :special-members: __init__