Parameter Learning
==================

PyBNesian implements learning parameter learning for :class:`Factor <pybnesian.Factor>` from data.

Currently, it only implements Maximum Likelihood Estimation (MLE) for
:class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` and
:class:`DiscreteFactor <pybnesian.DiscreteFactor>`.

.. autofunction:: pybnesian.MLE

.. autoclass:: pybnesian.LinearGaussianParams
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.MLELinearGaussianCPD
    :members:

.. autoclass:: pybnesian.DiscreteFactorParams
    :members:
    :special-members: __init__