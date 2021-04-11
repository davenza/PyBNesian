Independence Tests
------------------

This section includes conditional tests of independence. These tests are used in many constraint-based learning
algorithms such as :class:`PC <pybnesian.learning.algorithms.PC>`, :class:`MMPC <pybnesian.learning.algorithms.MMPC>`,
:class:`MMHC <pybnesian.learning.algorithms.MMHC>` and :class:`DMMHC <pybnesian.learning.algorithms.DMMHC>`.

Abstract classes
^^^^^^^^^^^^^^^^

.. autoclass:: pybnesian.learning.independences.IndependenceTest
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.independences.DynamicIndependenceTest
    :members:
    :special-members: __str__

Concrete classes
^^^^^^^^^^^^^^^^

.. autoclass:: pybnesian.learning.independences.LinearCorrelation
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.independences.KMutualInformation
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.independences.RCoT
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.independences.DynamicLinearCorrelation
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.independences.DynamicKMutualInformation
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.independences.DynamicRCoT
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Bibliography
^^^^^^^^^^^^

.. [CMIknn] Runge, J. (2018). Conditional independence testing based on a nearest-neighbor estimator of conditional
            mutual information. International Conference on Artificial Intelligence and Statistics, AISTATS 2018, 84,
            938–947.

.. [RCoT] Strobl, E. V., Zhang, K., & Visweswaran, S. (2019). Approximate kernel-based conditional independence tests
          for fast non-parametric causal discovery. Journal of Causal Inference, 7(1).