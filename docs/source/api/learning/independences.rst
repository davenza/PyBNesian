Independence Tests
------------------

This section includes conditional tests of independence. These tests are used in many constraint-based learning
algorithms such as :class:`PC <pybnesian.PC>`, :class:`MMPC <pybnesian.MMPC>`, :class:`MMHC <pybnesian.MMHC>` and :class:`DMMHC <pybnesian.DMMHC>`.

Abstract classes
^^^^^^^^^^^^^^^^

.. autoclass:: pybnesian.IndependenceTest
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicIndependenceTest
    :members:
    :special-members: __str__

Concrete classes
^^^^^^^^^^^^^^^^

.. autoclass:: pybnesian.LinearCorrelation
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.MutualInformation
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.KMutualInformation
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.RCoT
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.ChiSquare
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicLinearCorrelation
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicMutualInformation
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicKMutualInformation
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicRCoT
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicChiSquare
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

.. [MSCMI] [1] Mesner, O. C. and Shalizi C. R. (2021) Conditional mutual information estimation for mixed, discrete and 
            continuous data. IEEE Transactions on Information Theory, 67(1), 464–484.

.. [MixedCMIKnn] [1] Popescu, O.-I., Gerhardus, A. & Runge, J. (2023). Non-parametric conditional independence testing for 
            mixed continuous-categorical variables: A novel method and numerical evaluation. arXiv pre-print. 
            Available: https://arxiv.org/abs/2310.11132