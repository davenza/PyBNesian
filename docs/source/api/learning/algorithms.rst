Learning Algorithms
-------------------

.. autofunction:: pybnesian.hc

This classes implement many different learning structure algorithms.

.. autoclass:: pybnesian.GreedyHillClimbing
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.PC
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.MMPC
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.MMHC
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.DMMHC
    :members:
    :special-members: __init__

Learning Algorithms Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pybnesian.MeekRules
    :members:

Learning Callbacks
******************

.. autoclass:: pybnesian.Callback
    :members:
    :special-members: __init__

.. autoclass:: pybnesian.SaveModel
    :show-inheritance:
    :members:
    :special-members: __init__

Bibliography
^^^^^^^^^^^^
.. [pc-stable] Colombo, D., & Maathuis, M. H. (2014). Order-independent constraint-based causal structure learning.
               Journal of Machine Learning Research, 15, 3921–3962.
.. [mmhc] Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The max-min hill-climbing Bayesian network structure
          learning algorithm. Machine Learning, 65(1), 31–78.
.. [dmmhc] Trabelsi, G., Leray, P., Ben Ayed, M., & Alimi, A. M. (2013). Dynamic MMHC: A local search algorithm for
           dynamic Bayesian network structure learning. Advances in Intelligent Data Analysis XII, 8207 LNCS, 392–403.
.. [meek] Meek, C. (1995). Causal Inference and Causal Explanation with Background Knowledge. In Eleventh Conference on
          Uncertainty in Artificial Intelligence (UAI'95), 403–410.