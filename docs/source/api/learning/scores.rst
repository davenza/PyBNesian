Structure Scores
----------------

This section includes different learning scores that evaluate the goodness of a Bayesian network. This is used
for the score-and-search learning algorithms such as
:class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`, :class:`MMHC <pybnesian.MMHC>` and :class:`DMMHC <pybnesian.DMMHC>`.

Abstract classes
^^^^^^^^^^^^^^^^

.. autoclass:: pybnesian.Score
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.ValidatedScore
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
.. autoclass:: pybnesian.DynamicScore
    :members:
    :special-members: __init__, __str__

Concrete classes
^^^^^^^^^^^^^^^^
.. autoclass:: pybnesian.BIC
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.BGe
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.BDe
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.CVLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.HoldoutLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.ValidatedLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicBIC
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicBGe
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicBDe
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicCVLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicHoldoutLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.DynamicValidatedLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
