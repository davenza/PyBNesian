Structure Scores
----------------

This section includes different learning scores that evaluate the goodness of a Bayesian network. This is used
for the score-and-search learning algorithms such as
:class:`GreedyHillClimbing <pybnesian.learning.algorithms.GreedyHillClimbing>`,
:class:`MMHC <pybnesian.learning.algorithms.MMHC>` and :class:`DMMHC <pybnesian.learning.algorithms.DMMHC>`.

Abstract classes
^^^^^^^^^^^^^^^^

.. autoclass:: pybnesian.learning.scores.Score
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.scores.ValidatedScore
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
.. autoclass:: pybnesian.learning.scores.DynamicScore
    :members:
    :special-members: __init__, __str__

Concrete classes
^^^^^^^^^^^^^^^^
.. autoclass:: pybnesian.learning.scores.BIC
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.scores.BGe
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.scores.CVLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.scores.HoldoutLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.scores.ValidatedLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.scores.DynamicBIC
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.scores.DynamicBGe
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.scores.DynamicCVLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.scores.DynamicHoldoutLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.scores.DynamicValidatedLikelihood
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
