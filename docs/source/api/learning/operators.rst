Learning Operators
------------------

This section includes learning operators that are used to make small, local changes to a given Bayesian network
structure. This is used for the score-and-search learning algorithms such as
:class:`GreedyHillClimbing <pybnesian.learning.algorithms.GreedyHillClimbing>`,
:class:`MMHC <pybnesian.learning.algorithms.MMHC>` and :class:`DMMHC <pybnesian.learning.algorithms.DMMHC>`.

There are two type of classes in this section: operators and operator sets:

- The operators are the representation of a change in a Bayesian network structure.
- The operator sets coordinate sets of operators. They can find the best operator over the set and update the score and
  availability of each operator in the set.

Operators
^^^^^^^^^

.. autoclass:: pybnesian.learning.operators.Operator
    :members:
    :special-members: __init__, __str__, __eq__, __hash__

.. autoclass:: pybnesian.learning.operators.ArcOperator
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.operators.AddArc
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.operators.RemoveArc
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.operators.FlipArc
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.operators.ChangeNodeType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
Operator Sets
^^^^^^^^^^^^^
    
.. autoclass:: pybnesian.learning.operators.OperatorSet
    :members:
    :special-members: __init__, __str__
    
.. autoclass:: pybnesian.learning.operators.ArcOperatorSet
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
.. autoclass:: pybnesian.learning.operators.ChangeNodeTypeSet
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
.. autoclass:: pybnesian.learning.operators.OperatorPool
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Other
^^^^^
.. autoclass:: pybnesian.learning.operators.OperatorTabuSet
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.learning.operators.LocalScoreCache
    :members:
    :special-members: __init__, __str__