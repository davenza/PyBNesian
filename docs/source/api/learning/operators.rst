Learning Operators
------------------

This section includes learning operators that are used to make small, local changes to a given Bayesian network
structure. This is used for the score-and-search learning algorithms such as
:class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`, :class:`MMHC <pybnesian.MMHC>` and :class:`DMMHC <pybnesian.DMMHC>`.

There are two type of classes in this section: operators and operator sets:

- The operators are the representation of a change in a Bayesian network structure.
- The operator sets coordinate sets of operators. They can find the best operator over the set and update the score and
  availability of each operator in the set.

Operators
^^^^^^^^^

.. autoclass:: pybnesian.Operator
    :members:
    :special-members: __init__, __str__, __eq__, __hash__

.. autoclass:: pybnesian.ArcOperator
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.AddArc
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.RemoveArc
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.FlipArc
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.ChangeNodeType
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
Operator Sets
^^^^^^^^^^^^^
    
.. autoclass:: pybnesian.OperatorSet
    :members:
    :special-members: __init__, __str__
    
.. autoclass:: pybnesian.ArcOperatorSet
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
.. autoclass:: pybnesian.ChangeNodeTypeSet
    :show-inheritance:
    :members:
    :special-members: __init__, __str__
    
.. autoclass:: pybnesian.OperatorPool
    :show-inheritance:
    :members:
    :special-members: __init__, __str__

Other
^^^^^
.. autoclass:: pybnesian.OperatorTabuSet
    :members:
    :special-members: __init__, __str__

.. autoclass:: pybnesian.LocalScoreCache
    :members:
    :special-members: __init__, __str__