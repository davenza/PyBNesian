Data Manipulation
*****************

.. automodule:: pybnesian.dataset

DataFrame Operations
====================

.. autoclass:: pybnesian.dataset.CrossValidation
    :members:
    :special-members: __init__, __iter__
.. autoclass:: pybnesian.dataset.HoldOut
    :members:
    :special-members: __init__, __iter__

Dynamic Data
============

.. autoclass:: pybnesian.dataset.DynamicDataFrame
    :members:
    :special-members: __init__, __iter__

.. class:: DynamicVariable
    
    A DynamicVariable is the representation of a column in a :class:`DynamicDataFrame`.

    A DynamicVariable is a tuple (``variable_index``, ``temporal_index``). ``variable_index`` is a ``str`` or
    ``int`` that represents the name or index of the variable in the original static :class:`DataFrame`.
    ``temporal_index`` is an ``int`` that represents the temporal slice in the :class:`DynamicDataFrame`.
    See :func:`DynamicDataFrame.loc` for usage examples.