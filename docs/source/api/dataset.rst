Data Manipulation
*****************

PyBNesian implements some useful dataset manipulation techniques such as k-fold cross validation and hold-out.

DataFrame
^^^^^^^^^

Internally, PyBNesian uses a :class:`pyarrow.RecordBatch <pyarrow.RecordBatch>` to enable a zero-copy data exchange between C++ and Python.

Most of the classes and methods takes as argument, or returns a :class:`DataFrame <pybnesian.DataFrame>` type. This represents an
encapsulation of :class:`pyarrow.RecordBatch <pyarrow.RecordBatch>`:

- When a :class:`DataFrame <pybnesian.DataFrame>` is taken as argument in a function, both a :class:`pyarrow.RecordBatch <pyarrow.RecordBatch>` or a
  :class:`pandas.DataFrame <pandas.DataFrame>` can be used as a parameter.

- When PyBNesian specifies a :class:`DataFrame <pybnesian.DataFrame>` return  type, a :class:`pyarrow.RecordBatch <pyarrow.RecordBatch>` is returned. 
  This can be converted easily to a :class:`pandas.DataFrame <pandas.DataFrame>` using :meth:`pyarrow.RecordBatch.to_pandas`.

DataFrame Operations
====================

.. autoclass:: pybnesian.CrossValidation
    :members:
    :special-members: __init__, __iter__
.. autoclass:: pybnesian.HoldOut
    :members:
    :special-members: __init__, __iter__

Dynamic Data
============

.. autoclass:: pybnesian.DynamicDataFrame
    :members:
    :special-members: __init__, __iter__

.. class:: DynamicVariable
    
    A DynamicVariable is the representation of a column in a :class:`DynamicDataFrame <pybnesian.DynamicDataFrame>`.

    A DynamicVariable is a tuple (``variable_index``, ``temporal_index``). ``variable_index`` is a ``str`` or
    ``int`` that represents the name or index of the variable in the original static :class:`DataFrame <pybnesian.DataFrame>`.
    ``temporal_index`` is an ``int`` that represents the temporal slice in the :class:`DynamicDataFrame <pybnesian.DynamicDataFrame>`.
    See :func:`DynamicDataFrame.loc <pybnesian.DynamicDataFrame.loc>` for usage examples.