#include <pybind11/stl.h>
#include <dataset/crossvalidation_adaptator.hpp>
#include <dataset/holdout_adaptator.hpp>
#include <dataset/dynamic_dataset.hpp>
#include <util/util_types.hpp>

using dataset::DataFrame, dataset::CrossValidation, dataset::HoldOut, dataset::DynamicDataFrame,
    dataset::DynamicVariable;

using util::random_seed_arg;

void pybindings_dataset(py::module& root) {
    py::class_<CrossValidation> cv(root, "CrossValidation", R"doc(
This class implements k-fold cross-validation, i.e. it splits the data into k disjoint sets of train and test data.
)doc");

    cv.def(py::init([](DataFrame df, int k, std::optional<unsigned int> seed, bool include_null) {
               return CrossValidation(df, k, random_seed_arg(seed), include_null);
           }),
           py::arg("df"),
           py::arg("k") = 10,
           py::arg("seed") = std::nullopt,
           py::arg("include_null") = false,
           R"doc(
This constructor takes a :class:`DataFrame` and returns a k-fold cross-validation. It shuffles the data before applying
the cross-validation.

:param df: A :class:`DataFrame`.
:param k: Number of folds.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
:param include_null: Whether to include the rows where some columns may be null (missing). If false, the rows with some
                     missing values are filtered before performing the cross-validation. Else, all the rows are
                     included.
:raises ValueError: If k is greater than the number of rows.
)doc")
        .def(
            "__iter__",
            [](CrossValidation& self) { return py::make_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>(),
            R"doc(
Iterates over the k-fold cross-validation.

:returns: The iterator returns a tuple (:class:`DataFrame`, :class:`DataFrame`) which contains the training data and
          test data of each fold.

.. testsetup::
    
    import numpy as np
    import pandas as pd

.. doctest::

    >>> from pybnesian import CrossValidation
    >>> df = pd.DataFrame({'a': np.random.rand(20), 'b': np.random.rand(20)})
    >>> for (training_data, test_data) in CrossValidation(df):
    ...     assert training_data.num_rows == 18
    ...     assert test_data.num_rows == 2
)doc")
        .def("fold", &CrossValidation::fold, py::return_value_policy::take_ownership, py::arg("index"), R"doc(

Returns the index-th fold.

:param index: Fold index.
:returns: A tuple (:class:`DataFrame`, :class:`DataFrame`) which contains the training data and test data of each fold.
)doc")
        .def(
            "indices",
            [](CrossValidation& self) { return py::make_iterator(self.begin_indices(), self.end_indices()); },
            py::keep_alive<0, 1>(),
            R"doc(
Iterates over the row indices of each training and test :class:`DataFrame`.

:returns: A tuple (list, list) containing the row indices (with respect to the original :class:`DataFrame`) of the train
          and test data of each fold.

.. testsetup::
    
    import numpy as np
    import pandas as pd

.. doctest::

    >>> from pybnesian import CrossValidation
    >>> df = pd.DataFrame({'a': np.random.rand(20), 'b': np.random.rand(20)})
    >>> for (training_indices, test_indices) in CrossValidation(df).indices():
    ...     assert set(range(20)) == set(list(training_indices) + list(test_indices))
)doc");

    {
        // Show the overloads using this workaround:
        // https://github.com/pybind/pybind11/issues/2619
        py::options options;
        options.disable_function_signatures();

        cv.def(
              "loc", [](CrossValidation& self, std::string name) { return self.loc(name); }, py::arg("column"))
            .def(
                "loc", [](CrossValidation& self, int idx) { return self.loc(idx); }, py::arg("column"))
            .def(
                "loc",
                [](CrossValidation& self, std::vector<std::string> v) { return self.loc(v); },
                py::arg("columns"))
            .def(
                "loc", [](CrossValidation& self, std::vector<int> v) { return self.loc(v); }, py::arg("columns"), R"doc(

loc(self: pybnesian.CrossValidation, columns: str or int or List[str] or List[int]) -> CrossValidation

Selects columns from the :class:`CrossValidation` object.

:param columns: Columns to select. The columns can be represented by their index (int or List[int]) or by their name
                (str or List[str]).
:returns: A :class:`CrossValidation` object with the selected columns.
)doc");
    }

    py::class_<HoldOut>(root, "HoldOut", R"doc(
This class implements holdout validation, i.e. it splits the data into training and test sets.
)doc")
        .def(py::init([](const DataFrame& df, double test_ratio, std::optional<unsigned int> seed, bool include_null) {
                 return HoldOut(df, test_ratio, random_seed_arg(seed), include_null);
             }),
             py::arg("df"),
             py::arg("test_ratio") = 0.2,
             py::arg("seed") = std::nullopt,
             py::arg("include_null") = false,
             R"doc(
This constructor takes a :class:`DataFrame` and returns a split into training an test sets. It shuffles the data before
applying the holdout.

:param df: A :class:`DataFrame`.
:param test_ratio: Proportion of instances left for the test data.
:param seed: A random seed number. If not specified or ``None``, a random seed is generated.
:param include_null: Whether to include the rows where some columns may be null (missing). If false, the rows with some
                     missing values are filtered before performing the cross-validation. Else, all the rows are
                     included.
)doc")
        .def("training_data", &HoldOut::training_data, py::return_value_policy::reference_internal, R"doc(
Gets the training data.

:returns: Training data.
)doc")
        .def("test_data", &HoldOut::test_data, py::return_value_policy::reference_internal, R"doc(
Gets the test data.

:returns: Test data.
)doc");

    py::class_<DynamicVariable<int>>(root, "DynamicVariable<int>")
        .def(py::init<int, int>())
        .def(py::init<std::pair<int, int>>())
        .def_property(
            "variable",
            [](DynamicVariable<int>& self) { return self.variable; },
            [](DynamicVariable<int>& self, int other) { self.variable = other; })
        .def_property(
            "temporal_slice",
            [](DynamicVariable<int>& self) { return self.temporal_slice; },
            [](DynamicVariable<int>& self, int slice) { self.temporal_slice = slice; });

    py::class_<DynamicVariable<std::string>>(root, "DynamicVariable<std::string>", R"doc(
This class implements a DynamicVariable.
)doc")
        .def(py::init<std::string, int>())
        .def(py::init<std::pair<std::string, int>>())
        .def_property(
            "variable",
            [](DynamicVariable<std::string>& self) { return self.variable; },
            [](DynamicVariable<std::string>& self, std::string other) { self.variable = other; })
        .def_property(
            "temporal_slice",
            [](DynamicVariable<std::string>& self) { return self.temporal_slice; },
            [](DynamicVariable<std::string>& self, int slice) { self.temporal_slice = slice; });

    py::implicitly_convertible<std::pair<int, int>, DynamicVariable<int>>();
    py::implicitly_convertible<std::pair<std::string, int>, DynamicVariable<std::string>>();

    py::class_<DynamicDataFrame> ddf(root, "DynamicDataFrame", R"doc(
This class implements the adaptation of a :class:`DynamicDataFrame` to a dynamic context (temporal series). This
is useful to make easier to learn dynamic Bayesian networks.

A :class:`DynamicDataFrame` creates columns with different temporal delays from the data in the static
:class:`DataFrame`. Each column in the :class:`DynamicDataFrame` is named with the following pattern:
``[variable_name]_t_[temporal_index]``. The ``variable_name`` is the name of each column in the static
:class:`DataFrame`. The ``temporal_index`` is an index with a range [0-``markovian_order``]. The index "0" is considered
the "present", the index "1" delays the temporal one step into the "past", and so on...

:class:`DynamicDataFrame` contains two functions :func:`DynamicDataFrame.static_df` and
:func:`DynamicDataFrame.transition_df` that can be used to learn the static Bayesian network and transition Bayesian
network components of a dynamic Bayesian network.

All the operations are implemented using a zero-copy strategy to avoid wasting memory.

.. testsetup::

    import numpy as np
    import pandas as pd

.. _DynamicDataFrame example:

.. doctest::

    >>> from pybnesian import DynamicDataFrame
    >>> df = pd.DataFrame({'a': np.arange(10, dtype=float)})
    >>> ddf = DynamicDataFrame(df, 2)
    >>> ddf.transition_df().to_pandas()
       a_t_0  a_t_1  a_t_2
    0    2.0    1.0    0.0
    1    3.0    2.0    1.0
    2    4.0    3.0    2.0
    3    5.0    4.0    3.0
    4    6.0    5.0    4.0
    5    7.0    6.0    5.0
    6    8.0    7.0    6.0
    7    9.0    8.0    7.0
    >>> ddf.static_df().to_pandas()
       a_t_1  a_t_2
    0    1.0    0.0
    1    2.0    1.0
    2    3.0    2.0
    3    4.0    3.0
    4    5.0    4.0
    5    6.0    5.0
    6    7.0    6.0
    7    8.0    7.0
    8    9.0    8.0
)doc");

    ddf.def(py::init<const DataFrame&, int>(), py::arg("df"), py::arg("markovian_order"), R"doc(
Creates a :class:`DynamicDataFrame` from an static :class:`DataFrame` using a given markovian order.

:param df: A :class:`DataFrame`.
:param markovian_order: Markovian order of the transformation.
)doc")
        .def("markovian_order", &DynamicDataFrame::markovian_order, R"doc(
Gets the markovian order.

:returns: Markovian order of the :class:`DynamicDataFrame`.
)doc")
        .def("num_columns", &DynamicDataFrame::num_columns, R"doc(
Gets the number of columns.

:returns: The number of columns. This is equal to the number of columns of :func:`DynamicDataFrame.transition_df`.
)doc")
        .def("num_variables", &DynamicDataFrame::num_columns, R"doc(
Gets the number of variables.

:returns: The number of variables. This is exactly equal to the number of columns in :func:`DynamicDataFrame.origin_df`.
)doc")
        .def("num_rows", &DynamicDataFrame::num_rows, R"doc(
Gets the number of row.

:returns: Number of rows.
)doc")
        .def("origin_df", &DynamicDataFrame::origin_df, py::return_value_policy::reference_internal, R"doc(
Gets the original :class:`DataFrame`.

:returns: The :class:`DataFrame` passed to the constructor of :class:`DynamicDataFrame`.
)doc")
        .def("static_df", &DynamicDataFrame::static_df, py::return_value_policy::reference_internal, R"doc(
Gets the :class:`DataFrame` for the static Bayesian network. The static network estimates the probability
f(``t_1``,..., ``t_[markovian_order]``). See `DynamicDataFrame example`_.

:returns: A :class:`DataFrame` with columns from ``[variable_name]_t_1`` to ``[variable_name]_t_[markovian_order]``
)doc")
        .def("transition_df", &DynamicDataFrame::transition_df, py::return_value_policy::reference_internal, R"doc(
Gets the :class:`DataFrame` for the transition Bayesian network. The transition network estimates the conditional
probability f(``t_0`` | ``t_1``, ..., ``t_[markovian_order]``). See `DynamicDataFrame example`_.

:returns: A :class:`DataFrame` with columns from ``[variable_name]_t_0`` to ``[variable_name]_t_[markovian_order]``
)doc");

    {
        py::options options;
        options.disable_function_signatures();
        ddf.def(
               "temporal_slice",
               [](const DynamicDataFrame& self, int slice_index) { return self.temporal_slice(slice_index); },
               py::arg("index"))
            .def(
                "temporal_slice",
                [](const DynamicDataFrame& self, const std::vector<int>& slice_indices) {
                    return self.temporal_slice(slice_indices.begin(), slice_indices.end());
                },
                py::arg("indices"),
                R"doc(
temporal_slice(self: pybnesian.DynamicDataFrame, indices: int or List[int]) -> DataFrame

Gets a temporal slice or a set of temporal slices. The i-th temporal slice is composed by the columns
``[variable_name]_t_i``

:returns: A :class:`DataFrame` with the selected temporal slices.

.. testsetup::

    import numpy as np
    import pandas as pd

.. doctest::

    >>> from pybnesian import DynamicDataFrame
    >>> df = pd.DataFrame({'a': np.arange(10, dtype=float), 'b': np.arange(0, 100, 10, dtype=float)})
    >>> ddf = DynamicDataFrame(df, 2)
    >>> ddf.temporal_slice(1).to_pandas()
       a_t_1  b_t_1
    0    1.0   10.0
    1    2.0   20.0
    2    3.0   30.0
    3    4.0   40.0
    4    5.0   50.0
    5    6.0   60.0
    6    7.0   70.0
    7    8.0   80.0
    >>> ddf.temporal_slice([0, 2]).to_pandas()
       a_t_0  b_t_0  a_t_2  b_t_2
    0    2.0   20.0    0.0    0.0
    1    3.0   30.0    1.0   10.0
    2    4.0   40.0    2.0   20.0
    3    5.0   50.0    3.0   30.0
    4    6.0   60.0    4.0   40.0
    5    7.0   70.0    5.0   50.0
    6    8.0   80.0    6.0   60.0
    7    9.0   90.0    7.0   70.0
)doc")
            .def(
                "loc",
                [](const DynamicDataFrame& self, const DynamicVariable<int>& v) { return self.loc(v); },
                py::arg("column"))
            .def(
                "loc",
                [](const DynamicDataFrame& self, const DynamicVariable<std::string>& v) { return self.loc(v); },
                py::arg("column"))
            .def(
                "loc",
                [](const DynamicDataFrame& self, const std::vector<DynamicVariable<int>>& vec) {
                    return self.loc(vec);
                },
                py::arg("columns"))
            .def(
                "loc",
                [](const DynamicDataFrame& self, const std::vector<DynamicVariable<std::string>>& vec) {
                    return self.loc(vec);
                },
                py::arg("columns"),
                R"doc(
loc(self: pybnesian.DynamicDataFrame, columns: DynamicVariable or List[DynamicVariable]) -> DataFrame

Gets a column or set of columns from the :class:`DynamicDataFrame`. See :class:`DynamicVariable`.

:returns: A :class:`DataFrame` with the selected columns.

.. testsetup::

    import numpy as np
    import pandas as pd

.. doctest::

    >>> from pybnesian import DynamicDataFrame
    >>> df = pd.DataFrame({'a': np.arange(10, dtype=float),
    ...                    'b': np.arange(0, 100, 10, dtype=float)})
    >>> ddf = DynamicDataFrame(df, 2)
    >>> ddf.loc(("b", 1)).to_pandas()
       b_t_1
    0   10.0
    1   20.0
    2   30.0
    3   40.0
    4   50.0
    5   60.0
    6   70.0
    7   80.0
    >>> ddf.loc([("a", 0), ("b", 1)]).to_pandas()
       a_t_0  b_t_1
    0    2.0   10.0
    1    3.0   20.0
    2    4.0   30.0
    3    5.0   40.0
    4    6.0   50.0
    5    7.0   60.0
    6    8.0   70.0
    7    9.0   80.0

**All the DynamicVariables in the list must be of the same type**, so do not mix different types:

.. code-block:: python

    >>> ddf.loc([(0, 0), ("b", 1)]) # do NOT do this!

    # Either you use names or indices:
    >>> ddf.loc([("a", 0), ("b", 1)]) # GOOD
    >>> ddf.loc([(0, 1), (1, 1)]) # GOOD

)doc");
    }
}
