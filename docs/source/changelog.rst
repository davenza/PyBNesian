*********
Changelog
*********

v0.5.1
======

- Fixes vcpkg bad hashes for boost-core (`vcpkg/#38974 <https://github.com/microsoft/vcpkg/issues/38974>`_).
- Updates arrow to 17.0.0.

v0.5.0
======

- Changed the build process to statically link Apache Arrow. With this change and using the `PyCapsule interface <https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_, PyBNesian can interoperate with different versions of ``pyarrow>=14.0.0``. You can now upgrade pyarrow (``pip install --upgrade pyarrow``) without breaking PyBNesian. The dependencies are also managed by `vcpkg <https://vcpkg.io>`_, so the build process is simpler and orchestrated by scikit-build-core and a CMakeLists.txt.

- Some tests failed because ``pandas`` and ``scipy`` were updated. These issues have been fixed.

- A bug in the :func:`DiscreteFactor.sample <pybnesian.DiscreteFactor.sample>` function has been fixed. The previous implementation sampled equally from the first and last category of the :class:`DiscreteFactor <pybnesian.DiscreteFactor>`.


v0.4.3
======

- Fixed a bug in :class:`DiscreteFactor <pybnesian.DiscreteFactor>` and others hybrid factors, such as
  :class:`CLinearGaussianCPD <pybnesian.CLinearGaussianCPD>` and :class:`HCKDE <pybnesian.HCKDE>`, where categorical
  data would not be correctly validated. This could lead to erroneous results or undefined behavior (often leading to
  segmentation fault). Thanks to Carlos Li for reporting this bug.

- Support for Python 3.10 and ``pyarrow>=9.0`` has been added. Support for Python 3.6 has been deprecated, as
  ``pyarrow`` no longer supports it. 

- manylinux2014 wheels are now used instead of manylinux2010, since ``pyarrow`` no longer provides manylinux2010 wheels.

v0.4.2
======

- Fixed important bug in OpenCL for NVIDIA GPUs, as they define small OpenCL constant memory. See
  https://stackoverflow.com/questions/63080816/opencl-small-constant-memory-size-on-nvidia-gpu.


v0.4.1
======

- Added support for Apache Arrow 7.0.0.

v0.4.0
======

- Added method
  :func:`ConditionalBayesianNetworkBase.interface_arcs <pybnesian.ConditionalBayesianNetworkBase.interface_arcs>`.
- :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>` and :class:`MMHC <pybnesian.MMHC>` now accepts a blacklist
  of :class:`FactorType <pybnesian.FactorType>`.
- :func:`BayesianNetworkType.data_default_node_type <pybnesian.BayesianNetworkType.data_default_node_type>` now returns
  a list of :class:`FactorType <pybnesian.FactorType>` indicating the priority of each
  :class:`FactorType <pybnesian.FactorType>` for each data type.
- :func:`BayesianNetworkBase.set_unknown_node_types <pybnesian.BayesianNetworkBase.set_unknown_node_types>` now accepts
  an argument of :class:`FactorType <pybnesian.FactorType>` blacklist.
- Change :class:`HeterogeneousBN <pybnesian.HeterogeneousBN>` constructor and
  :func:`HeterogeneousBNType.default_node_types <pybnesian.HeterogeneousBNType.default_node_types>` to accept lists of
  default :class:`FactorType <pybnesian.FactorType>`.
- Adds constructors for :class:`HeterogeneousBN <pybnesian.HeterogeneousBN>` and
  :class:`CLGNetwork <pybnesian.CLGNetwork>` that can set the :class:`FactorType <pybnesian.FactorType>` for each node.

- Bug Fixes:

  - An overflow error in :class:`ChiSquare <pybnesian.ChiSquare>` hypothesis test was raised when the statistic were
    close to 0.
  - Arc blacklists/whitelists with repeated arcs were not correctly processed.
  - Fixed an error in the use of the patience parameter. Previously, the algorithm was executed as with a
    ``patience - 1`` value.
  - Improve the validation of objects returned from Python class extensions, so it errors when the extensions are not
    correctly implemented.
  - Fixed many serialization bugs. In particular, there were multiple bugs related with the serialization of models with
    Python extensions.
  - Included a fix for the Windows build (by setting a correct ``__cplusplus`` value).
  - Fixed a bug in :func:`LinearGaussianCPD.fit <pybnesian.Factor.fit>` with 2 parents. In some cases, it was
    detecting a linear dependence between the parents that did not exist.
  - Fixes a bug which causes that the Python-class extension functionality is removed.
    Related to: https://github.com/pybind/pybind11/issues/1333.


v0.3.4
======

- Improvements on the code that checks that a matrix positive definite.
- A bug affecting the learning of conditional Bayesian networks with :class:`MMHC <pybnesian.MMHC>` has been fixed. This
  bug also affected :class:`DMMHC <pybnesian.DMMHC>`.
- Fixed a bug that affected the type of the parameter ``bn_type`` of :func:`MMHC.estimate <pybnesian.MMHC.estimate>`,
  :func:`MMHC.estimate_conditional <pybnesian.MMHC.estimate_conditional>` and
  :func:`DMMHC.estimate <pybnesian.DMMHC.estimate>`.

v0.3.3
======

- Adds support for pyarrow 5.0.0 in the PyPi wheels.
- Added :func:`Arguments.args <pybnesian.Arguments.args>` to access the ``args`` and ``kwargs`` for a node.
- Added :func:`BayesianNetworkBase.underlying_node_type <pybnesian.BayesianNetworkBase.underlying_node_type>` to get the
  underlying node type of a node given some data.
- Improves the fitting of hybrid factors. Now, an specific discrete configuration can be left unfitted if the base
  continuous factor raises :class:`SingularCovarianceData <pybnesian.SingularCovarianceData>`.
- Improves the :class:`LinearGaussianCPD <pybnesian.LinearGaussianCPD>` fit when the covariance matrix of the data is
  singular.
- Improves the :class:`NormalReferenceRule <pybnesian.NormalReferenceRule>`,
  :class:`ScottsBandwidth <pybnesian.ScottsBandwidth>`, and :class:`UCV <pybnesian.UCV>` estimation when the covariance
  of the data is singular.
- Fixes a bug loading an heterogeneous Bayesian network from a file.
- Introduces a check that a needed category exists in discrete data.
- :class:`Assignment <pybnesian.Assignment>` now supports integer numbers converting them automatically to float.
- Fix a bug in :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>` that caused the return of Bayesian networks
  with :class:`UnknownFactorType <pybnesian.UnknownFactorType>`.
- Reduces memory usage when fitting and printing an hybrid :class:`Factor <pybnesian.Factor>`.
- Fixes a precision bug in :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>`.
- Improves :class:`CrossValidation <pybnesian.CrossValidation>` parameter checking.

v0.3.2
======

- Fixed a bug in the :class:`UCV <pybnesian.UCV>` bandwidth selector that may cause segmentation fault.
- Added some checks to ensure that the categorical data is of type string.
- Fixed the :class:`GreedyHillClimbing <pybnesian.GreedyHillClimbing>` iteration counter, which was begin increased
  twice per iteration.
- Added a default parameter value for ``include_cpd`` in
  :func:`BayesianNetworkBase.save <pybnesian.BayesianNetworkBase.save>` and
  :func:`DynamicBayesianNetworkBase.save <pybnesian.DynamicBayesianNetworkBase.save>`.
- Added more checks to detect ill-conditioned regression problems. The :class:`BIC <pybnesian.BIC>` score returns
  ``-infinity`` for ill-conditioned regression problems.

v0.3.1
======

- Fixed the build process to support CMake versions older than 3.13.
- Fixed a bug that might raise an error with a call to :func:`FactorType.new_factor <pybnesian.FactorType.new_factor>`
  with `*args` and `**kwargs` arguments . This bug was only reproducible if the library was compiled with gcc.
- Added CMake as prerequisite to compile the library in the docs.

v0.3.0
======

- Removed all the submodules to simplify the imports. Now, all the classes are accessible directly from the pybnesian
  root module.
- Added a :class:`ProductKDE <pybnesian.ProductKDE>` class that implements :class:`KDE <pybnesian.KDE>` with diagonal
  bandwidth matrix.
- Added an abstract class :class:`BandwidthSelector <pybnesian.BandwidthSelector>` to implement bandwidth selection for
  :class:`KDE <pybnesian.KDE>` and :class:`ProductKDE <pybnesian.ProductKDE>`. Three concrete implementations of
  bandwidth selection are included: :class:`ScottsBandwidth <pybnesian.ScottsBandwidth>`,
  :class:`NormalReferenceRule <pybnesian.NormalReferenceRule>` and :class:`UCV <pybnesian.UCV>`.
- Added :class:`Arguments <pybnesian.Arguments>`, :class:`Args <pybnesian.Args>` and :class:`Kwargs <pybnesian.Kwargs>`
  to store a set of arguments to be used to create new factors through
  :func:`FactorType.new_factor <pybnesian.FactorType.new_factor>`. The :class:`Arguments <pybnesian.Arguments>` are
  accepted by :func:`BayesianNetworkBase.fit <pybnesian.BayesianNetworkBase.fit>` and the constructors of
  :class:`CVLikelihood <pybnesian.CVLikelihood>`, :class:`HoldoutLikelihood <pybnesian.HoldoutLikelihood>` and
  :class:`ValidatedLikelihood <pybnesian.ValidatedLikelihood>`.

v0.2.1
======
- An error related to the processing of categorical data with too many categories has been corrected.
- Removed ``-march=native`` flag in the build script to avoid the use of instruction sets not available on some CPUs.

v0.2.0
======

- Added conditional linear Gaussian networks (:class:`CLGNetworkType <pybnesian.CLGNetworkType>`, 
  :class:`CLGNetwork <pybnesian.CLGNetwork>`,
  :class:`ConditionalCLGNetwork <pybnesian.ConditionalCLGNetwork>` and
  :class:`DynamicCLGNetwork <pybnesian.DynamicCLGNetwork>`).
- Implemented :class:`ChiSquare <pybnesian.ChiSquare>` (and 
  :class:`DynamicChiSquare <pybnesian.DynamicChiSquare>`) indepencence test.
- Implemented :class:`MutualInformation <pybnesian.MutualInformation>` (and
  :class:`DynamicMutualInformation <pybnesian.DynamicMutualInformation>`) indepencence test. This
  independence test is valid for hybrid data.
- Implemented :class:`BDe <pybnesian.BDe>` (Bayesian Dirichlet equivalent) score (and
  :class:`DynamicBDe <pybnesian.DynamicBDe>`).
- Added :class:`UnknownFactorType <pybnesian.UnknownFactorType>` as default
  :class:`FactorType <pybnesian.FactorType>` for Bayesian networks when the node type could not be deduced.
- Added :class:`Assignment <pybnesian.Assignment>` class to represent the assignment of values to variables.

API changes:

- Added method :func:`Score.data() <pybnesian.Score.data>`.
- Added
  :func:`BayesianNetworkType.data_default_node_type() <pybnesian.BayesianNetworkType.data_default_node_type>` for
  non-homogeneous :class:`BayesianNetworkType <pybnesian.BayesianNetworkType>`.
- Added constructor for :class:`HeterogeneousBN <pybnesian.HeterogeneousBN>` to specify a default
  :class:`FactorType <pybnesian.FactorType>` for each data type. Also, it adds
  :func:`HeterogeneousBNType.default_node_types() <pybnesian.HeterogeneousBNType.default_node_types>` and
  :func:`HeterogeneousBNType.single_default() <pybnesian.HeterogeneousBNType.single_default>`.
- Added
  :func:`BayesianNetworkBase.has_unknown_node_types() <pybnesian.BayesianNetworkBase.has_unknown_node_types>` and
  :func:`BayesianNetworkBase.set_unknown_node_types() <pybnesian.BayesianNetworkBase.set_unknown_node_types>`.
- Changed signature of
  :func:`BayesianNetworkType.compatible_node_type() <pybnesian.BayesianNetworkType.compatible_node_type>` to
  include the new node type as argument.
- Removed :func:`FactorType.opposite_semiparametric()`. This functionality has been replaced by
  :func:`BayesianNetworkType.alternative_node_type() <pybnesian.BayesianNetworkType.alternative_node_type>`.
- Included model as argument of :func:`Operator.opposite() <pybnesian.Operator.opposite>`.
- Added method :func:`OperatorSet.set_type_blacklist() <pybnesian.OperatorSet.set_type_blacklist>`.
  Added a type blacklist argument to :class:`ChangeNodeTypeSet <pybnesian.ChangeNodeTypeSet>`
  constructor.

v0.1.0
======

- First release! =).