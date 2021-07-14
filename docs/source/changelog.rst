*********
Changelog
*********

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