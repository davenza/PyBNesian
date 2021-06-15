*********
Changelog
*********

v0.2.0
======

- Added conditional linear Gaussian networks (:class:`CLGNetworkType <pybnesian.models.CLGNetworkType>`, 
  :class:`CLGNetwork <pybnesian.models.CLGNetwork>`,
  :class:`ConditionalCLGNetwork <pybnesian.models.ConditionalCLGNetwork>` and
  :class:`DynamicCLGNetwork <pybnesian.models.DynamicCLGNetwork>`).
- Implemented :class:`ChiSquare <pybnesian.learning.independences.ChiSquare>` (and 
  :class:`DynamicChiSquare <pybnesian.learning.independences.DynamicChiSquare>`) indepencence test.
- Implemented :class:`MutualInformation <pybnesian.learning.independences.MutualInformation>` (and
  :class:`DynamicMutualInformation <pybnesian.learning.independences.DynamicMutualInformation>`) indepencence test. This
  independence test is valid for hybrid data.
- Implemented :class:`BDe <pybnesian.learning.scores.BDe>` (Bayesian Dirichlet equivalent) score (and
  :class:`DynamicBDe <pybnesian.learning.scores.DynamicBDe>`).
- Added :class:`UnknownFactorType <pybnesian.factors.UnknownFactorType>` as default
  :class:`FactorType <pybnesian.factors.FactorType>` for Bayesian networks when the node type could not be deduced.

API changes:

- Added method :func:`Score.data() <pybnesian.learning.scores.Score.data>`.
- Added
  :func:`BayesianNetworkType.data_default_node_type() <pybnesian.models.BayesianNetworkType.data_default_node_type>` for
  non-homogeneous :class:`BayesianNetworkType <pybnesian.models.BayesianNetworkType>`.
- Added constructor for :class:`HeterogeneousBN <pybnesian.models.HeterogeneousBN>` to specify a default
  :class:`FactorType <pybnesian.factors.FactorType>` for each data type. Also, it adds
  :func:`HeterogeneousBNType.default_node_types() <pybnesian.models.HeterogeneousBNType.default_node_types>` and
  :func:`HeterogeneousBNType.single_default() <pybnesian.models.HeterogeneousBNType.single_default>`.
- Added
  :func:`BayesianNetworkBase.has_unknown_node_types() <pybnesian.models.BayesianNetworkBase.has_unknown_node_types>` and
  :func:`BayesianNetworkBase.set_unknown_node_types() <pybnesian.models.BayesianNetworkBase.set_unknown_node_types>`.
- Changed signature of
  :func:`BayesianNetworkType.compatible_node_type() <pybnesian.models.BayesianNetworkType.compatible_node_type>` to
  include the new node type as argument.
- Removed :func:`FactorType.opposite_semiparametric()`. This functionality has been replaced by
  :func:`BayesianNetworkType.alternative_node_type() <pybnesian.models.BayesianNetworkType.alternative_node_type>`.
- Included model as argument of :func:`Operator.opposite() <pybnesian.learning.operators.Operator.opposite>`.
- Added method :func:`OperatorSet.set_type_blacklist() <pybnesian.learning.operators.OperatorSet.set_type_blacklist>`.
  Added a type blacklist argument to :class:`ChangeNodeTypeSet <pybnesian.learning.operators.ChangeNodeTypeSet>`
  constructor.

v0.1.0
======

- First release! =).