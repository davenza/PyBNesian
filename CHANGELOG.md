# Changelog

## v0.2.0

- Added conditional linear Gaussian networks (`CLGNetworkType`, `CLGNetwork`, `ConditionalCLGNetwork` and `DynamicCLGNetwork`).
- Implemented `ChiSquare` (and `DynamicChiSquare`) indepencence test.
- Implemented `MutualInformation` (and `DynamicMutualInformation`) indepencence test. This is valid for hybrid data.
- Implemented `BDe` (Bayesian Dirichlet equivalent) score (and `DynamicBDe`).
- Added `UnknownFactorType` as default `FactorType` for Bayesian networks when the node type could not be deduced.
- Added `Assignment` class to represent the assignment of values to variables.

API changes:

- Added method `Score::data()`.
- Added `BayesianNetworkType::data_default_node_type()` for non-homogeneous `BayesianNetworkType`.
- Added constructor for `HeterogeneousBN` to specify a default `FactorType` for each data type. Also, it adds
    `HeterogeneousBN::default_node_types()` and `HeterogeneousBN::single_default()`.
- Added `BayesianNetworkBase::has_unknown_node_types()` and `BayesianNetworkBase::set_unknown_node_types()`.
- Changed signature of `BayesianNetworkType::compatible_node_type()` to include the new node type as argument.
- Removed `FactorType::opposite_semiparametric()`. This functionality has been replaced by
    `BayesianNetworkType::alternative_node_type()`.
- Included model as parameter of `Operator::opposite()`.
- Added method `OperatorSet::set_type_blacklist()`. Added a type blacklist argument to `ChangeNodeTypeSet` constructor.

## v0.1.0

- First release! =).