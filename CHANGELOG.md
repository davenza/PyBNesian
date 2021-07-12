# Changelog

## v0.3.1

- Fixed the build process to support CMake versions older than 3.13.
- Fixed a bug that might raise an error with a call to `FactorType::new_factor()`
  with `*args` and `**kwargs` arguments . This bug was only reproducible if the library was compiled with gcc.
- Added CMake as prerequisite to compile the library in the docs.

## v0.3.0

- Removed all the submodules to simplify the imports. Now, all the classes are accessible directly from the pybnesian
  root module.
- Added a `ProductKDE` class that implements `KDE` with diagonal bandwidth matrix.
- Added an abstract class `BandwidthSelector` to implement bandwidth selection for `KDE` and `ProductKDE`. Three
  concrete implementations of bandwidth selection are included: `ScottsBandwidth`, `NormalReferenceRule` and `UCV`.
- Added `Arguments`, `Args` and `Kwargs` to store a set of arguments to be used to create new factors through
  `FactorType::new_factor()`. The `Arguments` are accepted by `BayesianNetworkBase::fit()` and the constructors of
  `CVLikelihood`, `HoldoutLikelihood` and `ValidatedLikelihood`.

## v0.2.1

- An error related to the processing of categorical data with too many categories has been corrected.
- Removed `-march=native` flag in the build script to avoid the use of instruction sets not available on some CPUs.

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