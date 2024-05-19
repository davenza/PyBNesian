# Changelog

## v0.5.0

- Changed the build process to statically link Apache Arrow. With this change and using the
[PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html), PyBNesian can interoperate
with different versions of `pyarrow>=14.0.0`. You can now upgrade pyarrow (`pip install --upgrade pyarrow`)
without breaking PyBNesian. The dependencies are also managed by [vcpkg](https://vcpkg.io), so the
build process is simpler and orchestrated by scikit-build-core and a CMakeLists.txt.

- Some tests failed because `pandas` and `scipy` were updated. These issues have been fixed.

- A bug in the `DiscreteFactor.sample()` function has been fixed. The previous implementation sampled equally from the first and last category of the `DiscreteFactor`.

## v0.4.3

- Fixed a bug in `DiscreteFactor` and others hybrid factors, such as `CLinearGaussianCPD` and `HCKDE`, where categorical data would not be correctly validated. This could lead to erroneous results or undefined behavior (often leading to segmentation fault). Thanks to Carlos Li for reporting this bug.

- Support for Python 3.10 and `pyarrow>=9.0` has been added. Support for Python 3.6 has been deprecated, as `pyarrow` no longer supports it. 

- manylinux2014 wheels are now used instead of manylinux2010, since `pyarrow` no longer provides manylinux2010 wheels.

## v0.4.2

- Fixed important bug in OpenCL for NVIDIA GPUs, as they define small OpenCL constant memory. See [https://stackoverflow.com/questions/63080816/opencl-small-constant-memory-size-on-nvidia-gpu](https://stackoverflow.com/questions/63080816/opencl-small-constant-memory-size-on-nvidia-gpu).

## v0.4.1

- Added support for Apache Arrow 7.0.0.

## v0.4.0

- Added method `ConditionalBayesianNetworkBase.interface_arcs()`.
- `GreedyHillClimbing` and `MMHC` now accepts a blacklist of `FactorType`.
- `BayesianNetworkType.data_default_node_type()` now returns a list of `FactorType` indicating the priority of each `FactorType` for each data type.
- `BayesianNetworkBase.set_unknown_node_types()` now accepts an argument of `FactorType` blacklist.
- Change `HeterogeneousBN` constructor and `HeterogeneousBNType.default_node_types()` to accept lists of default
  `FactorType`.
- Adds constructors for `HeterogeneousBN` and `CLGNetwork` that can set the `FactorType` for each node.

- Bug Fixes:

  - An overflow error in `ChiSquare` hypothesis test was raised when the statistic were close to 0.
  - Arc blacklists/whitelists with repeated arcs were not correctly processed.
  - Fixed an error in the use of the patience parameter. Previously, the algorithm was executed as with a `patience - 1` value.
  - Improve the validation of objects returned from Python class extensions, so it errors when the extensions are not correctly implemented.
  - Fixed many serialization bugs. In particular, there were multiple bugs related with the serialization of models with Python extensions.
  - Included a fix for the Windows build (by setting a correct `__cplusplus` value).
  - Fixed a bug in `LinearGaussianCPD.fit()` with 2 parents. In some cases, it was detecting a linear dependence between the parents that did not exist.
  - Fixes a bug which causes that the Python-class extension functionality is removed. 
    Related to: [https://github.com/pybind/pybind11/issues/1333](https://github.com/pybind/pybind11/issues/1333).

## v0.3.4

- Improvements on the code that checks that a matrix is positive definite.
- A bug affecting the learning of conditional Bayesian networks with `MMHC` has been fixed. This bug also affected `DMMHC`.
- Fixed a bug that affected the type of the parameter `bn_type` of `MMHC.estimate()`, `MMHC.estimate_conditional()` and `DMMHC.estimate()`.

## v0.3.3

- Adds support for pyarrow 5.0.0 in the PyPi wheels.
- Added `Arguments.args()` to access the `args` and `kwargs` for a node.
- Added `BayesianNetworkBase.underlying_node_type()` to get the underlying node type of a node given some data.
- Improves the fitting of hybrid factors. Now, an specific discrete configuration can be left unfitted if the base continuous factor raises `SingularCovarianceData`.
- Improves the `LinearGaussianCPD` fit when the covariance matrix of the data is singular.
- Improves the `NormalReferenceRule`, `ScottsBandwidth`, and `UCV` estimation when the covariance of the data is singular.
- Fixes a bug loading an heterogeneous Bayesian network from a file.
- Introduces a check that a needed category exists in discrete data.
- `Assignment` now supports integer numbers converting them automatically to float.
- Fix a bug in `GreedyHillClimbing` that caused the return of Bayesian networks with `UnknownFactorType`.
- Reduces memory usage when fitting and printing an hybrid `Factor`.
- Fixes a precision bug in `GreedyHillClimbing`.
- Improves `CrossValidation` parameter checking.

## v0.3.2

- Fixed a bug in the `UCV` bandwidth selector that may cause segmentation fault.
- Added some checks to ensure that the categorical data is of type string.
- Fixed the `GreedyHillClimbing` iteration counter, which was begin increased twice per iteration.
- Added a default parameter value for `include_cpd` in `BayesianNetworkBase:save()` and
  `DynamicBayesianNetworkBase::save()`.
- Added more checks to detect ill-conditioned regression problems. The `BIC` score returns `-infinity` for
  ill-conditioned regression problems.

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