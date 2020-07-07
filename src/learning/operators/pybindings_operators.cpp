#include <learning/operators/operators.hpp>
#include <learning/operators/pybindings_operators.hpp>

using learning::operators::AddArc, learning::operators::RemoveArc, 
      learning::operators::FlipArc, learning::operators::ChangeNodeType, 
      learning::operators::OperatorTabuSet;

namespace learning::operators {

    py::object addarc_wrapper_constructor(BayesianNetworkType bn, std::string source, std::string target, double delta) {
        switch(bn) {
            case BayesianNetworkType::GBN: {
                auto* op = new AddArc<GaussianNetwork<>>(source, target, delta);
                py::object pyobject = py::cast(op);
                return std::move(pyobject);
            }
            case BayesianNetworkType::SPBN: {
                auto* op = new AddArc<SemiparametricBN<>>(source, target, delta);
                py::object pyobject = py::cast(op);
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("AddArc not available for BayesianNetwork " + bn.ToString());
        }
    }

    py::object removearc_wrapper_constructor(BayesianNetworkType bn, std::string source, std::string target, double delta) {
        switch(bn) {
            case BayesianNetworkType::GBN: {
                auto* op = new RemoveArc<GaussianNetwork<>>(source, target, delta);
                py::object pyobject = py::cast(op);
                return std::move(pyobject);
            }
            case BayesianNetworkType::SPBN: {
                auto* op = new RemoveArc<SemiparametricBN<>>(source, target, delta);
                py::object pyobject = py::cast(op);
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("RemoveArc not available for BayesianNetwork " + bn.ToString());
        }
    }

    py::object fliparc_wrapper_constructor(BayesianNetworkType bn, std::string source, std::string target, double delta) {
        switch(bn) {
            case BayesianNetworkType::GBN: {
                auto* op = new FlipArc<GaussianNetwork<>>(source, target, delta);
                py::object pyobject = py::cast(op);
                return std::move(pyobject);
            }
            case BayesianNetworkType::SPBN: {
                auto* op = new FlipArc<SemiparametricBN<>>(source, target, delta);
                py::object pyobject = py::cast(op);
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("FlipArc not available for BayesianNetwork " + bn.ToString());
        }
    }


    py::object changenodetype_wrapper_constructor(BayesianNetworkType bn, std::string node, FactorType new_factor, double delta) {
        switch(bn) {
            case BayesianNetworkType::SPBN: {
                auto* op = new ChangeNodeType<SemiparametricBN<>>(node, new_factor, delta);
                py::object pyobject = py::cast(op);
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("ChangeNodeType not available for BayesianNetwork " + bn.ToString());
        }
    }

    py::object operatortabuset_wrapper_constructor(BayesianNetworkType bn) {
        switch(bn) {
            case BayesianNetworkType::GBN: {
                auto* tabu = new OperatorTabuSet<GaussianNetwork<>>();
                py::object pyobject = py::cast(tabu, py::return_value_policy::move);
                return std::move(pyobject);
            }
            case BayesianNetworkType::SPBN: {
                auto* tabu = new OperatorTabuSet<SemiparametricBN<>>();
                py::object pyobject = py::cast(tabu, py::return_value_policy::move);
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("OperatorTabuSet not available for BayesianNetwork " + bn.ToString());
        }
    }
}