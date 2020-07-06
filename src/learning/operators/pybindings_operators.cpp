#include <learning/operators/operators.hpp>
#include <learning/operators/pybindings_operators.hpp>

using learning::operators::AddArc, learning::operators::RemoveArc, 
      learning::operators::FlipArc, learning::operators::ChangeNodeType;

namespace learning::operators {

    py::object addarc_wrapper_constructor(BayesianNetworkType bn, std::string source, std::string target, double delta) {
        switch(bn) {
            case BayesianNetworkType::GBN: {
                auto op = std::make_unique<AddArc<GaussianNetwork<>>>(source, target, delta);
                py::object pyobject = py::cast(op.get());
                return std::move(pyobject);
            }
            case BayesianNetworkType::SPBN: {
                auto op = std::make_unique<AddArc<SemiparametricBN<>>>(source, target, delta);
                py::object pyobject = py::cast(op.get());
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("AddArc not available for BayesianNetwork " + bn.ToString());
        }
    }

    py::object removearc_wrapper_constructor(BayesianNetworkType bn, std::string source, std::string target, double delta) {
        switch(bn) {
            case BayesianNetworkType::GBN: {
                auto op = std::make_unique<RemoveArc<GaussianNetwork<>>>(source, target, delta);
                py::object pyobject = py::cast(op.get());
                return std::move(pyobject);
            }
            case BayesianNetworkType::SPBN: {
                auto op = std::make_unique<RemoveArc<SemiparametricBN<>>>(source, target, delta);
                py::object pyobject = py::cast(op.get());
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("RemoveArc not available for BayesianNetwork " + bn.ToString());
        }
    }

    py::object fliparc_wrapper_constructor(BayesianNetworkType bn, std::string source, std::string target, double delta) {
        switch(bn) {
            case BayesianNetworkType::GBN: {
                auto op = std::make_unique<FlipArc<GaussianNetwork<>>>(source, target, delta);
                py::object pyobject = py::cast(op.get());
                return std::move(pyobject);
            }
            case BayesianNetworkType::SPBN: {
                auto op = std::make_unique<FlipArc<SemiparametricBN<>>>(source, target, delta);
                py::object pyobject = py::cast(op.get());
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("FlipArc not available for BayesianNetwork " + bn.ToString());
        }
    }


    py::object changenodetype_wrapper_constructor(BayesianNetworkType bn, std::string node, FactorType new_factor, double delta) {
        switch(bn) {
            case BayesianNetworkType::SPBN: {
                auto op = std::make_unique<ChangeNodeType<SemiparametricBN<>>>(node, new_factor, delta);
                py::object pyobject = py::cast(op.get());
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("AddArc not available for BayesianNetwork " + bn.ToString());
        }
    }
}