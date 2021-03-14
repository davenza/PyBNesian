#include <pybind11/pybind11.h>
#include <factors/factors.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <learning/parameters/mle_base.hpp>

namespace py = pybind11;

using factors::FactorType, factors::continuous::LinearGaussianCPD, factors::continuous::LinearGaussianCPDType,
    factors::discrete::DiscreteFactorType, factors::discrete::DiscreteFactor;
using learning::parameters::MLE;

namespace pybindings::learning::parameters {

py::object mle_python_wrapper(std::shared_ptr<FactorType>& f) {
    auto& ft = *f;
    if (ft == LinearGaussianCPDType::get_ref()) {
        // auto mle = std::make_shared<MLE<LinearGaussianCPD>>();
        return py::cast(new MLE<LinearGaussianCPD>());
    } else if (ft == DiscreteFactorType::get_ref()) {
        // auto mle = std::make_shared<MLE<DiscreteFactor>>();
        return py::cast(new MLE<DiscreteFactor>());
    } else {
        throw std::invalid_argument("MLE not available for NodeType " + ft.ToString());
    }
}

}  // namespace pybindings::learning::parameters
