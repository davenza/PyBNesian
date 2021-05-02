#include <pybind11/pybind11.h>
#include <factors/factors.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/discrete/DiscreteCPD.hpp>
#include <learning/parameters/mle_base.hpp>

namespace py = pybind11;

using factors::FactorType, factors::continuous::LinearGaussianCPD, factors::continuous::LinearGaussianCPDType,
    factors::discrete::DiscreteFactorType, factors::discrete::DiscreteCPD;
using learning::parameters::MLE;

namespace pybindings::learning::parameters {

py::object mle_python_wrapper(std::shared_ptr<FactorType>& f) {
    auto& ft = *f;
    if (ft == LinearGaussianCPDType::get_ref()) {
        return py::cast(new MLE<LinearGaussianCPD>());
    } else if (ft == DiscreteFactorType::get_ref()) {
        return py::cast(new MLE<DiscreteCPD>());
    } else {
        throw std::invalid_argument("MLE not available for NodeType " + ft.ToString());
    }
}

}  // namespace pybindings::learning::parameters
