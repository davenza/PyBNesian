#include <pybind11/pybind11.h>
#include <factors/factors.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>
#include <learning/parameters/mle_base.hpp>

namespace py = pybind11;

using factors::FactorType;
using factors::continuous::LinearGaussianCPD;
using learning::parameters::MLE;

namespace pybindings::learning::parameters {
    py::object mle_python_wrapper(FactorType f) {
        switch(f) {
            case FactorType::LinearGaussianCPD: {
                auto* mle = new MLE<LinearGaussianCPD>();
                auto pyobject = py::cast(mle);
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("MLE not available for FactorType " + f.ToString());
        }
    }
}
