#include <learning/parameters/pybindings_mle.hpp>
#include <learning/parameters/mle_base.hpp>
#include <factors/continuous/LinearGaussianCPD.hpp>

using factors::continuous::LinearGaussianCPD;

namespace learning::parameters {
    py::object mle_python_wrapper(FactorType f) {
        switch(f) {
            case FactorType::LinearGaussianCPD: {
                auto mle = std::make_unique<MLE<LinearGaussianCPD>>();
                auto pyobject = py::cast(mle.get());
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("MLE not available for FactorType " + f.ToString());
        }
    }
}
