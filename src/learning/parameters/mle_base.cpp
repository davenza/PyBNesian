#include <learning/parameters/mle_base.hpp>

namespace learning::parameters {
    py::object mle_python_wrapper(NodeType n) {
        switch(n) {
            case NodeType::LinearGaussianCPD: {
                auto mle = std::make_unique<MLE<LinearGaussianCPD>>();
                auto pyobject = py::cast(mle.get());
                return std::move(pyobject);
            }
            default:
                throw std::invalid_argument("MLE not available for NodeType " + n.ToString());
        }
    }
}
