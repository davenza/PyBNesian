#ifndef PYBNESIAN_KDE_BANDWIDTHESTIMATOR_HPP
#define PYBNESIAN_KDE_BANDWIDTHESTIMATOR_HPP

#include <dataset/dataset.hpp>

using dataset::DataFrame;

namespace kde {

class BandwidthEstimator {
public:
    virtual ~BandwidthEstimator() {}
    virtual VectorXd estimate_diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const = 0;
    virtual MatrixXd estimate_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const = 0;

    virtual bool is_python_derived() const { return false; }

    static std::shared_ptr<BandwidthEstimator> keep_python_alive(std::shared_ptr<BandwidthEstimator>& b) {
        if (b && b->is_python_derived()) {
            auto o = py::cast(b);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<BandwidthEstimator*>();
            return std::shared_ptr<BandwidthEstimator>(keep_python_state_alive, ptr);
        }

        return b;
    }

    virtual py::tuple __getstate__() const = 0;
};

}  // namespace kde

#endif  // PYBNESIAN_KDE_BANDWIDTHESTIMATOR_HPP