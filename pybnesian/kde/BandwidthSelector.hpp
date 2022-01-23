#ifndef PYBNESIAN_KDE_BANDWIDTHSELECTOR_HPP
#define PYBNESIAN_KDE_BANDWIDTHSELECTOR_HPP

#include <dataset/dataset.hpp>

using dataset::DataFrame;

namespace kde {

class BandwidthSelector {
public:
    virtual ~BandwidthSelector() {}
    virtual VectorXd diag_bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const = 0;
    virtual MatrixXd bandwidth(const DataFrame& df, const std::vector<std::string>& variables) const = 0;

    virtual bool is_python_derived() const { return false; }

    static std::shared_ptr<BandwidthSelector>& keep_python_alive(std::shared_ptr<BandwidthSelector>& b) {
        if (b && b->is_python_derived()) {
            auto o = py::cast(b);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<BandwidthSelector*>();
            b = std::shared_ptr<BandwidthSelector>(keep_python_state_alive, ptr);
        }

        return b;
    }

    static std::shared_ptr<BandwidthSelector> keep_python_alive(const std::shared_ptr<BandwidthSelector>& b) {
        if (b && b->is_python_derived()) {
            auto o = py::cast(b);
            auto keep_python_state_alive = std::make_shared<py::object>(o);
            auto ptr = o.cast<BandwidthSelector*>();
            return std::shared_ptr<BandwidthSelector>(keep_python_state_alive, ptr);
        }

        return b;
    }

    virtual std::string ToString() const = 0;

    virtual py::tuple __getstate__() const = 0;
};

}  // namespace kde

#endif  // PYBNESIAN_KDE_BANDWIDTHSELECTOR_HPP