#include <learning/operators/operators.hpp>

using learning::operators::AddArc;

namespace learning::operators {

    std::shared_ptr<Operator> AddArc::opposite() {
        return std::make_shared<RemoveArc>(this->source(), this->target(), -this->delta());
    }

}