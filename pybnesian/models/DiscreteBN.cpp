#include <models/DiscreteBN.hpp>

namespace models {

std::shared_ptr<BayesianNetworkBase> DiscreteBNType::new_bn(const std::vector<std::string>& nodes) const {
    return std::make_shared<DiscreteBN>(nodes);
}
std::shared_ptr<ConditionalBayesianNetworkBase> DiscreteBNType::new_cbn(
    const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const {
    return std::make_shared<ConditionalDiscreteBN>(nodes, interface_nodes);
}

}  // namespace models