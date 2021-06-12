#include <models/CLGNetwork.hpp>

namespace models {

std::shared_ptr<BayesianNetworkBase> CLGNetworkType::new_bn(const std::vector<std::string>& nodes) const {
    return std::make_shared<CLGNetwork>(nodes);
}

std::shared_ptr<ConditionalBayesianNetworkBase> CLGNetworkType::new_cbn(
    const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const {
    return std::make_shared<ConditionalCLGNetwork>(nodes, interface_nodes);
}

}  // namespace models