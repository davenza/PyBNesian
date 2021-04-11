#include <models/KDENetwork.hpp>

namespace models {

std::shared_ptr<BayesianNetworkBase> KDENetworkType::new_bn(const std::vector<std::string>& nodes) const {
    return std::make_shared<KDENetwork>(nodes);
}

std::shared_ptr<ConditionalBayesianNetworkBase> KDENetworkType::new_cbn(
    const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const {
    return std::make_shared<ConditionalKDENetwork>(nodes, interface_nodes);
}

}  // namespace models