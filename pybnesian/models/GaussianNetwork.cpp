#include <models/GaussianNetwork.hpp>

namespace models {

std::shared_ptr<BayesianNetworkBase> GaussianNetworkType::new_bn(const std::vector<std::string>& nodes) const {
    return std::make_shared<GaussianNetwork>(nodes);
}
std::shared_ptr<ConditionalBayesianNetworkBase> GaussianNetworkType::new_cbn(
    const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const {
    return std::make_shared<ConditionalGaussianNetwork>(nodes, interface_nodes);
}

}  // namespace models