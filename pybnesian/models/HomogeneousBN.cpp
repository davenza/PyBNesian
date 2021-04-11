#include <models/HomogeneousBN.hpp>

namespace models {

std::shared_ptr<BayesianNetworkBase> HomogeneousBNType::new_bn(const std::vector<std::string>& nodes) const {
    return std::make_shared<HomogeneousBN>(m_ftype, nodes);
}

std::shared_ptr<ConditionalBayesianNetworkBase> HomogeneousBNType::new_cbn(
    const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const {
    return std::make_shared<ConditionalHomogeneousBN>(m_ftype, nodes, interface_nodes);
}

}  // namespace models