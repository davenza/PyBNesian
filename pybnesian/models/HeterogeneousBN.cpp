#include <models/HeterogeneousBN.hpp>

namespace models {

std::shared_ptr<BayesianNetworkBase> HeterogeneousBNType::new_bn(const std::vector<std::string>& nodes) const {
    return std::make_shared<HeterogeneousBN>(m_default_ftype, nodes);
}

std::shared_ptr<ConditionalBayesianNetworkBase> HeterogeneousBNType::new_cbn(
    const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const {
    return std::make_shared<ConditionalHeterogeneousBN>(m_default_ftype, nodes, interface_nodes);
}

}  // namespace models