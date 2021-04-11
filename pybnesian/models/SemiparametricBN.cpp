#include <models/SemiparametricBN.hpp>

namespace models {

std::shared_ptr<BayesianNetworkBase> SemiparametricBNType::new_bn(const std::vector<std::string>& nodes) const {
    return std::make_shared<SemiparametricBN>(nodes);
}

std::shared_ptr<ConditionalBayesianNetworkBase> SemiparametricBNType::new_cbn(
    const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const {
    return std::make_shared<ConditionalSemiparametricBN>(nodes, interface_nodes);
}

}  // namespace models