#ifndef PGM_DATASET_UTIL_TYPES_HPP
#define PGM_DATASET_UTIL_TYPES_HPP

#include <models/SemiparametricBN_NodeType.hpp>

using models::NodeType;

namespace util {
    using ArcVector = std::vector<std::pair<std::string, std::string>>;
    using NodeTypeVector = std::vector<std::pair<std::string, NodeType>>;
}

#endif //PGM_DATASET_VALIDATEDTYPE_HPP