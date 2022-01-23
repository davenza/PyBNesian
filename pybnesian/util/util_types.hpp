#ifndef PYBNESIAN_UTIL_UTIL_TYPES_HPP
#define PYBNESIAN_UTIL_UTIL_TYPES_HPP

#include <optional>
#include <vector>
#include <factors/factors.hpp>
#include <graph/graph_types.hpp>

using factors::FactorType;
using graph::Arc, graph::ArcHash, graph::Edge, graph::EdgeHash, graph::EdgeEqualTo;

namespace util {

using ArcStringVector = std::vector<std::pair<std::string, std::string>>;
using EdgeStringVector = std::vector<std::pair<std::string, std::string>>;

using PairNameType = std::pair<std::string, std::shared_ptr<FactorType>>;
using FactorTypeVector = std::vector<PairNameType>;

FactorTypeVector& keep_FactorTypeVector_python_alive(FactorTypeVector& v);
FactorTypeVector keep_FactorTypeVector_python_alive(const FactorTypeVector& v);

struct NameFactorTypeHash {
    size_t operator()(const PairNameType& p) const {
        size_t h = std::hash<std::string>{}(p.first);
        util::hash_combine(h, p.second->hash());
        return h;
    }
};

struct NameFactorTypeEqualTo {
    bool operator()(const PairNameType& lhs, const PairNameType& rhs) const {
        return lhs.first == rhs.first && *lhs.second == *rhs.second;
    }
};

using FactorTypeSet = std::unordered_set<PairNameType, NameFactorTypeHash, NameFactorTypeEqualTo>;

struct FactorTypeHash {
    size_t operator()(const std::shared_ptr<FactorType>& ft) const { return ft->hash(); }
};

struct FactorTypeEqualTo {
    bool operator()(const std::shared_ptr<FactorType>& lhs, const std::shared_ptr<FactorType>& rhs) const {
        return *lhs == *rhs;
    }
};

using ArcSet = std::unordered_set<Arc, ArcHash>;
using EdgeSet = std::unordered_set<Edge, EdgeHash, EdgeEqualTo>;

class random_seed_arg {
public:
    random_seed_arg() : m_value(std::random_device{}()) {}
    random_seed_arg(unsigned int arg) : m_value(arg) {}
    random_seed_arg(std::optional<unsigned int> arg) {
        if (arg) {
            m_value = *arg;
        } else {
            m_value = std::random_device{}();
        }
    }

    operator unsigned int() { return m_value; }

private:
    unsigned int m_value;
};

}  // namespace util

#endif  // PYBNESIAN_UTIL_UTIL_TYPES_HPP