#ifndef PGM_DATASET_SEMIPARAMETERICBN_NODETYPE_HPP
#define PGM_DATASET_SEMIPARAMETERICBN_NODETYPE_HPP

namespace models {

    class NodeType {
    public:
        enum Value : uint8_t {
            LinearGaussianCPD,
            CKDE
        };

        struct Hash {
            inline std::size_t operator ()(NodeType const node_type) const {
                return static_cast<std::size_t>(node_type.value);
            }
        };

        using HashType = Hash;

        NodeType() = default;
        constexpr NodeType(Value node_type) : value(node_type) { }

        operator Value() const { return value; }  
        explicit operator bool() = delete;

        constexpr bool operator==(NodeType a) const { return value == a.value; }
        constexpr bool operator==(Value v) const { return value == v; }
        constexpr bool operator!=(NodeType a) const { return value != a.value; }
        constexpr bool operator!=(Value v) const { return value != v; }

        std::string ToString() const { 
            switch(value) {
                case Value::LinearGaussianCPD:
                    return "LinearGaussianCPD";
                case Value::CKDE:
                    return "CKDE";
                default:
                    throw std::invalid_argument("Unreachable code in BayesianNetworkType.");
            }
        }

        NodeType opposite() {
            if (value == NodeType::LinearGaussianCPD) 
                return NodeType::CKDE;
            else
                return NodeType::LinearGaussianCPD;
        }

    private:
        Value value;
    };

}

#endif //PGM_DATASET_SEMIPARAMETERICBN_NODETYPE_HPP