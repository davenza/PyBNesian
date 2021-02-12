#ifndef PYBNESIAN_FACTORS_FACTORS_HPP
#define PYBNESIAN_FACTORS_FACTORS_HPP

#include <string>
#include <stdint.h>
#include <cstddef>
#include <stdexcept>

namespace factors {

    class FactorType {
    public:
        enum Value : uint8_t {
            LinearGaussianCPD,
            CKDE,
            DiscreteFactor
        };

        struct Hash {
            inline std::size_t operator ()(FactorType const node_type) const {
                return static_cast<std::size_t>(node_type.value);
            }
        };

        using HashType = Hash;

        FactorType() = default;
        constexpr FactorType(Value node_type) : value(node_type) { }
        constexpr FactorType(uint8_t node_type) : value(static_cast<Value>(node_type)) { }

        operator Value() const { return value; }  
        explicit operator bool() = delete;

        constexpr bool operator==(FactorType a) const { return value == a.value; }
        constexpr bool operator==(Value v) const { return value == v; }
        constexpr bool operator!=(FactorType a) const { return value != a.value; }
        constexpr bool operator!=(Value v) const { return value != v; }

        std::string ToString() const { 
            switch(value) {
                case Value::LinearGaussianCPD:
                    return "LinearGaussianCPD";
                case Value::CKDE:
                    return "CKDE";
                case Value::DiscreteFactor:
                    return "DiscreteFactor";
                default:
                    throw std::invalid_argument("Unreachable code in BayesianNetworkType.");
            }
        }

        FactorType opposite_semiparametric() const {
            if (value == FactorType::LinearGaussianCPD) 
                return FactorType::CKDE;
            else if (value == FactorType::CKDE)
                return FactorType::LinearGaussianCPD;
            else
                throw std::invalid_argument("Not valid FactorType for SemiparametricCPD.");
        }
    private:
        Value value;
    };

}


#endif //PYBNESIAN_FACTORS_FACTORS_HPP