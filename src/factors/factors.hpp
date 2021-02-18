#ifndef PYBNESIAN_FACTORS_FACTORS_HPP
#define PYBNESIAN_FACTORS_FACTORS_HPP

#include <string>
#include <stdint.h>
#include <cstddef>
#include <stdexcept>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace factors {

    class NodeType {
    public:
        enum Value : uint8_t {
            LinearGaussianCPD,
            CKDE,
            SemiparametricCPD,
            DiscreteFactor
        };

        struct Hash {
            inline std::size_t operator ()(NodeType const node_type) const {
                return static_cast<std::size_t>(node_type.value);
            }
        };

        using HashType = Hash;

        NodeType() = default;
        constexpr NodeType(Value node_type) : value(node_type) { }
        constexpr NodeType(uint8_t node_type) : value(static_cast<Value>(node_type)) { }

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
                case Value::SemiparametricCPD:
                    return "SemiparametricCPD";
                case Value::DiscreteFactor:
                    return "DiscreteFactor";
                default:
                    throw std::invalid_argument("Unreachable code in NodeType.");
            }
        }

        static NodeType from_string(const std::string& name) {
            if (name == "LinearGaussianCPD")
                return NodeType::LinearGaussianCPD;
            else if(name == "CKDE")
                return NodeType::CKDE;
            else if (name == "SemiparametricCPD")
                return NodeType::SemiparametricCPD;
            else if (name == "DiscreteFactor")
                return NodeType::DiscreteFactor;
            else
                throw std::invalid_argument("Not valid NodeType.");
        }

        NodeType opposite_semiparametric() const {
            if (value == NodeType::LinearGaussianCPD) 
                return NodeType::CKDE;
            else if (value == NodeType::CKDE)
                return NodeType::LinearGaussianCPD;
            else
                throw std::invalid_argument("Not valid NodeType for SemiparametricCPD.");
        }
    private:
        Value value;
    };

    template<typename F>
    void save_factor(const F& factor, std::string name) {
        auto open = py::module::import("io").attr("open");
        
        if (name.size() < 7 || name.substr(name.size()-7) != ".pickle")
            name += ".pickle";

        auto file = open(name, "wb");
        py::module::import("pickle").attr("dump")(py::cast(&factor), file, 2);
        file.attr("close")();
    }

    py::object load_factor(const std::string& name);
}


#endif //PYBNESIAN_FACTORS_FACTORS_HPP