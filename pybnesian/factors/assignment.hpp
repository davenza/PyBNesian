#ifndef PYBNESIAN_FACTORS_ASSIGNMENT_HPP
#define PYBNESIAN_FACTORS_ASSIGNMENT_HPP

#include <variant>
#include <Eigen/Dense>
#include <util/hash_utils.hpp>

using Eigen::VectorXi;

namespace factors {

class AssignmentValue {
public:
    AssignmentValue(const std::string& v) : m_value(v) {}
    AssignmentValue(double v) : m_value(v) {}
    AssignmentValue(const AssignmentValue&) = default;
    AssignmentValue(std::variant<std::string, double> v) : m_value(v) {}

    bool operator==(const std::string& other) const {
        return std::visit(
            [&other](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<std::string, T>)
                    return v == other;
                else
                    return false;
            },
            m_value);
    }

    bool operator!=(const std::string& other) const { return !(*this == other); }

    bool operator==(double other) const {
        return std::visit(
            [other](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<double, T>)
                    return v == other;
                else
                    return false;
            },
            m_value);
    }

    bool operator!=(double other) const { return !(*this == other); }

    bool operator==(const AssignmentValue& other) const {
        return std::visit(
            [](auto&& v, auto&& o) {
                using T = std::decay_t<decltype(v)>;
                using T2 = std::decay_t<decltype(o)>;

                if constexpr (std::is_same_v<T, T2>) {
                    return v == o;
                } else {
                    return false;
                }
            },
            m_value,
            other.m_value);
    }

    bool operator!=(const AssignmentValue& other) const { return !(*this == other); }

    size_t hash() const {
        return std::visit([](auto&& v) { return std::hash<std::decay_t<decltype(v)>>{}(v); }, m_value);
    }

    operator std::string() const {
        auto str = std::get_if<std::string>(&m_value);

        if (str) {
            return *str;
        } else {
            throw std::runtime_error("Assignment value is not string.");
        }
    }

    operator double() const {
        auto v = std::get_if<double>(&m_value);

        if (v) {
            return *v;
        } else {
            throw std::runtime_error("Assignment value is not double.");
        }
    }

    std::string ToString() const {
        return std::visit(
            [](auto&& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<std::string, T>)
                    return v;
                else if constexpr (std::is_same_v<double, T>)
                    return std::to_string(v);
                else
                    static_assert(util::always_false<T>, "Not supported type.");
            },
            m_value);
    }

    py::object __getstate__() const { return py::cast(m_value); }

    static AssignmentValue __setstate__(py::object& o) {
        return AssignmentValue(o.cast<std::variant<std::string, double>>());
    }

private:
    std::variant<std::string, double> m_value;
};

class Assignment {
public:
    using iterator = typename std::unordered_map<std::string, AssignmentValue>::iterator;
    using const_iterator = typename std::unordered_map<std::string, AssignmentValue>::const_iterator;
    using value_type = typename std::unordered_map<std::string, AssignmentValue>::value_type;
    using size_type = typename std::unordered_map<std::string, AssignmentValue>::size_type;

    Assignment() = default;
    Assignment(std::unordered_map<std::string, AssignmentValue> ass) : m_assignment(ass) {}
    Assignment(const Assignment&) = default;

    // Based on the hashing of a frozenset
    // https://stackoverflow.com/questions/20832279/python-frozenset-hashing-algorithm-implementation
    // https://github.com/python/cpython/blob/main/Objects/setobject.c
    size_t hash() const {
        size_t hash = 0;

        std::hash<std::string> hfunction;

        for (const auto& item : m_assignment) {
            auto partial_hash = hfunction(item.first);
            util::hash_combine(partial_hash, item.second.hash());

            hash ^= ((partial_hash ^ 89869747UL) ^ (partial_hash << 16)) * 3644798167UL;
        }

        /* Factor in the number of active entries */
        hash ^= (m_assignment.size() + 1) * 1927868237UL;
        // hash = hash * 69069U + 907133923UL;

        return hash;
    }

    const AssignmentValue& value(const std::string& var) const {
        auto it = m_assignment.find(var);

        if (it == m_assignment.end()) {
            throw std::invalid_argument("Variable " + var + " not found in the assignment.");
        }

        return it->second;
    }

    template <typename It>
    bool has_variables(It begin, It end) const {
        for (auto it = begin; it != end; ++it) {
            if (m_assignment.find(*it) == m_assignment.end()) return false;
        }

        return false;
    }

    size_t index(const std::vector<std::string>& variables,
                 const std::vector<std::vector<std::string>>& variable_values,
                 const VectorXi& strides) const {
        size_t index = 0;

        for (size_t i = 0; i < variables.size(); ++i) {
            auto vindex = std::distance(variable_values[i].begin(),
                                        std::find(variable_values[i].begin(),
                                                  variable_values[i].end(),
                                                  static_cast<std::string>(value(variables[i]))));

            index += vindex * strides(i);
        }

        return index;
    }

    static Assignment from_index(int index,
                                 const std::vector<std::string>& variables,
                                 const std::vector<std::vector<std::string>>& variable_values,
                                 const VectorXi& cardinality,
                                 const VectorXi& strides) {
        std::unordered_map<std::string, AssignmentValue> map;

        for (size_t i = 0, i_end = variables.size(); i < i_end; ++i) {
            auto vindex = (index / strides(i)) % cardinality(i);

            map.insert({variables[i], variable_values[i][vindex]});
        }

        return Assignment(map);
    }

    bool empty() const noexcept { return m_assignment.empty(); }

    size_type size() const noexcept { return m_assignment.size(); }

    std::pair<iterator, bool> insert(const value_type& value) { return m_assignment.insert(value); }

    std::pair<iterator, bool> insert(value_type&& value) {
        return m_assignment.insert(std::forward<value_type>(value));
    }

    const_iterator begin() const noexcept { return m_assignment.begin(); }

    const_iterator cbegin() const noexcept { return m_assignment.cbegin(); }

    const_iterator end() const noexcept { return m_assignment.end(); }

    const_iterator cend() const noexcept { return m_assignment.cend(); }

    std::string ToString() const {
        if (m_assignment.empty()) {
            return "[]";
        } else {
            std::stringstream ss;
            auto begin = m_assignment.begin();
            ss << "[" << begin->first << " = " << begin->second.ToString();

            for (auto it = ++begin, end = m_assignment.end(); it != end; ++it) {
                ss << ", " << it->first << " = " << it->second.ToString();
            }

            ss << "]";

            return ss.str();
        }
    }

    bool operator==(const Assignment& other) const { return m_assignment == other.m_assignment; }

    bool operator!=(const Assignment& other) const { return m_assignment != other.m_assignment; }

    py::object __getstate__() const { return py::cast(m_assignment); }

    static Assignment __setstate__(py::object& o) {
        return Assignment(o.cast<std::unordered_map<std::string, AssignmentValue>>());
    }

private:
    std::unordered_map<std::string, AssignmentValue> m_assignment;
};

struct AssignmentHash {
    size_t operator()(const Assignment& ass) const { return ass.hash(); }
};

}  // namespace factors

#endif  // PYBNESIAN_FACTORS_ASSIGNMENT_HPP