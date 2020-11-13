#ifndef PYBNESIAN_ASSIGNMENT_HPP
#define PYBNESIAN_ASSIGNMENT_HPP

namespace util {

    template<typename V>
    class Assignment {
    public:
        Assignment() : m_map() {}

        void insert(const std::string& variable, V value) {
            m_map.insert(std::make_pair(variable, value));
        }

        V get(const std::string& variable) {
            
        }
    
    private:
        std::unordered_map<std::string, V> m_map;
    };
}

#endif //PYBNESIAN_ASSIGNMENT_HPP