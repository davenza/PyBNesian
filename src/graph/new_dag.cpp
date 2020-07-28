#include <graph/new_dag.hpp>


namespace graph {

    std::string DirectedGraph::parents_to_string(int idx) const {
        auto pa = parents(idx);
        if (!pa.empty()) {
            std::string str = "[" + pa[0];
            for (auto it = pa.begin() + 1; it != pa.end(); ++it) {
                str += ", " + *it;
            }
            str += "]";
            return str;
        } else {
            return "[]";
        } 
    }

    bool DirectedGraph::has_path(int source, int target) const {
        check_valid_indices(source, target);

        if (has_arc(source, target))
            return true;
        else {
            const auto& children = m_nodes[source].children();
            std::vector<int> stack {children.begin(), children.end()};

            while (!stack.empty()) {
                auto v = stack.back();
                stack.pop_back();

                const auto& children = m_nodes[v].children();
        
                if (children.find(target) != children.end())
                    return true;
                
                stack.insert(stack.end(), children.begin(), children.end());
            }

            return false;
        }

    }
}