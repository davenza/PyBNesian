#include <graph/new_dag.hpp>


namespace graph {


    bool DirectedGraph::has_path(int source, int target) const {
        if (is_valid(source) && is_valid(target)) {
            
            if (has_arc(source, target))
                return true;
            else {
                const auto& children = m_nodes[source].children();
                std::vector<int> stack(children.begin(), children.end());

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

        } else {
            // TODO: Raise error?
        }
    }
}