// For std::iota
#include <numeric>
#include <indicators/cursor_control.hpp>
#include <learning/algorithms/constraint.hpp>
#include <learning/algorithms/mmpc.hpp>
#include <Eigen/Dense>
#include <util/combinations.hpp>
#include <util/progress.hpp>
#include <util/vector.hpp>
#include <util/validate_whitelists.hpp>

using Eigen::VectorXd, Eigen::VectorXi, Eigen::MatrixXd;
using util::Combinations, util::AllSubsets;

namespace learning::algorithms {

    enum MMPC_Progress {
        MMPC_FORWARD_PHASE_STOP = -1,
        MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC = -2
    };

    struct CPCAssoc {
        MatrixXd min_assoc;
        VectorXd maxmin_assoc;
        VectorXi maxmin_index;
    };

    template<typename BN>
    class BNCPCAssoc;

    template<>
    class BNCPCAssoc<PartiallyDirectedGraph> {

        BNCPCAssoc(const PartiallyDirectedGraph& g) : m_graph(g),
                                                      m_assoc() {
            m_assoc = CPCAssoc {
                .min_assoc = MatrixXd::Zero(g.num_nodes(), g.num_nodes()),
                .maxmin_assoc = VectorXd::Constant(g.num_nodes(), std::numeric_limits<double>::infinity()),
                .maxmin_index = VectorXi::Constant(g.num_nodes(), MMPC_FORWARD_PHASE_STOP)
            };
        }

        double& min_assoc(int row_index, int col_index) {
            return m_assoc.min_assoc(row_index, col_index);
        }

        double min_assoc(int row_index, int col_index) const {
            return m_assoc.min_assoc(row_index, col_index);
        }

        double& maxmin_assoc(int index) {
            return m_assoc.maxmin_assoc(index);
        }

        double maxmin_assoc(int index) const {
            return m_assoc.maxmin_assoc(index);
        }

        int& maxmin_index(int index) {
            return m_assoc.maxmin_index(index);
        }

        int maxmin_index(int index) const {
            return m_assoc.maxmin_index(index);
        }
    private:
        const PartiallyDirectedGraph& m_graph;
        CPCAssoc m_assoc;
    };

    template<>
    class BNCPCAssoc<ConditionalPartiallyDirectedGraph> {
    public:
        BNCPCAssoc(const ConditionalPartiallyDirectedGraph& g) : m_graph(g),
                                                                 m_assoc(),
                                                                 m_interface_assoc() {
            m_assoc = CPCAssoc {
                .min_assoc = MatrixXd::Zero(g.num_total_nodes(), g.num_nodes()),
                .maxmin_assoc = VectorXd::Constant(g.num_nodes(), std::numeric_limits<double>::infinity()),
                .maxmin_index = VectorXi::Constant(g.num_nodes(), MMPC_FORWARD_PHASE_STOP)
            };

            m_interface_assoc = CPCAssoc {
                .min_assoc = MatrixXd::Zero(g.num_nodes(), g.num_interface_nodes()),
                .maxmin_assoc = VectorXd::Constant(g.num_interface_nodes(), std::numeric_limits<double>::infinity()),
                .maxmin_index = VectorXi::Constant(g.num_interface_nodes(), MMPC_FORWARD_PHASE_STOP)
            };
        }

        double& min_assoc_node(int row_index, int col_index) {
            return m_assoc.min_assoc(m_graph.joint_collapsed_from_index(row_index), 
                                     m_graph.collapsed_from_index(col_index));
        }

        double min_assoc_node(int row_index, int col_index) const {
            return m_assoc.min_assoc(m_graph.joint_collapsed_from_index(row_index), 
                                     m_graph.collapsed_from_index(col_index));
        }

        double& min_assoc_interface(int row_index, int col_index) {
            return m_interface_assoc.min_assoc(m_graph.collapsed_from_index(row_index), 
                                               m_graph.interface_collapsed_from_index(col_index));
        }

        double min_assoc_interface(int row_index, int col_index) const {
            return m_interface_assoc.min_assoc(m_graph.collapsed_from_index(row_index), 
                                               m_graph.interface_collapsed_from_index(col_index));
        }

        double& min_assoc(int row_index, int col_index) {
            if (m_graph.is_interface(col_index))
                return min_assoc_interface(row_index, col_index);
            else
                return min_assoc_node(row_index, col_index);
        }

        double min_assoc(int row_index, int col_index) const {
            if (m_graph.is_interface(col_index))
                return min_assoc_interface(row_index, col_index);
            else
                return min_assoc_node(row_index, col_index);
        }


        double& maxmin_assoc_node(int index) {
            return m_assoc.maxmin_assoc(m_graph.collapsed_from_index(index));
        }

        double maxmin_assoc_node(int index) const {
            return m_assoc.maxmin_assoc(m_graph.collapsed_from_index(index));
        }

        double& maxmin_assoc_interface(int index) {
            return m_interface_assoc.maxmin_assoc(m_graph.interface_collapsed_from_index(index));
        }

        double maxmin_assoc_interface(int index) const {
            return m_interface_assoc.maxmin_assoc(m_graph.interface_collapsed_from_index(index));
        }

        double& maxmin_assoc(int index) {
            if (g.is_interface(index))
                return maxmin_assoc_interface(index);
            else
                return maxmin_assoc_node(index);
        }

        double maxmin_assoc(int index) const {
            if (g.is_interface(index))
                return maxmin_assoc_interface(index);
            else
                return maxmin_assoc_node(index);
        }

        int& maxmin_index_node(int index) {
            return m_assoc.maxmin_index(m_graph.collapsed_from_index(index));
        }

        int maxmin_index_node(int index) const {
            return m_assoc.maxmin_index(m_graph.collapsed_from_index(index));
        }

        int& maxmin_index_interface(int index) {
            return m_interface_assoc.maxmin_index(m_graph.interface_collapsed_from_index(index));
        }

        int maxmin_index_interface(int index) const {
            return m_interface_assoc.maxmin_index(m_graph.interface_collapsed_from_index(index));
        }

        int& maxmin_index(int index) {
            if (g.is_interface(index))
                return maxmin_index_interface(index);
            else
                return maxmin_index_node(index);
        }

        int maxmin_index(int index) const {
            if (g.is_interface(index))
                return maxmin_index_interface(index);
            else
                return maxmin_index_node(index);
        }
    private:
        const ConditionalPartiallyDirectedGraph& m_graph;
        CPCAssoc m_assoc;
        CPCAssoc m_interface_assoc;
    };

    template<typename VectorType>
    void recompute_assoc(const IndependenceTest& test,
                         const PartiallyDirectedGraph& g,
                         int variable,
                         const std::unordered_set<int>& cpc,
                         const std::unordered_set<int>& to_be_checked,
                         VectorType& min_assoc,
                         util::BaseProgressBar& progress) {

        const auto& variable_name = g.name(variable);
        progress.set_text("MMPC Forward: sepset order " + std::to_string(cpc.size()) +  " for " + variable_name);
        progress.set_max_progress(to_be_checked.size());
        progress.set_progress(0);

        
        std::vector<std::string> cpc_vec;
        cpc_vec.reserve(cpc.size());
        for (auto c : cpc) {
            cpc_vec.push_back(g.name(c));
        }

        for (auto other : to_be_checked) {
            min_assoc(other) = test.pvalue(variable_name, g.name(other), cpc_vec.begin(), cpc_vec.end());
            progress.tick();
        }
    }

    template<typename VectorType>
    void update_min_assoc(const IndependenceTest& test,
                          const PartiallyDirectedGraph& g,
                          int variable,
                          const std::unordered_set<int>& to_be_checked,
                          const std::unordered_set<int>& cpc,
                          VectorType& min_assoc,
                          int last_added_cpc,
                          util::BaseProgressBar& progress) {
                
        const auto& variable_name = g.name(variable);

        if (cpc.empty()) {
            progress.set_text("MMPC Forward: no sepset for " + variable_name);
            progress.set_max_progress(to_be_checked.size());
            progress.set_progress(0);

            for (auto v : to_be_checked) {
                min_assoc(v) = test.pvalue(variable_name, g.name(v));
                progress.tick();
            }
        } else if (cpc.size() == 1) {
            progress.set_text("MMPC Forward: sepset order 1 for " + variable_name);
            progress.set_max_progress(to_be_checked.size());
            progress.set_progress(0);

            const auto& last_added_name = g.name(last_added_cpc);
            for (auto v : to_be_checked) {
                double pvalue = test.pvalue(variable_name, g.name(v), last_added_name);
                min_assoc(v) = std::max(min_assoc(v), pvalue);
                progress.tick();
            }
        } else if (cpc.size() == 2) {
            const auto& last_added_name = g.name(last_added_cpc);
            
            std::vector<std::string> cond;
            cond.reserve(2);
            for (auto pc : cpc) {
                cond.push_back(g.name(pc));
            }

            progress.set_text("MMPC Forward: sepset order 2 for " + variable_name);
            progress.set_max_progress(to_be_checked.size());
            progress.set_progress(0);

            for (auto v : to_be_checked) {
                const auto& v_name = g.name(v);

                double pvalue = test.pvalue(variable_name, v_name, last_added_name);
                min_assoc(v) = std::max(min_assoc(v), pvalue);

                pvalue = test.pvalue(variable_name, v_name, cond.begin(), cond.end());
                min_assoc(v) = std::max(min_assoc(v), pvalue);

                progress.tick();
            }
        } else {
            progress.set_text("MMPC Forward: sepset up to order " + std::to_string(cpc.size()) + " for " + variable_name);
            progress.set_max_progress(to_be_checked.size());
            progress.set_progress(0);

            const auto& last_added_name = g.name(last_added_cpc);

            std::vector<std::string> fixed = {last_added_name};

            std::vector<std::string> old_cpc;
            old_cpc.reserve(cpc.size());
            for (auto pc : cpc) {
                if (pc != last_added_cpc) {
                    old_cpc.push_back(g.name(pc));
                }
            }

            std::vector<std::string> cond(2);
            cond[1] = last_added_name;

            // Conditioning in all the subsets of 2 to CPC.size()-1 size, including last variable added.
            AllSubsets<std::string> comb;
            if (cpc.size() > 3) {
                comb = AllSubsets(old_cpc, std::move(fixed), 3, cpc.size()-1);
            }

            for (auto v : to_be_checked) {
                const auto& v_name = g.name(v);
                // Conditioning in just the last variable added.
                double pvalue = test.pvalue(variable_name, v_name, last_added_name);
                min_assoc(v) = std::max(min_assoc(v), pvalue);

                // Conditioning in the last variable and another variable added.
                for (const auto& pc : old_cpc) {
                    cond[0] = pc;
                    pvalue = test.pvalue(variable_name, v_name, cond.begin(), cond.end());
                    min_assoc(v) = std::max(min_assoc(v), pvalue);
                }
                
                if (cpc.size() > 3) {
                    for (const auto& subset : comb) {
                        pvalue = test.pvalue(variable_name, v_name, subset.begin(), subset.end());
                        min_assoc(v) = std::max(min_assoc(v), pvalue);
                    }
                }

                // Conditioning in all the variables.
                old_cpc.push_back(last_added_name);
                pvalue = test.pvalue(variable_name, v_name, old_cpc.begin(), old_cpc.end());
                min_assoc(v) = std::max(min_assoc(v), pvalue);
                old_cpc.pop_back();
            }

            progress.tick();
        }
    }

    int find_maxmin_assoc(const VectorXd& min_assoc, std::unordered_set<int>& to_be_checked, double alpha) {
        int to_add = MMPC_FORWARD_PHASE_STOP;
        double to_add_pvalue = std::numeric_limits<double>::infinity();

        for (auto it = to_be_checked.begin(), end = to_be_checked.end(); it != end;) {
            if (min_assoc(*it) > alpha) {
                it = to_be_checked.erase(it);
            } else {
                if (min_assoc(*it) < to_add_pvalue) {
                    to_add = *it;
                    to_add_pvalue = min_assoc(*it);
                }
                ++it;
            }
        }

        return to_add;
    }

    template<typename G, typename VectorType>
    std::unordered_set<int> mmpc_forward_phase(const IndependenceTest& test,
                                               const G& g,
                                               int variable,
                                               double alpha,
                                               std::unordered_set<int>& cpc,
                                               std::unordered_set<int>& to_be_checked,
                                               VectorType& min_assoc,
                                               int last_added,
                                               util::BaseProgressBar& progress) {
        bool changed_cpc = true;

        if (cpc.empty()) {
            min_assoc.fill(0);
        } else if (last_added == MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC) {
            // The CPC is not empty because of whitelists, so we compute the association of the selected CPC.
            recompute_assoc(test, g, variable, cpc, to_be_checked, min_assoc, progress);
            int to_add = find_maxmin_assoc(min_assoc, to_be_checked, alpha);

            if (to_add != MMPC_FORWARD_PHASE_STOP) {
                cpc.insert(to_add);
                to_be_checked.erase(to_add);
                last_added = to_add;
            } else {
                changed_cpc = false;
            }
        }

        while (changed_cpc && !to_be_checked.empty()) {
            update_min_assoc(test, g, variable, to_be_checked, cpc, min_assoc, last_added, progress);
            int to_add = find_maxmin_assoc(min_assoc, to_be_checked, alpha);

            if (to_add != MMPC_FORWARD_PHASE_STOP) {
                cpc.insert(to_add);
                to_be_checked.erase(to_add);
                last_added = to_add;
            } else {
                changed_cpc = false;
            }
        }

        return cpc;
    }

    // template<typename VectorType>
    // std::unordered_set<int> mmpc_forward_phase_node(const IndependenceTest& test,
    //                                            const ConditionalPartiallyDirectedGraph& g,
    //                                            int variable,
    //                                            double alpha,
    //                                            std::unordered_set<int>& cpc,
    //                                            std::unordered_set<int>& to_be_checked,
    //                                            VectorType& min_assoc,
    //                                            int last_added,
    //                                            util::BaseProgressBar& progress) {
    
    // }

    // template<typename VectorType>
    // std::unordered_set<int> mmpc_forward_phase_interface(const IndependenceTest& test,
    //                                            const ConditionalPartiallyDirectedGraph& g,
    //                                            int variable,
    //                                            double alpha,
    //                                            std::unordered_set<int>& cpc,
    //                                            std::unordered_set<int>& to_be_checked,
    //                                            VectorType& min_assoc,
    //                                            int last_added,
    //                                            util::BaseProgressBar& progress) {
    
    // }

    bool is_whitelisted_pc(int variable, int candidate_pc, const ArcSet& arc_whitelist, const EdgeSet& edge_whitelist) {
        return edge_whitelist.count({variable, candidate_pc}) > 0 || 
               arc_whitelist.count({variable, candidate_pc}) > 0 || 
               arc_whitelist.count({candidate_pc, variable}) > 0;
    }

    void mmpc_backward_phase(const IndependenceTest& test,
                             const PartiallyDirectedGraph& g,
                             int variable,
                             double alpha,
                             std::unordered_set<int>& cpc,
                             const ArcSet& arc_whitelist,
                             const EdgeSet& edge_whitelist,
                             util::BaseProgressBar& progress) {
        
        const auto& variable_name = g.name(variable);
        
        if (cpc.size() > 1) {
            std::vector<std::string> subset_variables;
            subset_variables.reserve(cpc.size());
            for (auto pc : cpc) {
                subset_variables.push_back(g.name(pc));
            }

            progress.set_text("MMPC Backwards for " + variable_name);
            progress.set_max_progress(cpc.size());
            progress.set_progress(0);

            for (auto it = cpc.begin(), end = cpc.end(); it != end;) {
                if (is_whitelisted_pc(variable, *it, arc_whitelist, edge_whitelist)) {
                    ++it;
                    progress.tick();
                    continue;
                }

                const auto& it_name = g.name(*it);
                util::swap_remove_v(subset_variables, it_name);

                // Marginal independence
                if (test.pvalue(variable_name, it_name) > alpha) {
                    it = cpc.erase(it);
                    progress.set_max_progress(cpc.size());
                    progress.tick();
                    continue;
                }

                // Independence sepset length 1.
                bool found_sepset = false;
                for (auto it_other = subset_variables.begin(), end_other = subset_variables.end(); it_other != end_other; ++it_other) {
                    if (test.pvalue(variable_name, it_name, *it_other) > alpha) {
                        it = cpc.erase(it);
                        progress.set_max_progress(cpc.size());
                        found_sepset = true;
                        break;
                    }
                }

                if (!found_sepset && subset_variables.size() > 2) {
                    // Independence sepset length 2 to subset size - 1.
                    AllSubsets comb(subset_variables, 2, subset_variables.size()-1);

                    for (const auto& s : comb) {
                        if (test.pvalue(variable_name, it_name, s.begin(), s.end()) > alpha) {
                            it = cpc.erase(it);
                            progress.set_max_progress(cpc.size());
                            found_sepset = true;
                            break;
                        }
                    }
                }

                // Independence sepset length of subset size.
                if (!found_sepset && subset_variables.size() > 1 && 
                    test.pvalue(variable_name, it_name, subset_variables.begin(), subset_variables.end()) > alpha) {
                    it = cpc.erase(it);
                    progress.set_max_progress(cpc.size());
                    found_sepset = true;
                }

                if (!found_sepset) {
                    // No sepset found, so include again the variable.
                    subset_variables.push_back(it_name);
                    ++it;
                }

                progress.tick();
            }
        }
    }

    std::unordered_set<int> mmpc_variable(const IndependenceTest& test,
                                          const PartiallyDirectedGraph& g,
                                          int variable,
                                          double alpha,
                                          const ArcSet& arc_whitelist,
                                          const EdgeSet& edge_blacklist,
                                          const EdgeSet& edge_whitelist,
                                          util::BaseProgressBar& progress) {
        std::unordered_set<int> cpc;
        std::unordered_set<int> to_be_checked;

        for (int i = 0; i < g.num_nodes(); ++i) {
            if (i != variable && edge_blacklist.count({variable, i}) == 0) {
                to_be_checked.insert(i);
            }
        }

        for (const auto& edge : edge_whitelist) {
            if (edge.first == variable) {
                cpc.insert(edge.second);
                to_be_checked.erase(edge.second);
            }

            if (edge.second == variable) {
                cpc.insert(edge.first);
                to_be_checked.erase(edge.first);
            }
        }

        for (const auto& arc : arc_whitelist) {
            if (arc.first == variable) {
                cpc.insert(arc.second);
                to_be_checked.erase(arc.second);
            }

            if (arc.second == variable) {
                cpc.insert(arc.first);
                to_be_checked.erase(arc.first);
            }
        }

        VectorXd min_assoc(g.num_nodes());

        int last_added = 0;
        if (!cpc.empty()) last_added = MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC;

        mmpc_forward_phase(test, g, variable, alpha, cpc, to_be_checked, min_assoc, last_added, progress);
        mmpc_backward_phase(test, g, variable, alpha, cpc, arc_whitelist, edge_whitelist, progress);
        return cpc;
    }

    void marginal_cpcs_all_variables(const IndependenceTest& test,
                                     const PartiallyDirectedGraph& g,
                                     double alpha,
                                     std::vector<std::unordered_set<int>>& cpcs,
                                     std::vector<std::unordered_set<int>>& to_be_checked,
                                     const EdgeSet& edge_blacklist,
                                     CPCAssoc& cpc_assoc,
                                     util::BaseProgressBar& progress) {
        auto nnodes = g.num_nodes();

        progress.set_text("MMPC Forward: No sepset");
        progress.set_max_progress((nnodes*(nnodes-1) / 2));
        progress.set_progress(0);

        for (int i = 0, i_end = nnodes-1; i < i_end; ++i) {
            for (int j = i+1; j < nnodes; ++j) {
                if ((cpcs[i].empty() || cpcs[j].empty()) && edge_blacklist.count({i,j}) == 0) {
                    double pvalue = test.pvalue(g.name(i), g.name(j));
                    if (pvalue < alpha) {
                        if (cpcs[i].empty()) {
                            cpc_assoc.min_assoc(j, i) = pvalue;
                            if (cpc_assoc.min_assoc(j, i) < cpc_assoc.maxmin_assoc(i)) {
                                cpc_assoc.maxmin_assoc(i) = cpc_assoc.min_assoc(j, i);
                                cpc_assoc.maxmin_index(i) = j;
                            }
                        }
                        
                        if (cpcs[j].empty()) {
                            cpc_assoc.min_assoc(i, j) = pvalue;
                            if (cpc_assoc.min_assoc(i, j) < cpc_assoc.maxmin_assoc(j)) {
                                cpc_assoc.maxmin_assoc(j) = cpc_assoc.min_assoc(i, j);
                                cpc_assoc.maxmin_index(j) = i;
                            }
                        }
                    } else {
                        to_be_checked[i].erase(j);
                        to_be_checked[j].erase(i);
                    }
                }

                progress.tick();
            }
        }
    }

    void marginal_cpcs_all_variables(const IndependenceTest& test,
                                     const ConditionalPartiallyDirectedGraph& g,
                                     double alpha,
                                     std::vector<std::unordered_set<int>>& cpcs,
                                     std::vector<std::unordered_set<int>>& to_be_checked,
                                     const EdgeSet& edge_blacklist,
                                     CPCAssoc& node_assoc,
                                     CPCAssoc& interface_assoc,
                                     util::BaseProgressBar& progress) {
        auto nnodes = g.num_nodes();
        auto inodes = g.num_interface_nodes();

        progress.set_text("MMPC Forward: No sepset");
        progress.set_max_progress(inodes*nnodes + (nnodes*(nnodes-1) / 2));
        progress.set_progress(0);

        // Cache marginal between nodes
        for (int i = 0, i_end = nnodes-1; i < i_end; ++i) {
            const auto& i_name = g.collapsed_name(i);
            auto i_index = g.index(i_name);
            auto i_jcindex = g.joint_collapsed_index(i_name);
            for (int j = i+1; j < nnodes; ++j) {
                const auto& j_name = g.collapsed_name(j);
                auto j_index = g.index(j_name);

                if ((cpcs[i_index].empty() || cpcs[j_index].empty()) && edge_blacklist.count({i_index, j_index}) == 0) {
                    double pvalue = test.pvalue(i_name, j_name);
                    if (pvalue < alpha) {
                        if (cpcs[i_index].empty()) {
                            auto j_jcindex = g.joint_collapsed_index(j_name);
                            node_assoc.min_assoc(j_jcindex, i) = pvalue;
                            if (pvalue < node_assoc.maxmin_assoc(i)) {
                                node_assoc.maxmin_assoc(i) = pvalue;
                                node_assoc.maxmin_index(i) = j;
                            }
                        }

                        if (cpcs[j_index].empty()) {
                            node_assoc.min_assoc(i_jcindex, j) = pvalue;
                            if (pvalue < node_assoc.maxmin_assoc(j)) {
                                node_assoc.maxmin_assoc(j) = pvalue;
                                node_assoc.maxmin_index(j) = i;
                            }
                        }
                    } else {
                        to_be_checked[i_index].erase(j_index);
                        to_be_checked[j_index].erase(i_index);
                    }
                }

                progress.tick();
            }
        }

        // Cache between nodes and interface_nodes
        for (const auto& node : g.nodes()) {
            auto nindex = g.index(node);
            auto ncollapsed = g.collapsed_index(node);
            for (const auto& inode : g.interface_nodes()) {
                auto iindex = g.index(inode);

                if ((cpcs[nindex].empty() || cpcs[iindex].empty()) && edge_blacklist.count({nindex, iindex}) == 0) {
                    double pvalue = test.pvalue(node, inode);
                    if (pvalue < alpha) {
                        if (cpcs[nindex].empty()) {
                            auto i_jcindex = g.joint_collapsed_index(inode);

                            node_assoc.min_assoc(i_jcindex, ncollapsed) = pvalue;
                            if (pvalue < node_assoc.maxmin_assoc(ncollapsed)) {
                                node_assoc.maxmin_assoc(ncollapsed) = pvalue;
                                node_assoc.maxmin_index(ncollapsed) = iindex;
                            }
                        }

                        if (cpcs[iindex].empty()) {
                            auto icollapsed = g.interface_collapsed_index(inode);

                            interface_assoc.min_assoc(ncollapsed, icollapsed) = pvalue;
                            if (pvalue < interface_assoc.maxmin_assoc(icollapsed)) {
                                interface_assoc.maxmin_assoc(icollapsed) = pvalue;
                                interface_assoc.maxmin_index(icollapsed) = nindex;
                            }
                        }
                    } else {
                        to_be_checked[nindex].erase(iindex);
                        to_be_checked[iindex].erase(nindex);
                    }
                }

                progress.tick();
            }
        }
    }

    bool update_univariate_assoc(const PartiallyDirectedGraph& g,
                                double pvalue,
                                int node,
                                int other_node,
                                double alpha,
                                CPCAssoc& node_assoc) {
        

        double new_max = node_assoc.min_assoc(other_node, node) =
                std::max(node_assoc.min_assoc(other_node, node), pvalue);

        if (new_max > alpha) {
            return true;
        } else if (new_max < node_assoc.maxmin_assoc(node)) {
            node_assoc.maxmin_assoc(node) = new_max;
            node_assoc.maxmin_index(node) = other_node;
        }

        return false;
    }

    void univariate_cpcs_all_variables(const IndependenceTest& test,
                                       const PartiallyDirectedGraph& g,
                                       double alpha,
                                       std::vector<std::unordered_set<int>>& cpcs,
                                       std::vector<std::unordered_set<int>>& to_be_checked,
                                       CPCAssoc& cpc_assoc,
                                       util::BaseProgressBar& progress) {
        auto nnodes = g.num_nodes();
        progress.set_text("MMPC Forward: sepset order 1");
        progress.set_max_progress(nnodes);
        progress.set_progress(0);
        
        for (int i = 0; i < nnodes; ++i) {
            if (cpcs[i].size() == 1) {
                int cpc_variable = *cpcs[i].begin();
                const auto& i_name = g.name(i);
                const auto& cpc_name = g.name(cpc_variable);
                for (auto it = to_be_checked[i].begin(), end = to_be_checked[i].end(); it != end;) {
                    auto p = *it;
                    bool repeated_test = cpcs[p].size() == 1 && cpc_variable == *cpcs[p].begin() 
                                         && to_be_checked[p].count(i) > 0;

                    if (!repeated_test || i < p) {
                        const auto& p_name = g.name(p);
                        double pvalue = test.pvalue(i_name, p_name, cpc_name);

                        if (update_univariate_assoc(g, pvalue, i, p, alpha, cpc_assoc))
                            it = to_be_checked[i].erase(it);
                        else
                            ++it;

                        if (repeated_test && update_univariate_assoc(g, pvalue, p, i, alpha, cpc_assoc))
                            to_be_checked[p].erase(i);
                    } else {
                        ++it;
                    }
                }
            }

            progress.tick();
        }
    }

    bool update_univariate_assoc(const ConditionalPartiallyDirectedGraph& g,
                                double pvalue,
                                int node,
                                int other_node,
                                double alpha,
                                CPCAssoc& node_assoc,
                                CPCAssoc& interface_assoc) {
        
        if (g.is_interface(node)) {
            auto node_ifcollapsed = g.interface_collapsed_from_index(node);
            auto other_collapsed = g.collapsed_from_index(other_node);
            double new_max = 
                interface_assoc.min_assoc(other_collapsed, node_ifcollapsed) =
                std::max(interface_assoc.min_assoc(other_collapsed, node_ifcollapsed), pvalue);

            if (new_max > alpha) {
                return true;
            } else if (new_max < interface_assoc.maxmin_assoc(node_ifcollapsed)) {
                interface_assoc.maxmin_assoc(node_ifcollapsed) = new_max;
                interface_assoc.maxmin_index(node_ifcollapsed) = other_node;
            }
        } else {
            auto node_collapsed = g.collapsed_from_index(node);
            auto other_jc = g.joint_collapsed_from_index(other_node);
            double new_max = 
                node_assoc.min_assoc(other_jc, node_collapsed) =
                std::max(node_assoc.min_assoc(other_jc, node_collapsed), pvalue);

            if (new_max > alpha) {
                return true;
            } else if (new_max < node_assoc.maxmin_assoc(node_collapsed)) {
                node_assoc.maxmin_assoc(node_collapsed) = new_max;
                node_assoc.maxmin_index(node_collapsed) = other_node;
            }
        }

        return false;
    }

    void univariate_cpcs_all_variables(const IndependenceTest& test,
                                       const ConditionalPartiallyDirectedGraph& g,
                                       double alpha,
                                       std::vector<std::unordered_set<int>>& cpcs,
                                       std::vector<std::unordered_set<int>>& to_be_checked,
                                       CPCAssoc& node_assoc,
                                       CPCAssoc& interface_assoc,
                                       util::BaseProgressBar& progress) {
        auto nnodes = g.num_total_nodes();
        progress.set_text("MMPC Forward: sepset order 1");
        progress.set_max_progress(nnodes);
        progress.set_progress(0);
        
        for (int i = 0; i < nnodes; ++i) {
            const auto& node = g.name(i);

            if (cpcs[i].size() == 1) {
                int cpc_variable = *cpcs[i].begin();
                const auto& cpc_name = g.name(cpc_variable);
                for (auto it = to_be_checked[i].begin(), end = to_be_checked[i].end(); it != end;) {
                    auto p = *it;
                    bool repeated_test = cpcs[p].size() == 1 && cpc_variable == *cpcs[p].begin()
                                         && to_be_checked[p].count(i) > 0;

                    if (!repeated_test || i < p) {
                        const auto& p_name = g.name(p);
                        double pvalue = test.pvalue(node, p_name, cpc_name);

                        if (update_univariate_assoc(g, pvalue, i, p, alpha, node_assoc, interface_assoc))
                            it = to_be_checked[i].erase(it);
                        else
                            ++it;

                        if (repeated_test && update_univariate_assoc(g, pvalue, p, i, alpha, node_assoc, interface_assoc)) {
                            to_be_checked[p].erase(i);
                        }
                    } else {
                        ++it;
                    }
                }
            }

            progress.tick();
        }
    }

    std::pair<std::vector<std::unordered_set<int>>,
              std::vector<std::unordered_set<int>>> generate_cpcs(const PartiallyDirectedGraph& g,
                                                                  const ArcSet& arc_whitelist,
                                                                  const EdgeSet& edge_blacklist,
                                                                  const EdgeSet& edge_whitelist) {
        std::vector<std::unordered_set<int>> cpcs(g.num_nodes());
        std::vector<std::unordered_set<int>> to_be_checked(g.num_nodes());

        // Add whitelisted CPCs
        for (const auto& edge : edge_whitelist) {
            cpcs[edge.first].insert(edge.second);
            cpcs[edge.second].insert(edge.first);
        }

        for (const auto& arc : arc_whitelist) {
            cpcs[arc.first].insert(arc.second);
            cpcs[arc.second].insert(arc.first);
        }

        // Generate to_be_checked indices for whitelisted CPCs
        for (int i = 0; i < g.num_nodes(); ++i) {
            for (int j = 0; j < g.num_nodes(); ++j) {
                if (j != i && edge_blacklist.count({i,j}) == 0 && cpcs[i].count(j) == 0) {
                    to_be_checked[i].insert(j);
                }
            }
        }

        return std::make_pair(cpcs, to_be_checked);
    }

    std::pair<std::vector<std::unordered_set<int>>,
              std::vector<std::unordered_set<int>>> generate_cpcs(const ConditionalPartiallyDirectedGraph& g,
                                                                  const ArcSet& arc_whitelist,
                                                                  const EdgeSet& edge_blacklist,
                                                                  const EdgeSet& edge_whitelist) {
        std::vector<std::unordered_set<int>> cpcs(g.num_total_nodes());
        std::vector<std::unordered_set<int>> to_be_checked(g.num_total_nodes());

        // Add whitelisted CPCs
        for (const auto& edge : edge_whitelist) {
            cpcs[edge.first].insert(edge.second);
            cpcs[edge.second].insert(edge.first);
        }

        for (const auto& arc : arc_whitelist) {
            cpcs[arc.first].insert(arc.second);
            cpcs[arc.second].insert(arc.first);
        }

        // Generate to_be_checked indices for nodes
        for (const auto& node : g.nodes()) {
            auto index = g.index(node);
            for (const auto& other : g.all_nodes()) {
                auto other_index = g.index(other);
                
                if (index != other_index && edge_blacklist.count({index, other_index}) == 0 && cpcs[index].count(other_index) == 0) {
                    to_be_checked[index].insert(other_index);
                }
            }
        }

        // Generate to_be_checked indices for interface_nodes
        for (const auto& inode : g.interface_nodes()) {
            auto iindex = g.index(inode);
            for (const auto& other : g.nodes()) {
                auto other_index = g.index(other);

                if (edge_blacklist.count({iindex, other_index}) == 0 && cpcs[iindex].count(other_index) == 0)
                    to_be_checked[iindex].insert(other_index);

            }
        }

        return std::make_pair(cpcs, to_be_checked);
    }

    // 
    // WARNING!: This method should be called with a Graph without removed nodes.
    // 
    std::vector<std::unordered_set<int>> mmpc_all_variables(const IndependenceTest& test,
                                                            const PartiallyDirectedGraph& g,
                                                            double alpha,
                                                            const ArcSet& arc_whitelist,
                                                            const EdgeSet& edge_blacklist,
                                                            const EdgeSet& edge_whitelist,
                                                            util::BaseProgressBar& progress) {

        auto [cpcs, to_be_checked] = generate_cpcs(g, arc_whitelist, edge_blacklist, edge_whitelist);
    
        auto cpc_assoc = CPCAssoc {
            .min_assoc = MatrixXd::Zero(g.num_nodes(), g.num_nodes()),
            .maxmin_assoc = VectorXd::Constant(g.num_nodes(), std::numeric_limits<double>::infinity()),
            .maxmin_index = VectorXi::Constant(g.num_nodes(), MMPC_FORWARD_PHASE_STOP)
        };

        marginal_cpcs_all_variables(test, g, alpha, cpcs, to_be_checked, 
                                    edge_blacklist, cpc_assoc, progress);

        bool all_finished = true;
        for (int i = 0; i < g.num_nodes(); ++i) {
            if (cpc_assoc.maxmin_index(i) != MMPC_FORWARD_PHASE_STOP) {
                all_finished = false;
                cpcs[i].insert(cpc_assoc.maxmin_index(i));
                to_be_checked[i].erase(cpc_assoc.maxmin_index(i));
            }

            if (cpcs[i].size() == 1) {
                cpc_assoc.maxmin_assoc(i) = std::numeric_limits<double>::infinity();
                cpc_assoc.maxmin_index(i) = MMPC_FORWARD_PHASE_STOP;
            }
        }

        if (!all_finished) {
            univariate_cpcs_all_variables(test, g, alpha, cpcs, to_be_checked, cpc_assoc, progress);

            for (int i = 0; i < g.num_nodes(); ++i) {
                auto col_min_assoc = cpc_assoc.min_assoc.col(i);
                // The cpc is whitelisted.
                if (cpcs[i].size() > 1) {
                    mmpc_forward_phase(test, g, i, alpha, cpcs[i], to_be_checked[i], 
                                    col_min_assoc, MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC, progress);
                } else if (cpc_assoc.maxmin_index(i) != MMPC_FORWARD_PHASE_STOP) {
                    cpcs[i].insert(cpc_assoc.maxmin_index(i));
                    to_be_checked[i].erase(cpc_assoc.maxmin_index(i));
                    mmpc_forward_phase(test, g, i, alpha, cpcs[i], to_be_checked[i], col_min_assoc, 
                                        cpc_assoc.maxmin_index(i), progress);
                }

                mmpc_backward_phase(test, g, i, alpha, cpcs[i], arc_whitelist, edge_whitelist, progress);
            }
        }

        return cpcs;
    }

    std::vector<std::unordered_set<int>> mmpc_all_variables(const IndependenceTest& test,
                                                            const ConditionalPartiallyDirectedGraph& g,
                                                            double alpha,
                                                            const ArcSet& arc_whitelist,
                                                            const EdgeSet& edge_blacklist,
                                                            const EdgeSet& edge_whitelist,
                                                            util::BaseProgressBar& progress) {
        auto [cpcs, to_be_checked] = generate_cpcs(g, arc_whitelist, edge_blacklist, edge_whitelist);

        auto node_assoc = CPCAssoc {
            .min_assoc = MatrixXd::Zero(g.num_total_nodes(), g.num_nodes()),
            .maxmin_assoc = VectorXd::Constant(g.num_nodes(), std::numeric_limits<double>::infinity()),
            .maxmin_index = VectorXi::Constant(g.num_nodes(), MMPC_FORWARD_PHASE_STOP)
        };

        auto interface_assoc = CPCAssoc {
            .min_assoc = MatrixXd::Zero(g.num_nodes(), g.num_interface_nodes()),
            .maxmin_assoc = VectorXd::Constant(g.num_interface_nodes(), std::numeric_limits<double>::infinity()),
            .maxmin_index = VectorXi::Constant(g.num_interface_nodes(), MMPC_FORWARD_PHASE_STOP)
        };

        marginal_cpcs_all_variables(test, g, alpha, cpcs, to_be_checked, 
                            edge_blacklist, node_assoc, interface_assoc, progress);

        bool all_finished = true;

        for (int i = 0; i < g.num_nodes(); ++i) {
            auto index = g.index_from_collapsed(i);
            if (node_assoc.maxmin_index(i) != MMPC_FORWARD_PHASE_STOP) {
                all_finished = false;
                cpcs[index].insert(node_assoc.maxmin_index(i));
                to_be_checked[index].erase(node_assoc.maxmin_index(i));
            }

            if (cpcs[index].size() == 1) {
                node_assoc.maxmin_assoc(i) = std::numeric_limits<double>::infinity();
                node_assoc.maxmin_index(i) = MMPC_FORWARD_PHASE_STOP;
            }
        }

        for (int i = 0; i < g.num_interface_nodes(); ++i) {
            auto index = g.index_from_interface_collapsed(i);
            if (interface_assoc.maxmin_index(i) != MMPC_FORWARD_PHASE_STOP) {
                all_finished = false;
                cpcs[index].insert(interface_assoc.maxmin_index(i));
                to_be_checked[index].erase(interface_assoc.maxmin_index(i));
            }

            if (cpcs[index].size() == 1) {
                interface_assoc.maxmin_assoc(i) = std::numeric_limits<double>::infinity();
                interface_assoc.maxmin_index(i) = MMPC_FORWARD_PHASE_STOP;
            }
        }

        if (!all_finished) {
            univariate_cpcs_all_variables(test, g, alpha, cpcs, to_be_checked, 
                                          node_assoc, interface_assoc, progress);

            for (int i = 0; i < g.num_nodes(); ++i) {
                auto col_min_assoc = node_assoc.min_assoc.col(i);
                auto index = g.index_from_collapsed(i);

                // if (cpcs[index].size() > 1) {
                //     mmpc_forward_phase_node(test, g, i, alpha, cpcs[index], to_be_checked[index], 
                //                         col_min_assoc, MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC, progress);
                // } else if (node_assoc.maxmin_index(index) != MMPC_FORWARD_PHASE_STOP) {
                //     cpcs[index].insert(node_assoc.maxmin_index(index));
                //     to_be_checked[index].erase(node_assoc.maxmin_index(index));

                //     mmpc_forward_phase_node(test, g, i, alpha, cpcs[index], to_be_checked[index], col_min_assoc, 
                //                         node_assoc.maxmin_index(index), progress);

                // }

                // mmpc_backward_phase(test, g, i, alpha, cpcs[index], arc_whitelist, edge_whitelist, progress);
            }

        }

        return cpcs;
    }

    PartiallyDirectedGraph MMPC::estimate(const IndependenceTest& test,
                                          const std::vector<std::string>& nodes,
                                          const ArcStringVector& varc_blacklist,
                                          const ArcStringVector& varc_whitelist,
                                          const EdgeStringVector& vedge_blacklist,
                                          const EdgeStringVector& vedge_whitelist,
                                          double alpha,
                                          double ambiguous_threshold,
                                          bool allow_bidirected,
                                          int verbose) const {


        PartiallyDirectedGraph skeleton;
        if (nodes.empty())
            skeleton = PartiallyDirectedGraph(test.variable_names());
        else {
            if (!test.has_variables(nodes))
                throw std::invalid_argument("IndependenceTest do not contain all the variables in nodes list.");
            skeleton = PartiallyDirectedGraph(nodes);
        }

        auto restrictions = util::validate_restrictions(skeleton, 
                                                        varc_blacklist,
                                                        varc_whitelist,
                                                        vedge_blacklist,
                                                        vedge_whitelist);

        for (const auto& a : restrictions.arc_whitelist) {
            skeleton.add_arc(a.first, a.second);
        }

        indicators::show_console_cursor(false);
        auto progress = util::progress_bar(verbose);

        auto cpcs = mmpc_all_variables(test, skeleton, alpha, restrictions.arc_whitelist, 
                                        restrictions.edge_blacklist, restrictions.edge_whitelist, *progress);

        for (auto i = 0; i < skeleton.num_nodes(); ++i) {
            for (auto p : cpcs[i]) {
                if (i < p && cpcs[p].count(i) > 0 && !skeleton.has_arc(i, p) && !skeleton.has_arc(p, i)) {
                    skeleton.add_edge(i, p);
                }
            }
        }

        direct_arc_blacklist(skeleton, restrictions.arc_blacklist);
        direct_unshielded_triples(skeleton, test, restrictions.arc_blacklist, restrictions.arc_whitelist, 
                                  alpha, std::nullopt, true, ambiguous_threshold, allow_bidirected, *progress);

        progress->set_max_progress(3);
        progress->set_text("Applying Meek rules");

        bool changed = true;
        while(changed) {
            changed = false;
            progress->set_progress(0);

            changed |= MeekRules::rule1(skeleton);
            progress->tick();
            changed |= MeekRules::rule2(skeleton);
            progress->tick();
            changed |= MeekRules::rule3(skeleton);
            progress->tick();
        }

        progress->mark_as_completed("Finished MMPC!");

        indicators::show_console_cursor(true);
        return skeleton;
    }



    ConditionalPartiallyDirectedGraph MMPC::estimate_conditional(const IndependenceTest& test,
                                                                 const std::vector<std::string>& nodes,
                                                                 const std::vector<std::string>& interface_nodes,
                                                                 const ArcStringVector& varc_blacklist, 
                                                                 const ArcStringVector& varc_whitelist,
                                                                 const EdgeStringVector& vedge_blacklist,
                                                                 const EdgeStringVector& vedge_whitelist,
                                                                 double alpha,
                                                                 double ambiguous_threshold,
                                                                 bool allow_bidirected,
                                                                 int verbose) const {

        if (nodes.empty())
            throw std::invalid_argument("Node list cannot be empty to train a Conditional graph.");
        if (interface_nodes.empty())
            return MMPC::estimate(test, nodes, varc_blacklist, varc_whitelist, vedge_blacklist, vedge_whitelist, 
                                    alpha, ambiguous_threshold, allow_bidirected, verbose).conditional_graph();
        
        if (!test.has_variables(nodes) || !test.has_variables(interface_nodes))
            throw std::invalid_argument("IndependenceTest do not contain all the variables in nodes/interface_nodes lists.");

        ConditionalPartiallyDirectedGraph skeleton(nodes, interface_nodes);

        auto restrictions = util::validate_restrictions(skeleton, 
                                                        varc_blacklist,
                                                        varc_whitelist,
                                                        vedge_blacklist,
                                                        vedge_whitelist);

        for (const auto& a : restrictions.arc_whitelist) {
            skeleton.add_arc(a.first, a.second);
        }

        auto progress = util::progress_bar(verbose);

        // auto cpcs = mmpc_all_variables(test, skeleton, alpha, restrictions.arc_whitelist, 
        //                             restrictions.edge_blacklist, restrictions.edge_whitelist, *progress);



        // std::unordered_set<std::string> set_nodes {nodes.begin(), nodes.end()};
        // std::unordered_set<std::string> set_interface_nodes {interface_nodes.begin(), interface_nodes.end()};

        // auto restrictions = util::validate_restrictions(skeleton, 
        //                                                 varc_blacklist,
        //                                                 varc_whitelist,
        //                                                 vedge_blacklist,
        //                                                 vedge_whitelist);

        // for (const auto& a : restrictions.arc_whitelist) {
        //     skeleton.add_arc(a.first, a.second);
        // }
                            
    }
}