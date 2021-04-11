// For std::iota
#include <numeric>
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

enum MMPC_Progress { MMPC_FORWARD_PHASE_STOP = -1, MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC = -2 };

struct CPCAssoc {
    MatrixXd min_assoc;
    VectorXd maxmin_assoc;
    VectorXi maxmin_index;
};

template <typename BN>
class BNCPCAssoc;

template <typename BN>
class BNCPCAssocCol {
public:
    BNCPCAssocCol(BNCPCAssoc<BN>& self, int col) : m_self(self), m_col(col) {}

    void fill(double v) { m_self.fill_col(m_col, v); }

    void reset_maxmin() {
        maxmin_assoc() = m_self.alpha();
        maxmin_index() = MMPC_FORWARD_PHASE_STOP;
    }

    double& min_assoc(int index) { return m_self.min_assoc(index, m_col); }

    double min_assoc(int index) const { return m_self.min_assoc(index, m_col); }

    double& maxmin_assoc() { return m_self.maxmin_assoc(m_col); }

    double maxmin_assoc() const { return m_self.maxmin_assoc(m_col); }

    int& maxmin_index() { return m_self.maxmin_index(m_col); }

    int maxmin_index() const { return m_self.maxmin_index(m_col); }

    void initialize_assoc(int index, double pvalue) {
        min_assoc(index) = pvalue;

        if (pvalue < maxmin_assoc()) {
            maxmin_assoc() = pvalue;
            maxmin_index() = index;
        }
    }

    void update_assoc(int index, double pvalue) {
        double new_max = min_assoc(index) = std::max(min_assoc(index), pvalue);

        if (new_max < maxmin_assoc()) {
            maxmin_assoc() = new_max;
            maxmin_index() = index;
        }
    }

private:
    BNCPCAssoc<BN>& m_self;
    int m_col;
};

template <typename BN>
BNCPCAssocCol(BNCPCAssoc<BN>&, int) -> BNCPCAssocCol<BN>;

template <>
class BNCPCAssocCol<VectorXd> {
public:
    BNCPCAssocCol(VectorXd& col, double alpha)
        : m_col(col), m_maxmin_assoc(alpha), m_maxmin_index(MMPC_FORWARD_PHASE_STOP), m_alpha(alpha) {}

    void fill(double v) { m_col.fill(v); }

    void reset_maxmin() {
        m_maxmin_assoc = m_alpha;
        m_maxmin_index = MMPC_FORWARD_PHASE_STOP;
    }

    double& min_assoc(int index) { return m_col(index); }

    double min_assoc(int index) const { return m_col(index); }

    double& maxmin_assoc() { return m_maxmin_assoc; }

    double maxmin_assoc() const { return m_maxmin_assoc; }

    int& maxmin_index() { return m_maxmin_index; }

    int maxmin_index() const { return m_maxmin_index; }

    void initialize_assoc(int index, double pvalue) {
        m_col(index) = pvalue;

        if (pvalue < m_maxmin_assoc) {
            m_maxmin_assoc = pvalue;
            m_maxmin_index = index;
        }
    }

    void update_assoc(int index, double pvalue) {
        double new_max = min_assoc(index) = std::max(min_assoc(index), pvalue);

        if (new_max < m_maxmin_assoc) {
            m_maxmin_assoc = new_max;
            m_maxmin_index = index;
        }
    }

private:
    VectorXd& m_col;
    double m_maxmin_assoc;
    int m_maxmin_index;
    double m_alpha;
};

template <>
class BNCPCAssoc<PartiallyDirectedGraph> {
public:
    BNCPCAssoc(const PartiallyDirectedGraph& g, double alpha) : m_graph(g), m_assoc(), m_alpha(alpha) {
        m_assoc = CPCAssoc{/*.min_assoc = */ MatrixXd::Zero(g.num_nodes(), g.num_nodes()),
                           /*.maxmin_assoc = */ VectorXd::Constant(g.num_nodes(), m_alpha),
                           /*.maxmin_index = */ VectorXi::Constant(g.num_nodes(), MMPC_FORWARD_PHASE_STOP)};
    }

    const PartiallyDirectedGraph& graph() { return m_graph; }

    CPCAssoc& raw_assoc() { return m_assoc; }

    double alpha() { return m_alpha; }

    void reset_maxmin(int index) {
        maxmin_assoc(index) = m_alpha;
        maxmin_index(index) = MMPC_FORWARD_PHASE_STOP;
    }

    void fill_col(int col_index, double v) { m_assoc.min_assoc.col(col_index).fill(v); }

    double& min_assoc(int row_index, int col_index) { return m_assoc.min_assoc(row_index, col_index); }

    double min_assoc(int row_index, int col_index) const { return m_assoc.min_assoc(row_index, col_index); }

    BNCPCAssocCol<PartiallyDirectedGraph> min_assoc_col(int col_index) { return BNCPCAssocCol(*this, col_index); }

    double& maxmin_assoc(int index) { return m_assoc.maxmin_assoc(index); }

    double maxmin_assoc(int index) const { return m_assoc.maxmin_assoc(index); }

    int& maxmin_index(int index) { return m_assoc.maxmin_index(index); }

    int maxmin_index(int index) const { return m_assoc.maxmin_index(index); }

    void initialize_assoc(int row_index, int col_index, double pvalue) {
        min_assoc(row_index, col_index) = pvalue;
        if (pvalue < m_assoc.maxmin_assoc(col_index)) {
            maxmin_assoc(col_index) = pvalue;
            maxmin_index(col_index) = row_index;
        }
    }

    void update_assoc(int row_index, int col_index, double pvalue) {
        double new_max = min_assoc(row_index, col_index) = std::max(min_assoc(row_index, col_index), pvalue);
        if (new_max < m_assoc.maxmin_assoc(col_index)) {
            maxmin_assoc(col_index) = new_max;
            maxmin_index(col_index) = row_index;
        }
    }

private:
    const PartiallyDirectedGraph& m_graph;
    CPCAssoc m_assoc;
    double m_alpha;
};

template <>
class BNCPCAssoc<ConditionalPartiallyDirectedGraph> {
public:
    BNCPCAssoc(const ConditionalPartiallyDirectedGraph& g, double alpha)
        : m_graph(g), m_assoc(), m_interface_assoc(), m_alpha(alpha) {
        m_assoc = CPCAssoc{/*.min_assoc = */ MatrixXd::Zero(g.num_joint_nodes(), g.num_nodes()),
                           /*.maxmin_assoc = */ VectorXd::Constant(g.num_nodes(), m_alpha),
                           /*.maxmin_index = */ VectorXi::Constant(g.num_nodes(), MMPC_FORWARD_PHASE_STOP)};

        m_interface_assoc =
            CPCAssoc{/*.min_assoc = */ MatrixXd::Zero(g.num_nodes(), g.num_interface_nodes()),
                     /*.maxmin_assoc = */ VectorXd::Constant(g.num_interface_nodes(), m_alpha),
                     /*.maxmin_index = */ VectorXi::Constant(g.num_interface_nodes(), MMPC_FORWARD_PHASE_STOP)};
    }

    const ConditionalPartiallyDirectedGraph& graph() { return m_graph; }

    CPCAssoc& raw_assoc() { return m_assoc; }

    CPCAssoc& raw_interface_assoc() { return m_interface_assoc; }

    void reset_maxmin(int index) {
        maxmin_assoc(index) = m_alpha;
        maxmin_index(index) = MMPC_FORWARD_PHASE_STOP;
    }

    double alpha() { return m_alpha; }

    void fill_col(int col_index, double v) {
        if (m_graph.is_interface(col_index)) {
            m_interface_assoc.min_assoc.col(m_graph.interface_collapsed_from_index(col_index)).fill(v);
        } else {
            m_assoc.min_assoc.col(m_graph.collapsed_from_index(col_index)).fill(v);
        }
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

    BNCPCAssocCol<ConditionalPartiallyDirectedGraph> min_assoc_col(int col_index) {
        return BNCPCAssocCol(*this, col_index);
    }

    double& maxmin_assoc_node(int index) { return m_assoc.maxmin_assoc(m_graph.collapsed_from_index(index)); }

    double maxmin_assoc_node(int index) const { return m_assoc.maxmin_assoc(m_graph.collapsed_from_index(index)); }

    double& maxmin_assoc_interface(int index) {
        return m_interface_assoc.maxmin_assoc(m_graph.interface_collapsed_from_index(index));
    }

    double maxmin_assoc_interface(int index) const {
        return m_interface_assoc.maxmin_assoc(m_graph.interface_collapsed_from_index(index));
    }

    double& maxmin_assoc(int index) {
        if (m_graph.is_interface(index))
            return maxmin_assoc_interface(index);
        else
            return maxmin_assoc_node(index);
    }

    double maxmin_assoc(int index) const {
        if (m_graph.is_interface(index))
            return maxmin_assoc_interface(index);
        else
            return maxmin_assoc_node(index);
    }

    int& maxmin_index_node(int index) { return m_assoc.maxmin_index(m_graph.collapsed_from_index(index)); }

    int maxmin_index_node(int index) const { return m_assoc.maxmin_index(m_graph.collapsed_from_index(index)); }

    int& maxmin_index_interface(int index) {
        return m_interface_assoc.maxmin_index(m_graph.interface_collapsed_from_index(index));
    }

    int maxmin_index_interface(int index) const {
        return m_interface_assoc.maxmin_index(m_graph.interface_collapsed_from_index(index));
    }

    int& maxmin_index(int index) {
        if (m_graph.is_interface(index))
            return maxmin_index_interface(index);
        else
            return maxmin_index_node(index);
    }

    int maxmin_index(int index) const {
        if (m_graph.is_interface(index))
            return maxmin_index_interface(index);
        else
            return maxmin_index_node(index);
    }

    void initialize_assoc(int row_index, int col_index, double pvalue) {
        if (m_graph.is_interface(col_index)) {
            min_assoc_interface(row_index, col_index) = pvalue;
            if (pvalue < maxmin_assoc_interface(col_index)) {
                maxmin_assoc_interface(col_index) = pvalue;
                maxmin_index_interface(col_index) = row_index;
            }

        } else {
            min_assoc_node(row_index, col_index) = pvalue;
            if (pvalue < maxmin_assoc_node(col_index)) {
                maxmin_assoc_node(col_index) = pvalue;
                maxmin_index_node(col_index) = row_index;
            }
        }
    }

    void update_assoc(int row_index, int col_index, double pvalue) {
        if (m_graph.is_interface(col_index)) {
            double new_max = min_assoc_interface(row_index, col_index) =
                std::max(min_assoc_interface(row_index, col_index), pvalue);
            if (new_max < maxmin_assoc_interface(col_index)) {
                maxmin_assoc_interface(col_index) = new_max;
                maxmin_index_interface(col_index) = row_index;
            }
        } else {
            double new_max = min_assoc_node(row_index, col_index) =
                std::max(min_assoc_node(row_index, col_index), pvalue);
            if (new_max < maxmin_assoc_node(col_index)) {
                maxmin_assoc_node(col_index) = new_max;
                maxmin_index_node(col_index) = row_index;
            }
        }
    }

private:
    const ConditionalPartiallyDirectedGraph& m_graph;
    CPCAssoc m_assoc;
    CPCAssoc m_interface_assoc;
    double m_alpha;
};

template <typename G>
BNCPCAssoc(const G&, double) -> BNCPCAssoc<G>;

template <typename G, typename ColAssoc>
void recompute_assoc(const IndependenceTest& test,
                     const G& g,
                     int variable,
                     const std::unordered_set<int>& cpc,
                     std::unordered_set<int>& to_be_checked,
                     ColAssoc& assoc,
                     util::BaseProgressBar& progress) {
    const auto& variable_name = g.name(variable);
    progress.set_text("MMPC Forward: sepset order " + std::to_string(cpc.size()) + " for " + variable_name);
    progress.set_max_progress(to_be_checked.size());
    progress.set_progress(0);

    std::vector<std::string> cpc_vec;
    cpc_vec.reserve(cpc.size());
    for (auto c : cpc) {
        cpc_vec.push_back(g.name(c));
    }

    assoc.reset_maxmin();

    for (auto it = to_be_checked.begin(); it != to_be_checked.end();) {
        double pvalue = test.pvalue(variable_name, g.name(*it), cpc_vec);
        assoc.initialize_assoc(*it, pvalue);
        progress.tick();
    }
}

template <typename G, typename ColAssoc>
void update_min_assoc(const IndependenceTest& test,
                      const G& g,
                      int variable,
                      const std::unordered_set<int>& to_be_checked,
                      const std::unordered_set<int>& cpc,
                      ColAssoc& assoc,
                      int last_added_cpc,
                      util::BaseProgressBar& progress) {
    const auto& variable_name = g.name(variable);

    assoc.reset_maxmin();

    if (cpc.empty()) {
        progress.set_text("MMPC Forward: no sepset for " + variable_name);
        progress.set_max_progress(to_be_checked.size());
        progress.set_progress(0);

        for (auto v : to_be_checked) {
            double pvalue = test.pvalue(variable_name, g.name(v));
            assoc.initialize_assoc(v, pvalue);
            progress.tick();
        }
    } else if (cpc.size() == 1) {
        progress.set_text("MMPC Forward: sepset order 1 for " + variable_name);
        progress.set_max_progress(to_be_checked.size());
        progress.set_progress(0);

        const auto& last_added_name = g.name(last_added_cpc);
        for (auto v : to_be_checked) {
            double pvalue = test.pvalue(variable_name, g.name(v), last_added_name);
            assoc.update_assoc(v, pvalue);
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
            assoc.update_assoc(v, pvalue);

            pvalue = test.pvalue(variable_name, v_name, cond);
            assoc.update_assoc(v, pvalue);

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
            comb = AllSubsets(old_cpc, std::move(fixed), 3, cpc.size() - 1);
        }

        for (auto v : to_be_checked) {
            const auto& v_name = g.name(v);
            // Conditioning in just the last variable added.
            double pvalue = test.pvalue(variable_name, v_name, last_added_name);
            assoc.update_assoc(v, pvalue);

            // Conditioning in the last variable and another variable added.
            for (const auto& pc : old_cpc) {
                cond[0] = pc;
                pvalue = test.pvalue(variable_name, v_name, cond);
                assoc.update_assoc(v, pvalue);
            }

            if (cpc.size() > 3) {
                for (const auto& subset : comb) {
                    pvalue = test.pvalue(variable_name, v_name, subset);
                    assoc.update_assoc(v, pvalue);
                }
            }

            // Conditioning in all the variables.
            old_cpc.push_back(last_added_name);
            pvalue = test.pvalue(variable_name, v_name, old_cpc);
            assoc.update_assoc(v, pvalue);
            old_cpc.pop_back();
        }

        progress.tick();
    }
}

template <typename ColAssoc>
void update_to_be_checked(const ColAssoc& assoc, std::unordered_set<int>& to_be_checked, double alpha) {
    for (auto it = to_be_checked.begin(), end = to_be_checked.end(); it != end;) {
        if (assoc.min_assoc(*it) > alpha) {
            it = to_be_checked.erase(it);
        } else {
            ++it;
        }
    }
}

template <typename G, typename ColAssoc>
void mmpc_forward_phase(const IndependenceTest& test,
                        const G& g,
                        int variable,
                        double alpha,
                        std::unordered_set<int>& cpc,
                        std::unordered_set<int>& to_be_checked,
                        ColAssoc& assoc,
                        int last_added,
                        util::BaseProgressBar& progress) {
    bool changed_cpc = true;

    if (cpc.empty()) {
        assoc.fill(0);
    } else if (last_added == MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC) {
        // The CPC is not empty because of whitelists, so we compute the association of the selected CPC.
        recompute_assoc(test, g, variable, cpc, to_be_checked, assoc, progress);

        int to_add = assoc.maxmin_index();

        if (to_add != MMPC_FORWARD_PHASE_STOP) {
            cpc.insert(to_add);
            to_be_checked.erase(to_add);
            last_added = to_add;
            update_to_be_checked(assoc, to_be_checked, alpha);
        } else {
            changed_cpc = false;
        }
    }

    while (changed_cpc && !to_be_checked.empty()) {
        update_min_assoc(test, g, variable, to_be_checked, cpc, assoc, last_added, progress);
        // int to_add = find_maxmin_assoc(assoc, to_be_checked, alpha);
        int to_add = assoc.maxmin_index();

        if (to_add != MMPC_FORWARD_PHASE_STOP) {
            cpc.insert(to_add);
            to_be_checked.erase(to_add);
            last_added = to_add;
            update_to_be_checked(assoc, to_be_checked, alpha);
        } else {
            changed_cpc = false;
        }
    }
}

bool is_whitelisted_pc(int variable, int candidate_pc, const ArcSet& arc_whitelist, const EdgeSet& edge_whitelist) {
    return edge_whitelist.count({variable, candidate_pc}) > 0 || arc_whitelist.count({variable, candidate_pc}) > 0 ||
           arc_whitelist.count({candidate_pc, variable}) > 0;
}

template <typename G>
void mmpc_backward_phase(const IndependenceTest& test,
                         const G& g,
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
            for (auto it_other = subset_variables.begin(), end_other = subset_variables.end(); it_other != end_other;
                 ++it_other) {
                if (test.pvalue(variable_name, it_name, *it_other) > alpha) {
                    it = cpc.erase(it);
                    progress.set_max_progress(cpc.size());
                    found_sepset = true;
                    break;
                }
            }

            if (!found_sepset && subset_variables.size() > 2) {
                // Independence sepset length 2 to subset size - 1.
                AllSubsets comb(subset_variables, 2, subset_variables.size() - 1);

                for (const auto& s : comb) {
                    if (test.pvalue(variable_name, it_name, s) > alpha) {
                        it = cpc.erase(it);
                        progress.set_max_progress(cpc.size());
                        found_sepset = true;
                        break;
                    }
                }
            }

            // Independence sepset length of subset size.
            if (!found_sepset && subset_variables.size() > 1 &&
                test.pvalue(variable_name, it_name, subset_variables) > alpha) {
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
    BNCPCAssocCol<VectorXd> assoc_col(min_assoc, alpha);

    int last_added = 0;
    if (!cpc.empty()) last_added = MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC;

    mmpc_forward_phase(test, g, variable, alpha, cpc, to_be_checked, assoc_col, last_added, progress);
    mmpc_backward_phase(test, g, variable, alpha, cpc, arc_whitelist, edge_whitelist, progress);
    return cpc;
}

template <typename G>
void marginal_cpcs_all_variables(const IndependenceTest& test,
                                 const G& g,
                                 double alpha,
                                 std::vector<std::unordered_set<int>>& cpcs,
                                 std::vector<std::unordered_set<int>>& to_be_checked,
                                 const EdgeSet& edge_blacklist,
                                 BNCPCAssoc<G>& assoc,
                                 util::BaseProgressBar& progress) {
    auto nnodes = g.num_nodes();

    progress.set_text("MMPC Forward: No sepset");
    progress.set_max_progress((nnodes * (nnodes - 1) / 2));
    progress.set_progress(0);

    for (int i = 0, i_end = nnodes - 1; i < i_end; ++i) {
        const auto& i_name = g.collapsed_name(i);
        auto i_index = g.index(i_name);
        for (int j = i + 1; j < nnodes; ++j) {
            const auto& j_name = g.collapsed_name(j);
            auto j_index = g.index(j_name);
            if ((cpcs[i_index].empty() || cpcs[j_index].empty()) && edge_blacklist.count({i_index, j_index}) == 0) {
                double pvalue = test.pvalue(i_name, j_name);
                if (pvalue < alpha) {
                    if (cpcs[i_index].empty()) {
                        assoc.initialize_assoc(j_index, i_index, pvalue);
                    }

                    if (cpcs[j_index].empty()) {
                        assoc.initialize_assoc(i_index, j_index, pvalue);
                    }
                } else {
                    to_be_checked[i_index].erase(j_index);
                    to_be_checked[j_index].erase(i_index);
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
                                 BNCPCAssoc<ConditionalPartiallyDirectedGraph>& assoc,
                                 util::BaseProgressBar& progress) {
    auto nnodes = g.num_nodes();
    auto inodes = g.num_interface_nodes();

    progress.set_text("MMPC Forward: No sepset");
    progress.set_max_progress(inodes * nnodes + (nnodes * (nnodes - 1) / 2));
    progress.set_progress(0);

    // Cache marginal between nodes
    marginal_cpcs_all_variables<ConditionalPartiallyDirectedGraph>(
        test, g, alpha, cpcs, to_be_checked, edge_blacklist, assoc, progress);

    // Cache between nodes and interface_nodes
    for (const auto& node : g.nodes()) {
        auto nindex = g.index(node);
        for (const auto& inode : g.interface_nodes()) {
            auto iindex = g.index(inode);

            if ((cpcs[nindex].empty() || cpcs[iindex].empty()) && edge_blacklist.count({nindex, iindex}) == 0) {
                double pvalue = test.pvalue(node, inode);
                if (pvalue < alpha) {
                    if (cpcs[nindex].empty()) {
                        assoc.initialize_assoc(iindex, nindex, pvalue);
                    }

                    if (cpcs[iindex].empty()) {
                        assoc.initialize_assoc(nindex, iindex, pvalue);
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

template <typename G>
void univariate_cpcs_all_variables(const IndependenceTest& test,
                                   const G& g,
                                   int num_total_nodes,
                                   double alpha,
                                   std::vector<std::unordered_set<int>>& cpcs,
                                   std::vector<std::unordered_set<int>>& to_be_checked,
                                   BNCPCAssoc<G>& assoc,
                                   util::BaseProgressBar& progress) {
    progress.set_text("MMPC Forward: sepset order 1");
    progress.set_max_progress(num_total_nodes);
    progress.set_progress(0);

    for (int i = 0; i < num_total_nodes; ++i) {
        if (cpcs[i].size() == 1) {
            int cpc_variable = *cpcs[i].begin();
            const auto& i_name = g.name(i);
            const auto& cpc_name = g.name(cpc_variable);
            for (auto it = to_be_checked[i].begin(), end = to_be_checked[i].end(); it != end;) {
                auto p = *it;
                bool repeated_test =
                    cpcs[p].size() == 1 && cpc_variable == *cpcs[p].begin() && to_be_checked[p].count(i) > 0;

                if (!repeated_test || i < p) {
                    const auto& p_name = g.name(p);
                    double pvalue = test.pvalue(i_name, p_name, cpc_name);

                    assoc.update_assoc(p, i, pvalue);
                    if (assoc.min_assoc(p, i) > alpha)
                        it = to_be_checked[i].erase(it);
                    else
                        ++it;

                    if (repeated_test) {
                        assoc.update_assoc(i, p, pvalue);
                        if (assoc.min_assoc(i, p) > alpha) to_be_checked[p].erase(i);
                    }
                } else {
                    ++it;
                }
            }
        }

        progress.tick();
    }
}

std::pair<std::vector<std::unordered_set<int>>, std::vector<std::unordered_set<int>>> generate_cpcs(
    const PartiallyDirectedGraph& g,
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

    // Generate to_be_checked indices
    for (int i = 0, i_end = g.num_nodes() - 1; i < i_end; ++i) {
        const auto& x_name = g.collapsed_name(i);
        auto x_index = g.index(x_name);
        for (int j = i + 1; j < g.num_nodes(); ++j) {
            const auto& y_name = g.collapsed_name(j);
            auto y_index = g.index(y_name);

            if (edge_blacklist.count({x_index, y_index}) == 0) {
                if (cpcs[x_index].count(y_index) == 0) {
                    to_be_checked[x_index].insert(y_index);
                }

                if (cpcs[y_index].count(x_index) == 0) {
                    to_be_checked[y_index].insert(x_index);
                }
            }
        }
    }

    return std::make_pair(cpcs, to_be_checked);
}

std::pair<std::vector<std::unordered_set<int>>, std::vector<std::unordered_set<int>>> generate_cpcs(
    const ConditionalPartiallyDirectedGraph& g,
    const ArcSet& arc_whitelist,
    const EdgeSet& edge_blacklist,
    const EdgeSet& edge_whitelist) {
    std::vector<std::unordered_set<int>> cpcs(g.num_joint_nodes());
    std::vector<std::unordered_set<int>> to_be_checked(g.num_joint_nodes());

    // Add whitelisted CPCs
    for (const auto& edge : edge_whitelist) {
        cpcs[edge.first].insert(edge.second);
        cpcs[edge.second].insert(edge.first);
    }

    for (const auto& arc : arc_whitelist) {
        cpcs[arc.first].insert(arc.second);
        cpcs[arc.second].insert(arc.first);
    }

    // Generate to_be_checked indices
    for (const auto& node : g.nodes()) {
        auto index = g.index(node);
        for (const auto& other : g.joint_nodes()) {
            auto other_index = g.index(other);

            if (index != other_index && edge_blacklist.count({index, other_index}) == 0) {
                if (cpcs[index].count(other_index) == 0) to_be_checked[index].insert(other_index);
                if (cpcs[other_index].count(index) == 0) to_be_checked[other_index].insert(index);
            }
        }
    }

    return std::make_pair(cpcs, to_be_checked);
}

template <typename G>
std::vector<std::unordered_set<int>> mmpc_all_variables(const IndependenceTest& test,
                                                        const G& g,
                                                        int num_total_nodes,
                                                        double alpha,
                                                        const ArcSet& arc_whitelist,
                                                        const EdgeSet& edge_blacklist,
                                                        const EdgeSet& edge_whitelist,
                                                        util::BaseProgressBar& progress) {
    auto [cpcs, to_be_checked] = generate_cpcs(g, arc_whitelist, edge_blacklist, edge_whitelist);

    BNCPCAssoc assoc(g, alpha);

    marginal_cpcs_all_variables(test, g, alpha, cpcs, to_be_checked, edge_blacklist, assoc, progress);

    bool all_finished = true;
    for (int i = 0; i < num_total_nodes; ++i) {
        if (assoc.maxmin_index(i) != MMPC_FORWARD_PHASE_STOP) {
            all_finished = false;
            cpcs[i].insert(assoc.maxmin_index(i));
            to_be_checked[i].erase(assoc.maxmin_index(i));
        }

        if (cpcs[i].size() == 1) {
            assoc.reset_maxmin(i);
        }
    }

    if (!all_finished) {
        univariate_cpcs_all_variables(test, g, num_total_nodes, alpha, cpcs, to_be_checked, assoc, progress);

        for (int i = 0; i < num_total_nodes; ++i) {
            auto col_min_assoc = assoc.min_assoc_col(i);
            // The cpc is whitelisted.
            if (cpcs[i].size() > 1) {
                mmpc_forward_phase(test,
                                   g,
                                   i,
                                   alpha,
                                   cpcs[i],
                                   to_be_checked[i],
                                   col_min_assoc,
                                   MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC,
                                   progress);
            } else if (assoc.maxmin_index(i) != MMPC_FORWARD_PHASE_STOP) {
                cpcs[i].insert(assoc.maxmin_index(i));
                to_be_checked[i].erase(assoc.maxmin_index(i));
                mmpc_forward_phase(
                    test, g, i, alpha, cpcs[i], to_be_checked[i], col_min_assoc, assoc.maxmin_index(i), progress);
            }

            mmpc_backward_phase(test, g, i, alpha, cpcs[i], arc_whitelist, edge_whitelist, progress);
        }
    }

    return cpcs;
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
    return mmpc_all_variables(test, g, g.num_nodes(), alpha, arc_whitelist, edge_blacklist, edge_whitelist, progress);
}

//
// WARNING!: This method should be called with a Graph without removed nodes.
//
std::vector<std::unordered_set<int>> mmpc_all_variables(const IndependenceTest& test,
                                                        const ConditionalPartiallyDirectedGraph& g,
                                                        double alpha,
                                                        const ArcSet& arc_whitelist,
                                                        const EdgeSet& edge_blacklist,
                                                        const EdgeSet& edge_whitelist,
                                                        util::BaseProgressBar& progress) {
    return mmpc_all_variables(
        test, g, g.num_joint_nodes(), alpha, arc_whitelist, edge_blacklist, edge_whitelist, progress);
}

template <typename G>
void estimate(G& skeleton,
              const IndependenceTest& test,
              const ArcStringVector& varc_blacklist,
              const ArcStringVector& varc_whitelist,
              const EdgeStringVector& vedge_blacklist,
              const EdgeStringVector& vedge_whitelist,
              double alpha,
              double ambiguous_threshold,
              bool allow_bidirected,
              int verbose) {
    auto restrictions =
        util::validate_restrictions(skeleton, varc_blacklist, varc_whitelist, vedge_blacklist, vedge_whitelist);

    for (const auto& a : restrictions.arc_whitelist) {
        skeleton.add_arc(a.first, a.second);
    }

    auto progress = util::progress_bar(verbose);

    auto cpcs = mmpc_all_variables(test,
                                   skeleton,
                                   alpha,
                                   restrictions.arc_whitelist,
                                   restrictions.edge_blacklist,
                                   restrictions.edge_whitelist,
                                   *progress);

    for (auto i = 0; i < skeleton.num_nodes(); ++i) {
        for (auto p : cpcs[i]) {
            if (i < p && cpcs[p].count(i) > 0 && !skeleton.has_arc(i, p) && !skeleton.has_arc(p, i)) {
                if constexpr (graph::is_unconditional_graph_v<G>) {
                    skeleton.add_edge(i, p);
                } else if constexpr (graph::is_conditional_graph_v<G>) {
                    if (skeleton.is_interface(i))
                        skeleton.add_arc(i, p);
                    else if (skeleton.is_interface(p))
                        skeleton.add_arc(p, i);
                    else
                        skeleton.add_edge(i, p);
                } else {
                    static_assert(util::always_false<G>, "Wrong graph type");
                }
            }
        }
    }

    direct_arc_blacklist(skeleton, restrictions.arc_blacklist);
    direct_unshielded_triples(skeleton,
                              test,
                              restrictions.arc_blacklist,
                              restrictions.arc_whitelist,
                              alpha,
                              std::nullopt,
                              true,
                              ambiguous_threshold,
                              allow_bidirected,
                              *progress);

    progress->set_max_progress(3);
    progress->set_text("Applying Meek rules");

    bool changed = true;
    while (changed) {
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
    if (alpha <= 0 || alpha >= 1) throw std::invalid_argument("alpha must be a number between 0 and 1.");
    if (ambiguous_threshold < 0 || ambiguous_threshold > 1)
        throw std::invalid_argument("ambiguous_threshold must be a number between 0 and 1.");

    PartiallyDirectedGraph skeleton;
    if (nodes.empty())
        skeleton = PartiallyDirectedGraph(test.variable_names());
    else {
        if (!test.has_variables(nodes))
            throw std::invalid_argument("IndependenceTest do not contain all the variables in nodes list.");
        skeleton = PartiallyDirectedGraph(nodes);
    }

    learning::algorithms::estimate(skeleton,
                                   test,
                                   varc_blacklist,
                                   varc_whitelist,
                                   vedge_blacklist,
                                   vedge_whitelist,
                                   alpha,
                                   ambiguous_threshold,
                                   allow_bidirected,
                                   verbose);

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
    if (alpha <= 0 || alpha >= 1) throw std::invalid_argument("alpha must be a number between 0 and 1.");
    if (ambiguous_threshold < 0 || ambiguous_threshold > 1)
        throw std::invalid_argument("ambiguous_threshold must be a number between 0 and 1.");

    if (nodes.empty()) throw std::invalid_argument("Node list cannot be empty to train a Conditional graph.");
    if (interface_nodes.empty())
        return MMPC::estimate(test,
                              nodes,
                              varc_blacklist,
                              varc_whitelist,
                              vedge_blacklist,
                              vedge_whitelist,
                              alpha,
                              ambiguous_threshold,
                              allow_bidirected,
                              verbose)
            .conditional_graph();

    if (!test.has_variables(nodes) || !test.has_variables(interface_nodes))
        throw std::invalid_argument(
            "IndependenceTest do not contain all the variables in nodes/interface_nodes lists.");

    ConditionalPartiallyDirectedGraph skeleton(nodes, interface_nodes);

    learning::algorithms::estimate(skeleton,
                                   test,
                                   varc_blacklist,
                                   varc_whitelist,
                                   vedge_blacklist,
                                   vedge_whitelist,
                                   alpha,
                                   ambiguous_threshold,
                                   allow_bidirected,
                                   verbose);
    return skeleton;
}

}  // namespace learning::algorithms
