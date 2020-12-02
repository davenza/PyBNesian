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

    template<typename VectorType>
    void recompute_assoc(const IndependenceTest& test,
                         int variable,
                         const std::unordered_set<int>& cpc,
                         const std::unordered_set<int>& to_be_checked,
                         VectorType& min_assoc,
                         util::BaseProgressBar* progress) {

        std::vector<int> cpc_vec(cpc.begin(), cpc.end());
        progress->set_text("MMPC Forward: sepset order " + std::to_string(cpc_vec.size()) +  " for " + test.name(variable));
        progress->set_max_progress(to_be_checked.size());
        progress->set_progress(0);
        for (auto other : to_be_checked) {
            min_assoc(other) = test.pvalue(variable, other, cpc_vec.begin(), cpc_vec.end());
            progress->tick();
        }
    }

    template<typename VectorType>
    void update_min_assoc(const IndependenceTest& test, 
                          int variable,
                          const std::unordered_set<int>& to_be_checked,
                          const std::unordered_set<int>& cpc,
                          VectorType& min_assoc,
                          int last_added_cpc,
                          util::BaseProgressBar* progress) {
                
        if (cpc.empty()) {
            progress->set_text("MMPC Forward: no sepset for " + test.name(variable));
            progress->set_max_progress(to_be_checked.size());
            progress->set_progress(0);

            for (auto v : to_be_checked) {
                double pvalue = test.pvalue(variable, v);
                min_assoc(v) = pvalue;
                progress->tick();
            }
        } else if (cpc.size() == 1) {
            progress->set_text("MMPC Forward: sepset order 1 for " + test.name(variable));
            progress->set_max_progress(to_be_checked.size());
            progress->set_progress(0);

            for (auto v : to_be_checked) {
                double pvalue = test.pvalue(variable, v, last_added_cpc);
                min_assoc(v) = std::max(min_assoc(v), pvalue);
                progress->tick();
            }
        } else if (cpc.size() == 2) {
            std::vector<int> cond(2);
            cond[1] = last_added_cpc;

            progress->set_text("MMPC Forward: sepset order 2 for " + test.name(variable));
            progress->set_max_progress(to_be_checked.size());
            progress->set_progress(0);

            for (auto v : to_be_checked) {
                double pvalue = test.pvalue(variable, v, last_added_cpc);
                min_assoc(v) = std::max(min_assoc(v), pvalue);

                for (auto pc : cpc) {
                    if (pc == last_added_cpc)
                        continue;

                    cond[0] = pc;
                    pvalue = test.pvalue(variable, v, cond.begin(), cond.end());
                    min_assoc(v) = std::max(min_assoc(v), pvalue);
                }

                progress->tick();
            }
        } else {
            progress->set_text("MMPC Forward: sepset up to order " + std::to_string(cpc.size()) + " for " + test.name(variable));
            progress->set_max_progress(to_be_checked.size());
            progress->set_progress(0);

            std::vector<int> fixed = {last_added_cpc};
            std::vector<int> old_cpc {cpc.begin(), cpc.end()};
            util::swap_remove_v(old_cpc, last_added_cpc);
            
            std::vector<int> cond(2);
            cond[1] = last_added_cpc;

            // Conditioning in all the subsets of 2 to CPC.size()-1 size, including last variable added.
            AllSubsets<int> comb;
            if (cpc.size() > 3) {
                comb = AllSubsets(old_cpc, std::move(fixed), 3, cpc.size()-1);
            }

            for (auto v : to_be_checked) {
                // Conditioning in just the last variable added.
                double pvalue = test.pvalue(variable, v, last_added_cpc);
                min_assoc(v) = std::max(min_assoc(v), pvalue);

                // Conditioning in the last variable and another variable added.
                for (auto pc : old_cpc) {
                    cond[0] = pc;
                    pvalue = test.pvalue(variable, v, cond.begin(), cond.end());
                    min_assoc(v) = std::max(min_assoc(v), pvalue);
                }
                
                if (cpc.size() > 3) {
                    for (const auto& subset : comb) {
                        pvalue = test.pvalue(variable, v, subset.begin(), subset.end());
                        min_assoc(v) = std::max(min_assoc(v), pvalue);
                    }
                }

                // Conditioning in all the variables.
                old_cpc.push_back(last_added_cpc);
                pvalue = test.pvalue(variable, v, old_cpc.begin(), old_cpc.end());
                min_assoc(v) = std::max(min_assoc(v), pvalue);
                old_cpc.pop_back();
            }

            progress->tick();
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

    template<typename VectorType>
    std::unordered_set<int> mmpc_forward_phase(const IndependenceTest& test, 
                                               int variable,
                                               double alpha,
                                               std::unordered_set<int>& cpc,
                                               std::unordered_set<int>& to_be_checked,
                                               VectorType& min_assoc,
                                               int last_added,
                                               util::BaseProgressBar* progress) {
        bool changed_cpc = true;

        if (cpc.empty()) {
            min_assoc.fill(0);
        } else if (last_added == MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC) {
            // The CPC is not empty because of whitelists, so we compute the association of the selected CPC.
            recompute_assoc(test, variable, cpc, to_be_checked, min_assoc, progress);
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
            update_min_assoc(test, variable, to_be_checked, cpc, min_assoc, last_added, progress);
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

    bool is_whitelisted_pc(int variable, int candidate_pc, const ArcSet& arc_whitelist, const EdgeSet& edge_whitelist) {
        return edge_whitelist.count({variable, candidate_pc}) > 0 || 
               arc_whitelist.count({variable, candidate_pc}) > 0 || 
               arc_whitelist.count({candidate_pc, variable}) > 0;
    }

    void mmpc_backward_phase(const IndependenceTest& test, 
                             int variable,
                             double alpha,
                             std::unordered_set<int>& cpc,
                             const ArcSet& arc_whitelist,
                             const EdgeSet& edge_whitelist,
                             util::BaseProgressBar* progress) {

        if (cpc.size() > 1) {
            std::vector<int> subset_variables {cpc.begin(), cpc.end()};

            progress->set_text("MMPC Backwards for " + test.name(variable));
            progress->set_max_progress(cpc.size());
            progress->set_progress(0);

            for (auto it = cpc.begin(), end = cpc.end(); it != end;) {
                if (is_whitelisted_pc(variable, *it, arc_whitelist, edge_whitelist)) {
                    ++it;
                    progress->tick();
                    continue;
                }

                util::swap_remove_v(subset_variables, *it);

                // Marginal independence
                if (test.pvalue(variable, *it) > alpha) {
                    it = cpc.erase(it);
                    progress->set_max_progress(cpc.size());
                    progress->tick();
                    continue;
                }

                // Independence sepset length 1.
                bool found_sepset = false;
                for (auto it_other = subset_variables.begin(), end_other = subset_variables.end(); it_other != end_other; ++it_other) {
                    if (test.pvalue(variable, *it, *it_other) > alpha) {
                        it = cpc.erase(it);
                        progress->set_max_progress(cpc.size());
                        found_sepset = true;
                        break;
                    }
                }

                if (!found_sepset && subset_variables.size() > 2) {
                    // Independence sepset length 2 to subset size - 1.
                    AllSubsets comb(subset_variables, 2, subset_variables.size()-1);

                    for (const auto& s : comb) {
                        if (test.pvalue(variable, *it, s.begin(), s.end()) > alpha) {
                            it = cpc.erase(it);
                            progress->set_max_progress(cpc.size());
                            found_sepset = true;
                            break;
                        }
                    }
                }

                // Independence sepset length of subset size.
                if (!found_sepset && subset_variables.size() > 1 && 
                    test.pvalue(variable, *it, subset_variables.begin(), subset_variables.end()) > alpha) {
                    it = cpc.erase(it);
                    progress->set_max_progress(cpc.size());
                    found_sepset = true;
                }

                if (!found_sepset) {
                    // No sepset found, so include again the variable.
                    subset_variables.push_back(*it);
                    ++it;
                }

                progress->tick();
            }
        }
    }

    std::unordered_set<int> mmpc_variable(const IndependenceTest& test,
                                          int variable,
                                          double alpha,
                                          const ArcSet& arc_whitelist,
                                          const EdgeSet& edge_blacklist,
                                          const EdgeSet& edge_whitelist) {
        std::unordered_set<int> cpc;
        std::unordered_set<int> to_be_checked;

        for (int i = 0; i < test.num_columns(); ++i) {
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

        VectorXd min_assoc(test.num_columns());

        int last_added = 0;
        if (!cpc.empty()) last_added = MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC;

        util::VoidProgressBar void_bar;
        mmpc_forward_phase(test, variable, alpha, cpc, to_be_checked, min_assoc, last_added, &void_bar);
        mmpc_backward_phase(test, variable, alpha, cpc, arc_whitelist, edge_whitelist, &void_bar);
        return cpc;
    }

    void marginal_cpcs_all_variables(const IndependenceTest& test,
                                     double alpha,
                                     std::vector<std::unordered_set<int>>& cpcs,
                                     std::vector<std::unordered_set<int>>& to_be_checked,
                                     const EdgeSet& edge_blacklist,
                                     MatrixXd& min_assoc,
                                     VectorXd& maxmin_assoc,
                                     VectorXi& maxmin_index,
                                     util::BaseProgressBar* progress) {
        auto nnodes = test.num_columns();
        progress->set_text("MMPC Forward: No sepset");
        progress->set_max_progress((nnodes*(nnodes-1) / 2));
        progress->set_progress(0);

        for (int i = 0, i_end = nnodes-1; i < i_end; ++i) {
            for (int j = i+1; j < nnodes; ++j) {
                if ((cpcs[i].empty() || cpcs[j].empty()) && edge_blacklist.count({i,j}) == 0) {
                    double pvalue = test.pvalue(i, j);

                    if (pvalue < alpha) {
                        if (cpcs[i].empty()) {
                            min_assoc(j, i) = pvalue;
                            to_be_checked[i].insert(j);
                            if (min_assoc(j, i) < maxmin_assoc(i)) {
                                maxmin_assoc(i) = min_assoc(j, i);
                                maxmin_index(i) = j;
                            }
                        }
                        
                        if (cpcs[j].empty()) {
                            min_assoc(i, j) = pvalue;
                            to_be_checked[j].insert(i);
                            if (min_assoc(i, j) < maxmin_assoc(j)) {
                                maxmin_assoc(j) = min_assoc(i, j);
                                maxmin_index(j) = i;
                            }
                        }
                    }
                }

                progress->tick();
            }
        }
    }

    void univariate_cpcs_all_variables(const IndependenceTest& test,
                                       double alpha,
                                       std::vector<std::unordered_set<int>>& cpcs,
                                       std::vector<std::unordered_set<int>>& to_be_checked,
                                       MatrixXd& min_assoc,
                                       VectorXd& maxmin_assoc,
                                       VectorXi& maxmin_index,
                                       util::BaseProgressBar* progress) {
        auto nnodes = test.num_columns();
        progress->set_text("MMPC Forward: sepset order 1");
        progress->set_max_progress(nnodes);
        progress->set_progress(0);
        
        for (int i = 0; i < nnodes; ++i) {
            if (cpcs[i].size() == 1) {
                int cpc_variable = *cpcs[i].begin();
                for (auto it = to_be_checked[i].begin(), end = to_be_checked[i].end(); it != end;) {
                    auto p = *it;
                    bool repeated_test = cpcs[p].size() == 1 && cpc_variable == *cpcs[p].begin();

                    if (i < p || !repeated_test) {
                        double pvalue = test.pvalue(i, p, cpc_variable);

                        min_assoc(p, i) = std::max(min_assoc(p, i), pvalue);
                        if (repeated_test)
                            min_assoc(i, p) = std::max(min_assoc(i, p), pvalue);

                        if (min_assoc(p, i) > alpha) {
                            it = to_be_checked[i].erase(it);
                            if (repeated_test && min_assoc(i, p) > alpha) {
                                to_be_checked[p].erase(i);
                            }
                        } else {
                            if (min_assoc(p, i) < maxmin_assoc(i)) {
                                maxmin_assoc(i) = min_assoc(p, i);
                                maxmin_index(i) = p;
                                if (repeated_test && min_assoc(i, p) < maxmin_assoc(p)) {
                                    maxmin_assoc(p) = min_assoc(i, p);
                                    maxmin_index(p) = i;
                                }
                            }
                            ++it;
                        }
                    } else {
                        ++it;
                    }
                }
            }

            progress->tick();
        }
    }

    std::vector<std::unordered_set<int>> mmpc_all_variables(const IndependenceTest& test, 
                                                            double alpha,
                                                            const ArcSet& arc_whitelist,
                                                            const EdgeSet& edge_blacklist,
                                                            const EdgeSet& edge_whitelist,
                                                            util::BaseProgressBar* progress) {
        std::vector<std::unordered_set<int>> cpcs(test.num_columns());
        std::vector<std::unordered_set<int>> to_be_checked(test.num_columns());

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
        for (int i = 0; i < test.num_columns(); ++i) {
            if (!cpcs[i].empty()) {
                for (int j = 0; j < test.num_columns(); ++j) {
                    if (j != i && edge_blacklist.count({i,j}) == 0 && cpcs[i].count(j) == 0) {
                        to_be_checked[i].insert(j);
                    }
                }
            }
        }

        MatrixXd min_assoc = MatrixXd::Zero(test.num_columns(), test.num_columns());
        VectorXd maxmin_assoc = VectorXd::Constant(test.num_columns(), std::numeric_limits<double>::infinity());
        VectorXi maxmin_index = VectorXi::Constant(test.num_columns(), MMPC_FORWARD_PHASE_STOP);

        marginal_cpcs_all_variables(test, alpha, cpcs, to_be_checked, edge_blacklist, min_assoc, maxmin_assoc, maxmin_index, progress);

        bool all_finished = true;
        for (int i = 0; i < test.num_columns(); ++i) {
            if (maxmin_index(i) != MMPC_FORWARD_PHASE_STOP) {
                all_finished = false;
                cpcs[i].insert(maxmin_index(i));
                to_be_checked[i].erase(maxmin_index(i));
            }
        }

        if (!all_finished) {
            for (int i = 0; i < test.num_columns(); ++i) {
                if (cpcs[i].size() == 1) {
                    maxmin_assoc(i) = std::numeric_limits<double>::infinity();
                    maxmin_index(i) = MMPC_FORWARD_PHASE_STOP;
                }
            }

            univariate_cpcs_all_variables(test, alpha, cpcs, to_be_checked, min_assoc, maxmin_assoc, maxmin_index, progress);

            for (int i = 0; i < test.num_columns(); ++i) {
                auto col_min_assoc = min_assoc.col(i);
                // The cpc is whitelisted.
                if (cpcs[i].size() > 1) {
                    mmpc_forward_phase(test, i, alpha, cpcs[i], to_be_checked[i], 
                                    col_min_assoc, MMPC_FORWARD_PHASE_RECOMPUTE_ASSOC, progress);
                } else if (maxmin_index(i) != MMPC_FORWARD_PHASE_STOP) {
                    cpcs[i].insert(maxmin_index(i));
                    to_be_checked[i].erase(maxmin_index(i));

                    mmpc_forward_phase(test, i, alpha, cpcs[i], to_be_checked[i], col_min_assoc, maxmin_index(i), progress);
                }
                
                mmpc_backward_phase(test, i, alpha, cpcs[i], arc_whitelist, edge_whitelist, progress);
            }
        }

        return cpcs;
    }

    PartiallyDirectedGraph MMPC::estimate(const IndependenceTest& test,
                    const ArcStringVector& varc_blacklist, 
                    const ArcStringVector& varc_whitelist,
                    const EdgeStringVector& vedge_blacklist,
                    const EdgeStringVector& vedge_whitelist,
                    double alpha,
                    double ambiguous_threshold,
                    bool allow_bidirected,
                    int verbose) const {
        

        PartiallyDirectedGraph skeleton(test.column_names());

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

        auto cpcs = mmpc_all_variables(test, alpha, restrictions.arc_whitelist, 
                                        restrictions.edge_blacklist, restrictions.edge_whitelist, progress.get());

        for (auto i = 0; i < test.num_columns(); ++i) {
            for (auto p : cpcs[i]) {
                if (i < p && cpcs[p].count(i) > 0 && !skeleton.has_arc(i, p) && !skeleton.has_arc(p, i)) {
                    skeleton.add_edge(i, p);
                }
            }
        }


        direct_arc_blacklist(skeleton, restrictions.arc_blacklist);
        direct_unshielded_triples(skeleton, test, restrictions.arc_blacklist, restrictions.arc_whitelist, 
                                  alpha, std::nullopt, true, ambiguous_threshold, allow_bidirected, progress.get());

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
}