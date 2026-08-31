#ifndef PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_MUTUAL_INFORMATION_HPP
#define PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_MUTUAL_INFORMATION_HPP

#include <random>
#include <dataset/dataset.hpp>
#include <learning/independences/independence.hpp>
#include <vptree/vptree.hpp>

using dataset::DataFrame, dataset::Copy;
using Eigen::MatrixXi;
using Array_ptr = std::shared_ptr<arrow::Array>;
using vptree::VPTree;

namespace learning::independences::hybrid {
DataFrame scale_data(const DataFrame& df, const std::string& scaling);

double mi_general(VPTree& ztree,
                  DataFrame& df,
                  int k,
                  std::shared_ptr<arrow::DataType> datatype,
                  std::vector<bool>& is_discrete_column,
                  int tree_leafsize,
                  unsigned int seed);
double mi_pair(VPTree& ytree,
               DataFrame& df,
               int k,
               std::shared_ptr<arrow::DataType> datatype,
               std::vector<bool>& is_discrete_column,
               int tree_leafsize,
               unsigned int seed);

class MixedKMutualInformation : public IndependenceTest {
public:
    MixedKMutualInformation(DataFrame df,
                         int k,
                         unsigned int seed = std::random_device{}(),
                         int shuffle_neighbors = 5,
                         int samples = 1000,
                         std::string scaling = "min_max",
                         bool gamma_approx = true,
                         bool adaptive_k = true,
                         int tree_leafsize = 16)
        : m_df(df),
          m_scaled_df(scale_data(df, scaling)),
          m_datatype(),
          m_k(k),
          m_seed(seed),
          m_shuffle_neighbors(shuffle_neighbors),
          m_samples(samples),
          m_gamma_approx(gamma_approx),
          m_adaptive_k(adaptive_k),
          m_tree_leafsize(tree_leafsize) {
        m_datatype = m_scaled_df.same_type();
    }

    double pvalue(const std::string& x, const std::string& y) const override;
    double pvalue(const std::string& x, const std::string& y, const std::string& z) const override;
    double pvalue(const std::string& x, const std::string& y, const std::vector<std::string>& z) const override;

    double mi(const std::string& x, const std::string& y) const;
    double mi(const std::string& x, const std::string& y, const std::string& z) const;
    double mi(const std::string& x, const std::string& y, const std::vector<std::string>& z) const;

    int num_variables() const override { return m_df->num_columns(); }

    std::vector<std::string> variable_names() const override { return m_df.column_names(); }

    const std::string& name(int i) const override { return m_df.name(i); }

    bool has_variables(const std::string& name) const override { return m_df.has_columns(name); }

    bool has_variables(const std::vector<std::string>& cols) const override { return m_df.has_columns(cols); }

private:
    double shuffled_pvalue(double original_mi,
                           int k,
                           int shuffle_neighbors,
                           DataFrame& x_df,
                           VPTree& ztree,
                           DataFrame& z_df,
                           DataFrame& shuffled_df,
                           std::vector<bool>& is_discrete_column,
                           std::vector<std::string>& discrete_vars) const;

    int find_minimum_cluster_size(const std::vector<std::string>& discrete_vars) const;
    int find_minimum_shuffled_cluster_size(const DataFrame& shuffled_df,
                                                                 const std::vector<std::string>& discrete_vars) const;
    DataFrame m_df;
    DataFrame m_scaled_df;
    std::shared_ptr<arrow::DataType> m_datatype;
    int m_k;
    unsigned int m_seed;
    int m_shuffle_neighbors;
    int m_samples;
    bool m_gamma_approx;
    bool m_adaptive_k;
    int m_tree_leafsize;
};

}  // namespace learning::independences::hybrid

#endif  // PYBNESIAN_LEARNING_INDEPENDENCES_HYBRID_MS_MUTUAL_INFORMATION_HPP