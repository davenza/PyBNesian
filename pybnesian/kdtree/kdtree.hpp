#ifndef PYBNESIAN_KDTREE_KDTREE_HPP
#define PYBNESIAN_KDTREE_KDTREE_HPP

#include <dataset/dataset.hpp>
#include <queue>

using dataset::DataFrame;
using Eigen::Matrix, Eigen::Dynamic, Eigen::VectorXd, Eigen::VectorXi;

template <typename ArrowType>
using EigenVector = Matrix<typename ArrowType::c_type, Dynamic, 1>;

template <typename ArrowType>
using DowncastArray_ptr = std::shared_ptr<typename arrow::TypeTraits<ArrowType>::ArrayType>;

template <typename ArrowType>
using DowncastArray_vector = std::vector<DowncastArray_ptr<ArrowType>>;

namespace kdtree {

template <typename ArrowType>
class EuclideanDistance {
public:
    using CType = typename ArrowType::c_type;
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

    EuclideanDistance(const std::vector<std::shared_ptr<ArrayType>>& train_data,
                      const std::vector<std::shared_ptr<ArrayType>>& test_data)
        : m_train_data(train_data), m_test_data(test_data) {}

    inline CType distance(size_t train_index, size_t test_index) const {
        CType d = 0;
        for (auto it_train = m_train_data.begin(), it_train_end = m_train_data.end(), it_test = m_test_data.begin();
             it_train != it_train_end;
             ++it_train, ++it_test) {
            CType t = ((*it_test)->Value(test_index) - (*it_train)->Value(train_index));
            d += distance_p(t);
        }

        return d;
    }

    inline CType distance_p(CType difference) const { return difference * difference; }

    inline CType normalize(CType nonnormalized) const { return std::sqrt(nonnormalized); }

    inline CType update_component_distance(CType distance, CType old_component, CType new_component) const {
        return distance - old_component + new_component;
    }

private:
    const std::vector<std::shared_ptr<ArrayType>>& m_train_data;
    const std::vector<std::shared_ptr<ArrayType>>& m_test_data;
};

template <typename ArrowType>
class ManhattanDistance {
public:
    using CType = typename ArrowType::c_type;
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

    ManhattanDistance(const std::vector<std::shared_ptr<ArrayType>>& train_data,
                      const std::vector<std::shared_ptr<ArrayType>>& test_data)
        : m_train_data(train_data), m_test_data(test_data) {}

    inline CType distance(size_t train_index, size_t test_index) const {
        CType d = 0;
        for (auto it_train = m_train_data.begin(), it_train_end = m_train_data.end(), it_test = m_test_data.begin();
             it_train != it_train_end;
             ++it_train, ++it_test) {
            CType t = ((*it_test)->Value(test_index) - (*it_train)->Value(train_index));
            d += distance_p(t);
        }

        return d;
    }

    inline CType distance_p(CType difference) const { return std::abs(difference); }

    inline CType normalize(CType nonnormalized) const { return nonnormalized; }

    inline CType update_component_distance(CType distance, CType old_component, CType new_component) const {
        return distance - old_component + new_component;
    }

private:
    const std::vector<std::shared_ptr<ArrayType>>& m_train_data;
    const std::vector<std::shared_ptr<ArrayType>>& m_test_data;
};

template <typename ArrowType>
class ChebyshevDistance {
public:
    using CType = typename ArrowType::c_type;
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

    ChebyshevDistance(const std::vector<std::shared_ptr<ArrayType>>& train_data,
                      const std::vector<std::shared_ptr<ArrayType>>& test_data)
        : m_train_data(train_data), m_test_data(test_data) {}

    inline CType distance(size_t train_index, size_t test_index) const {
        CType d = 0;
        for (auto it_train = m_train_data.begin(), it_train_end = m_train_data.end(), it_test = m_test_data.begin();
             it_train != it_train_end;
             ++it_train, ++it_test) {
            CType t = ((*it_test)->Value(test_index) - (*it_train)->Value(train_index));
            d = std::max(d, abs(t));
        }

        return d;
    }

    inline CType distance_p(CType difference) const { return std::abs(difference); }

    inline CType normalize(CType nonnormalized) const { return nonnormalized; }

    inline CType update_component_distance(CType distance, CType, CType new_component) const {
        return std::max(distance, new_component);
    }

private:
    const std::vector<std::shared_ptr<ArrayType>>& m_train_data;
    const std::vector<std::shared_ptr<ArrayType>>& m_test_data;
};

template <typename ArrowType>
class MinkowskiP {
public:
    using CType = typename ArrowType::c_type;
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

    MinkowskiP(const std::vector<std::shared_ptr<ArrayType>>& train_data,
               const std::vector<std::shared_ptr<ArrayType>>& test_data,
               double p)
        : m_train_data(train_data), m_test_data(test_data), m_p(p) {}

    inline CType distance(size_t train_index, size_t test_index) const {
        CType d = 0;
        for (auto it_train = m_train_data.begin(), it_train_end = m_train_data.end(), it_test = m_test_data.begin();
             it_train != it_train_end;
             ++it_train, ++it_test) {
            CType t = ((*it_test)->Value(test_index) - (*it_train)->Value(train_index));
            d += distance_p(t);
        }

        return d;
    }

    inline CType distance_p(CType difference) const { return std::pow(std::abs(difference), static_cast<CType>(m_p)); }

    inline CType normalize(CType nonnormalized) const { return std::pow(nonnormalized, static_cast<CType>(1. / m_p)); }

    inline CType update_component_distance(CType distance, CType old_component, CType new_component) const {
        return distance - old_component + new_component;
    }

private:
    const std::vector<std::shared_ptr<ArrayType>>& m_train_data;
    const std::vector<std::shared_ptr<ArrayType>>& m_test_data;
    double m_p;
};

template <typename T>
struct IndexComparator {
    const T* data;

    IndexComparator(const T* data) : data(data){};

    inline bool operator()(size_t a, size_t b) { return data[a] < data[b]; }
};

template <typename ArrowType>
using Neighbor = std::pair<typename ArrowType::c_type, size_t>;

template <typename ArrowType>
struct NeighborComparator {
    inline bool operator()(const Neighbor<ArrowType>& a, const Neighbor<ArrowType>& b) { return a.first < b.first; }
};

template <typename ArrowType>
using NeighborQueue =
    std::priority_queue<Neighbor<ArrowType>, std::vector<Neighbor<ArrowType>>, NeighborComparator<ArrowType>>;

struct KDTreeNode {
    size_t split_id;
    double split_value;
    KDTreeNode* parent;
    std::unique_ptr<KDTreeNode> left;
    std::unique_ptr<KDTreeNode> right;
    bool is_leaf;
    std::vector<size_t>::iterator indices_begin;
    std::vector<size_t>::iterator indices_end;
};

template <typename ArrowType>
struct QueryNode {
    KDTreeNode* node;
    typename ArrowType::c_type min_distance;
    Matrix<typename ArrowType::c_type, Dynamic, 1> side_distance;
};

template <typename ArrowType>
struct QueryNodeComparator {
    inline bool operator()(const QueryNode<ArrowType>& a, const QueryNode<ArrowType>& b) {
        auto d = a.min_distance - b.min_distance;
        if (d != 0) {
            return d > 0;
        } else {
            return a.node->is_leaf < b.node->is_leaf;
        }
    }
};

template <typename ArrowType>
using QueryQueue =
    std::priority_queue<QueryNode<ArrowType>, std::vector<QueryNode<ArrowType>>, QueryNodeComparator<ArrowType>>;

template <typename ArrowType>
std::unique_ptr<KDTreeNode> build_kdtree(const DataFrame& df,
                                         int leafsize,
                                         std::vector<size_t>::iterator indices_begin,
                                         std::vector<size_t>::iterator indices_end,
                                         int updated_index,
                                         bool update_left,
                                         EigenVector<ArrowType> maxes,
                                         EigenVector<ArrowType> mines) {
    using CType = typename ArrowType::c_type;

    auto n = std::distance(indices_begin, indices_end);

    if (n <= leafsize) {
        auto leaf = std::make_unique<KDTreeNode>();
        leaf->is_leaf = true;
        leaf->indices_begin = indices_begin;
        leaf->indices_end = indices_end;
        return leaf;
    } else {
        if (updated_index != -1) {
            if (update_left) {
                maxes(updated_index) = -std::numeric_limits<CType>::infinity();
                auto array = df.downcast<ArrowType>(updated_index);
                auto raw_values = array->raw_values();

                for (auto it = indices_begin; it != indices_end; ++it) {
                    maxes(updated_index) = std::max(maxes(updated_index), raw_values[*it]);
                }

            } else {
                mines(updated_index) = std::numeric_limits<CType>::infinity();
                auto array = df.downcast<ArrowType>(updated_index);
                auto raw_values = array->raw_values();

                for (auto it = indices_begin; it != indices_end; ++it) {
                    mines(updated_index) = std::min(mines(updated_index), raw_values[*it]);
                }
            }
        }

        size_t split_id = 0;
        double spread_size = 0;
        for (int j = 0; j < df->num_columns(); ++j) {
            if (maxes(j) - mines(j) > spread_size) {
                split_id = j;
                spread_size = maxes(j) - mines(j);
            }
        }

        if (mines(split_id) == maxes(split_id)) {
            auto leaf = std::make_unique<KDTreeNode>();
            leaf->is_leaf = true;
            leaf->indices_begin = indices_begin;
            leaf->indices_end = indices_end;
            return leaf;
        }

        auto median_id = n / 2;
        auto mid_iter = indices_begin + median_id;

        auto dwn_split_array = df.downcast<ArrowType>(split_id);

        IndexComparator index_comparator(dwn_split_array->raw_values());

        std::nth_element(indices_begin, mid_iter, indices_end, index_comparator);

        auto node = std::make_unique<KDTreeNode>();

        node->split_id = split_id;
        node->split_value = static_cast<double>(dwn_split_array->Value(*mid_iter));
        node->parent = nullptr;

        node->left = build_kdtree<ArrowType>(df, leafsize, indices_begin, mid_iter, split_id, true, maxes, mines);
        node->left->parent = node.get();

        node->right = build_kdtree<ArrowType>(df, leafsize, mid_iter, indices_end, split_id, false, maxes, mines);
        node->right->parent = node.get();

        node->is_leaf = false;

        return node;
    }
}

class KDTree {
public:
    KDTree() : m_df(), m_column_names(), m_datatype(), m_root(), m_indices(), m_maxes(), m_mines() {}

    KDTree(DataFrame df, int leafsize = 16)
        : m_df(df),
          m_column_names(df.column_names()),
          m_datatype(df.same_type()),
          m_root(),
          m_indices(df->num_rows()),
          m_maxes(df->num_columns()),
          m_mines(df->num_columns()) {
        std::iota(m_indices.begin(), m_indices.end(), 0);

        switch (m_datatype->id()) {
            case Type::DOUBLE: {
                for (int j = 0; j < df->num_columns(); ++j) {
                    m_mines(j) = df.min<arrow::DoubleType>(j);
                    m_maxes(j) = df.max<arrow::DoubleType>(j);
                }

                break;
            }
            case Type::FLOAT: {
                for (int j = 0; j < df->num_columns(); ++j) {
                    m_mines(j) = df.min<arrow::FloatType>(j);
                    m_maxes(j) = df.max<arrow::FloatType>(j);
                }
                break;
            }
            default:
                throw std::invalid_argument("Wrong data type to apply KDTree.");
        }

        m_root = build_kdtree(df, leafsize);
    }

    void fit(DataFrame df, int leafsize = 16);
    std::vector<std::pair<VectorXd, VectorXi>> query(const DataFrame& test_df, int k = 1, double p = 2) const;
    template <typename ArrowType, typename DistanceType>
    std::pair<VectorXd, VectorXi> query_instance(const DowncastArray_vector<ArrowType>& test_downcast,
                                                 size_t i,
                                                 int k,
                                                 const DistanceType& distance) const;

    std::tuple<VectorXi, VectorXi, VectorXi> count_ball_subspaces(const DataFrame& test_df,
                                                                  const Array_ptr& x_data,
                                                                  const Array_ptr& y_data,
                                                                  const VectorXd& eps) const;

    template <typename ArrowType, typename DistanceType>
    std::tuple<int, int, int> count_ball_subspaces_instance(const DowncastArray_vector<ArrowType>& test_df,
                                                            const typename ArrowType::c_type* x_data,
                                                            const typename ArrowType::c_type* y_data,
                                                            size_t i,
                                                            const DistanceType& distance,
                                                            const typename ArrowType::c_type eps_value) const;

    const DataFrame& ranked_data() const { return m_df; }

private:
    std::unique_ptr<KDTreeNode> build_kdtree(const DataFrame& df, int leafsize);

    DataFrame m_df;
    std::vector<std::string> m_column_names;
    std::shared_ptr<arrow::DataType> m_datatype;
    std::unique_ptr<KDTreeNode> m_root;
    std::vector<size_t> m_indices;
    VectorXd m_maxes;
    VectorXd m_mines;
};

template <typename ArrowType, typename DistanceType>
std::pair<VectorXd, VectorXi> KDTree::query_instance(const DowncastArray_vector<ArrowType>& test_downcast,
                                                     size_t i,
                                                     int k,
                                                     const DistanceType& distance) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<typename ArrowType::c_type, Dynamic, 1>;

    NeighborQueue<ArrowType> neighbors;

    CType distance_upper_bound = std::numeric_limits<CType>::infinity();
    for (auto i = 0; i < k; ++i) {
        neighbors.push(std::make_pair(distance_upper_bound, -1));
    }

    VectorType side_distance(m_column_names.size());
    CType min_distance = 0;

    for (size_t j = 0; j < m_column_names.size(); ++j) {
        auto x_value = test_downcast[j]->Value(i);
        side_distance(j) = std::max(0., std::max(x_value - m_maxes(j), m_mines(j) - x_value));
        side_distance(j) = distance.distance_p(side_distance(j));
        min_distance = distance.update_component_distance(min_distance, 0, side_distance(j));
    }

    QueryQueue<ArrowType> query_nodes;

    query_nodes.push(QueryNode<ArrowType>{/*.node = */ m_root.get(),
                                          /*.min_distance = */ min_distance,
                                          /*.side_distance = */ side_distance});

    while (!query_nodes.empty()) {
        auto& query = query_nodes.top();
        auto node = query.node;

        if (query.min_distance >= distance_upper_bound) break;

        if (node->is_leaf) {
            for (auto it = node->indices_begin; it != node->indices_end; ++it) {
                auto d = distance.distance(*it, i);
                if (d < distance_upper_bound) {
                    neighbors.pop();
                    neighbors.push(std::make_pair(static_cast<double>(d), *it));
                    distance_upper_bound = neighbors.top().first;
                }
            }
            query_nodes.pop();
        } else {
            KDTreeNode* near_node;
            KDTreeNode* far_node;

            auto p = test_downcast[node->split_id]->Value(i);

            if (p < node->split_value) {
                near_node = node->left.get();
                far_node = node->right.get();
            } else {
                near_node = node->right.get();
                far_node = node->left.get();
            }

            QueryNode<ArrowType> near_query{/*.node = */ near_node,
                                            /*.min_distance = */ query.min_distance,
                                            /*.side_distance = */ query.side_distance};

            VectorType far_side_distance = query.side_distance;

            auto dis = node->split_value - p;
            far_side_distance(node->split_id) = distance.distance_p(dis);
            CType far_min_distance = distance.update_component_distance(
                query.min_distance, query.side_distance(node->split_id), far_side_distance(node->split_id));

            query_nodes.pop();
            query_nodes.push(near_query);

            if (far_min_distance < distance_upper_bound) {
                query_nodes.push(QueryNode<ArrowType>{/*.node = */ far_node,
                                                      /*.min_distance = */ far_min_distance,
                                                      /*.side_distance = */ far_side_distance});
            }
        }
    }

    VectorXd distances(k);
    VectorXi indices(k);

    auto u = k - 1;
    while (!neighbors.empty()) {
        auto& neigh = neighbors.top();
        distances(u) = distance.normalize(neigh.first);
        indices(u) = neigh.second;
        neighbors.pop();
        --u;
    }

    return std::make_pair(distances, indices);
}

template <typename ArrowType, typename DistanceType>
std::tuple<int, int, int> KDTree::count_ball_subspaces_instance(const DowncastArray_vector<ArrowType>& test_downcast,
                                                                const typename ArrowType::c_type* x_data,
                                                                const typename ArrowType::c_type* y_data,
                                                                size_t i,
                                                                const DistanceType& distance,
                                                                const typename ArrowType::c_type eps_value) const {
    using CType = typename ArrowType::c_type;
    using VectorType = Matrix<typename ArrowType::c_type, Dynamic, 1>;

    VectorType side_distance(test_downcast.size());
    CType min_distance = 0;

    for (size_t j = 0; j < test_downcast.size(); ++j) {
        auto p = test_downcast[j]->Value(i);
        side_distance(j) = std::max(0., std::max(p - m_maxes(j), m_mines(j) - p));
        side_distance(j) = distance.distance_p(side_distance(j));
        min_distance = distance.update_component_distance(min_distance, 0, side_distance(j));
    }

    int count_xz = 0, count_yz = 0, count_z = 0;

    QueryQueue<ArrowType> query_nodes;

    if (min_distance < eps_value) {
        query_nodes.push(QueryNode<ArrowType>{/*.node = */ m_root.get(),
                                              /*.min_distance = */ min_distance,
                                              /*.side_distance = */ side_distance});
    }

    while (!query_nodes.empty()) {
        auto& query = query_nodes.top();
        auto node = query.node;

        if (node->is_leaf) {
            for (auto it = node->indices_begin; it != node->indices_end; ++it) {
                auto d = distance.distance(*it, i);

                if (d < eps_value) {
                    ++count_z;
                    if (std::abs(x_data[*it] - x_data[i]) < eps_value) ++count_xz;
                    if (std::abs(y_data[*it] - y_data[i]) < eps_value) ++count_yz;
                }
            }

            query_nodes.pop();
        } else {
            KDTreeNode* near_node;
            KDTreeNode* far_node;

            auto p = test_downcast[node->split_id]->Value(i);
            if (p < node->split_value) {
                near_node = node->left.get();
                far_node = node->right.get();
            } else {
                near_node = node->right.get();
                far_node = node->left.get();
            }

            QueryNode<ArrowType> near_query{/*.node = */ near_node,
                                            /*.min_distance = */ query.min_distance,
                                            /*.side_distance = */ query.side_distance};

            CType far_dimension_distance = distance.distance_p(node->split_value - p);
            CType far_node_distance = distance.update_component_distance(
                query.min_distance, query.side_distance(node->split_id), far_dimension_distance);

            query_nodes.pop();
            query_nodes.push(near_query);

            if (far_node_distance < eps_value) {
                VectorType far_side_distance = near_query.side_distance;
                far_side_distance(node->split_id) = far_dimension_distance;
                query_nodes.push(QueryNode<ArrowType>{/*.node = */ far_node,
                                                      /*.min_distance = */ far_node_distance,
                                                      /*.side_distance = */ far_side_distance});
            }
        }
    }

    return std::make_tuple(count_xz, count_yz, count_z);
}

}  // namespace kdtree

#endif  // PYBNESIAN_KDTREE_KDTREE_HPP