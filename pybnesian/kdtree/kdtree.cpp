#include <kdtree/kdtree.hpp>

namespace kdtree {

std::unique_ptr<KDTreeNode> KDTree::build_kdtree(const DataFrame& df, int leafsize) {
    switch (df.same_type()->id()) {
        case Type::DOUBLE: {
            return kdtree::build_kdtree<arrow::DoubleType>(
                df, leafsize, m_indices.begin(), m_indices.end(), -1, true, m_maxes, m_mines);
        }
        case Type::FLOAT: {
            return kdtree::build_kdtree<arrow::FloatType>(df,
                                                          leafsize,
                                                          m_indices.begin(),
                                                          m_indices.end(),
                                                          -1,
                                                          true,
                                                          m_maxes.template cast<float>(),
                                                          m_mines.template cast<float>());
        }
        default:
            throw std::invalid_argument("Wrong data type to apply KDTree.");
    }
}

void KDTree::fit(DataFrame df, int leafsize) {
    m_df = df;
    m_column_names = df.column_names();
    m_datatype = df.same_type();
    m_indices.resize(df->num_rows());
    std::iota(m_indices.begin(), m_indices.end(), 0);
    m_maxes = VectorXd(df->num_columns());
    m_mines = VectorXd(df->num_columns());

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

std::vector<std::pair<VectorXd, VectorXi>> KDTree::query(const DataFrame& test_df, int k, double p) const {
    if (k >= m_df->num_rows()) {
        throw std::invalid_argument("\"k\" value equal or greater to training data size.");
    }

    test_df.raise_has_columns(m_column_names);

    if (test_df.same_type(m_column_names)->id() != m_datatype->id()) {
        throw std::invalid_argument("Test data type is different from training data types.");
    }

    std::vector<std::pair<VectorXd, VectorXi>> res;
    res.reserve(test_df->num_rows());

    switch (m_datatype->id()) {
        case Type::DOUBLE: {
            auto train_downcast = m_df.downcast_vector<arrow::DoubleType>(m_column_names);
            auto test_downcast = test_df.downcast_vector<arrow::DoubleType>(m_column_names);

            if (p == 1) {
                ManhattanDistance<arrow::DoubleType> dist(train_downcast, test_downcast);
                for (int i = 0; i < test_df->num_rows(); ++i) {
                    auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                    res.push_back(t);
                }
            } else if (p == 2) {
                EuclideanDistance<arrow::DoubleType> dist(train_downcast, test_downcast);
                for (int i = 0; i < test_df->num_rows(); ++i) {
                    auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                    res.push_back(t);
                }
            } else if (std::isinf(p)) {
                ChebyshevDistance<arrow::DoubleType> dist(train_downcast, test_downcast);
                for (int i = 0; i < test_df->num_rows(); ++i) {
                    auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                    res.push_back(t);
                }
            } else {
                MinkowskiP<arrow::DoubleType> dist(train_downcast, test_downcast, p);
                for (int i = 0; i < test_df->num_rows(); ++i) {
                    auto t = query_instance<arrow::DoubleType>(test_downcast, i, k, dist);
                    res.push_back(t);
                }
            }
            break;
        }
        case Type::FLOAT: {
            auto train_downcast = m_df.downcast_vector<arrow::FloatType>(m_column_names);
            auto test_downcast = test_df.downcast_vector<arrow::FloatType>(m_column_names);

            if (p == 1) {
                ManhattanDistance<arrow::FloatType> dist(train_downcast, test_downcast);
                for (int i = 0; i < test_df->num_rows(); ++i) {
                    auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                    res.push_back(t);
                }
            } else if (p == 2) {
                EuclideanDistance<arrow::FloatType> dist(train_downcast, test_downcast);
                for (int i = 0; i < test_df->num_rows(); ++i) {
                    auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                    res.push_back(t);
                }
            } else if (std::isinf(p)) {
                ChebyshevDistance<arrow::FloatType> dist(train_downcast, test_downcast);
                for (int i = 0; i < test_df->num_rows(); ++i) {
                    auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                    res.push_back(t);
                }
            } else {
                MinkowskiP<arrow::FloatType> dist(train_downcast, test_downcast, p);
                for (int i = 0; i < test_df->num_rows(); ++i) {
                    auto t = query_instance<arrow::FloatType>(test_downcast, i, k, dist);
                    res.push_back(t);
                }
            }
            break;
        }
        default:
            throw std::invalid_argument("Wrong data type to apply KDTree.");
    }

    return res;
}

std::tuple<VectorXi, VectorXi, VectorXi> KDTree::count_ball_subspaces(const DataFrame& test_df,
                                                                      const Array_ptr& x_data,
                                                                      const Array_ptr& y_data,
                                                                      const VectorXd& eps) const {
    VectorXi count_xz(test_df->num_rows());
    VectorXi count_yz(test_df->num_rows());
    VectorXi count_z(test_df->num_rows());

    switch (m_datatype->id()) {
        case Type::DOUBLE: {
            auto train = m_df.downcast_vector<arrow::DoubleType>();
            auto test = test_df.downcast_vector<arrow::DoubleType>();
            ChebyshevDistance<arrow::DoubleType> dist(train, test);

            auto x = std::static_pointer_cast<arrow::DoubleArray>(x_data)->raw_values();
            auto y = std::static_pointer_cast<arrow::DoubleArray>(y_data)->raw_values();

            for (int i = 0; i < test_df->num_rows(); ++i) {
                auto c = count_ball_subspaces_instance<arrow::DoubleType>(test, x, y, i, dist, eps(i));

                count_xz(i) = std::get<0>(c);
                count_yz(i) = std::get<1>(c);
                count_z(i) = std::get<2>(c);
            }
            break;
        }
        case Type::FLOAT: {
            auto train = m_df.downcast_vector<arrow::FloatType>();
            auto test = test_df.downcast_vector<arrow::FloatType>();
            ChebyshevDistance<arrow::FloatType> dist(train, test);

            auto x = std::static_pointer_cast<arrow::FloatArray>(x_data)->raw_values();
            auto y = std::static_pointer_cast<arrow::FloatArray>(y_data)->raw_values();

            for (int i = 0; i < test_df->num_rows(); ++i) {
                auto c = count_ball_subspaces_instance<arrow::FloatType>(test, x, y, i, dist, eps(i));

                count_xz(i) = std::get<0>(c);
                count_yz(i) = std::get<1>(c);
                count_z(i) = std::get<2>(c);
            }
            break;
        }
        default:
            throw std::invalid_argument("Wrong data type to apply KDTree.");
    }
    return std::make_tuple(count_xz, count_yz, count_z);
}

}  // namespace kdtree
